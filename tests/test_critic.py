import json

from fiable_agent.contracts import Checklist
from fiable_agent.critic import (
    ContractJudge,
    CriticContext,
    CriticPanel,
    GroundingJudge,
    PolicyJudge,
    ProposedAction,
    build_panel,
)
from fiable_agent.ledger import CostLedger
from fiable_agent.llm import ScriptedLLM, text_response
from fiable_agent.roles import ROLE_PAYER, ROLE_PROVIDER
from fiable_agent.state import ObservedState


def ctx_for(role, tool, args, state=None, instruction="", rag=""):
    state = state or ObservedState()
    return CriticContext(role, instruction, state, Checklist(role).check_action(tool, args, state), rag)


def test_contract_judge_blocks_with_reason():
    args = {"case_id": "C", "decision": "approve"}
    v = ContractJudge().review(ProposedAction("review.submit_md_case_decision", args),
                               ctx_for(ROLE_PAYER, "review.submit_md_case_decision", args))
    assert not v.allow and "signed_off" in v.reason and v.deterministic


def test_grounding_judge_requires_codes_in_evidence():
    st = ObservedState()
    st.record("chart.get_patient_chart_by_patient", {"patient_id": "PAT-1"}, True,
              json.dumps({"patient": {"id": "PAT-1"}, "conditions": [{"code": "E66.01"}],
                          "orders": [{"id": "SR-77", "code": "43775"}]}))
    args = {"patient_id": "PAT-1", "procedure_code": "43775", "diagnosis_codes": ["E66.01", "Z68.41"]}
    v = GroundingJudge().review(ProposedAction("cases.create_from_order", args),
                                ctx_for(ROLE_PROVIDER, "cases.create_from_order", args, st))
    assert not v.allow and "Z68.41" in v.reason
    args["diagnosis_codes"] = ["E66.01"]
    v = GroundingJudge().review(ProposedAction("cases.create_from_order", args),
                                ctx_for(ROLE_PROVIDER, "cases.create_from_order", args, st))
    assert v.allow


def test_policy_judge_llm_block_and_cost_logged():
    llm = ScriptedLLM([text_response('{"allow": false, "reason": "rationale cites no criterion"}',
                                     {"input_tokens": 900, "output_tokens": 30})])
    ledger = CostLedger()
    judge = PolicyJudge(llm, "anthropic/claude-sonnet-5", ledger)
    v = judge.review(ProposedAction("determination.finalize", {"case_id": "C", "decision": "denied"}),
                     ctx_for(ROLE_PAYER, "determination.finalize", {"case_id": "C", "decision": "denied"}))
    assert not v.allow and "criterion" in v.reason and not v.deterministic
    assert ledger.totals()["llm_calls_by_role"] == {"critic": 1}
    assert "compliance reviewer" in llm.requests[0]["system"]


def test_policy_judge_fail_open_on_garbage_and_errors():
    judge = PolicyJudge(ScriptedLLM([text_response("I think it's fine")]), "m")
    action = ProposedAction("determination.finalize", {"case_id": "C", "decision": "approved"})
    assert judge.review(action, ctx_for(ROLE_PAYER, action.tool, action.args)).allow

    class Boom:
        def create(self, **kw):
            raise TimeoutError("api down")

    v = PolicyJudge(Boom(), "m").review(action, ctx_for(ROLE_PAYER, action.tool, action.args))
    assert v.allow and v.severity == "warn"


def test_policy_heuristic_mode_blocks_thin_rationale():
    judge = PolicyJudge(None)
    thin = ProposedAction("auth.finalize_provider_determination",
                          {"case_id": "C", "final_action": "do_not_submit", "override_reason": "not needed"})
    assert not judge.review(thin, ctx_for(ROLE_PROVIDER, thin.tool, thin.args)).allow


def test_panel_majority_and_forced_override():
    llm = ScriptedLLM(default=text_response('{"allow": false, "reason": "not supported"}'))
    panel = build_panel(["contract", "grounding", "policy"], llm=llm, model="m", ledger=None,
                        max_blocks_per_action=2)
    args = {"case_id": "C", "decision": "approved"}
    action = ProposedAction("determination.finalize", args)
    decisions = [panel.decide(action, ctx_for(ROLE_PAYER, action.tool, args)) for _ in range(3)]
    assert [d.allow for d in decisions] == [False, False, True]
    assert decisions[-1].forced and "policy: not supported" in decisions[0].reason
    assert len(panel.log) == 3


def test_panel_allows_clean_action():
    panel = CriticPanel([ContractJudge(), GroundingJudge(), PolicyJudge(None)])
    args = {"case_id": "C", "decision": "approved"}
    st = ObservedState()
    st.record("review.get_md_review_data", {"case_id": "C"}, True, '{"case_id": "C"}')
    d = panel.decide(ProposedAction("determination.finalize", args),
                     ctx_for(ROLE_PAYER, "determination.finalize", args, st))
    assert d.allow and not d.forced

