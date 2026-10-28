"""Full loop with simulated MCP and LLM (no network or keys)."""

import asyncio
import json

from fiable_agent.agent import FiableAgent
from fiable_agent.config import load_config
from fiable_agent.llm import ScriptedLLM, text_response, tool_response
from fiable_agent.tools import FakeGateway, LocalToolRoots, ToolSpec

PAYER_TOOLS = [ToolSpec(n, f"doc for {n}") for n in (
    "review.get_md_review_data", "review.submit_md_case_decision", "determination.finalize",
    "payer_letter_center.generate_denial", "payer_letter_center.generate_approval", "triage.get")]

INSTRUCTION = ("# MD Review\nEscalated to you for MD decision on case CASE-9. Record the MD decision and "
               "finalize the determination yourself.\n- Use only payer namespaces: `review`, `determination`")

BAD_MD = {"case_id": "CASE-9", "decision": "deny", "determination_rationale": "criteria not met"}
GOOD_MD = {**BAD_MD, "signed_off": True, "denial_rationale": "BMI 38 without comorbidity (criterion 1.a)",
           "clinical_notes": "BMI 38, no comorbidities documented"}


def payer_gateway():
    return FakeGateway(PAYER_TOOLS, {
        "review.get_md_review_data": lambda a: (True, {"case_id": "CASE-9", "nurse_recommendation": "escalate"}),
    })


def scripted_agent():
    return ScriptedLLM([
        tool_response([("t1", "review__get_md_review_data", {"case_id": "CASE-9"})]),
        tool_response([("t2", "review__submit_md_case_decision", BAD_MD)]),
        tool_response([("t3", "review__submit_md_case_decision", GOOD_MD)]),
        tool_response([("t4", "determination__finalize", {"case_id": "CASE-9", "decision": "denied"})]),
        text_response("All done."),
        tool_response([("t5", "payer_letter_center__generate_denial", {"case_id": "CASE-9"})]),
        text_response("Denied per criterion 1.a; signed MD decision, determination finalized, denial letter issued."),
    ])


def make_agent(profile, llm, gateway, tmp_path, critic_llm=None):
    cfg = load_config(profile, overrides={"rag": {"handbook_dir": str(tmp_path / "no-handbook")}})
    return FiableAgent(cfg, llm, gateway, logs_dir=tmp_path / "logs", critic_llm=critic_llm,
                       local_roots=LocalToolRoots(handbook=str(tmp_path), readable=(str(tmp_path),),
                                                  writable=(str(tmp_path),)))


def test_full_pipeline_blocks_fixes_and_gates_finish(tmp_path):
    gateway = payer_gateway()
    llm = scripted_agent()
    critic = ScriptedLLM(default=text_response('{"allow": true, "reason": "supported"}'))
    agent = make_agent("default", llm, gateway, tmp_path, critic_llm=critic)
    result = asyncio.run(agent.run(INSTRUCTION, "http://localhost:8100/mcp"))

    assert result.role == "payer" and result.error is None and result.stopped_reason == "end_turn"
    assert result.blocked_actions == 1
    names = [n for n, _ in gateway.calls]
    assert names == ["review.get_md_review_data", "review.submit_md_case_decision",
                     "determination.finalize", "payer_letter_center.generate_denial"]
    assert gateway.calls[1][1]["signed_off"] is True  # the unsigned version never got through

    # The LLM saw the block and the finish warning
    tool_results = [b for req in llm.requests for m in req["messages"] if m["role"] == "user"
                    and isinstance(m["content"], list) for b in m["content"] if b.get("type") == "tool_result"]
    assert any("BLOCKED_BY_INTERNAL_CRITIC" in str(b["content"]) for b in tool_results)
    assert any("generate_denial" in str(m.get("content")) for m in llm.requests[-1]["messages"]
               if m["role"] == "user")

    # Sanitized names, prompt cache on system/tools/last message
    req = llm.requests[0]
    assert all("." not in t["name"] for t in req["tools"])
    assert req["system"][0]["cache_control"] == {"type": "ephemeral"}
    assert req["tools"][-1]["cache_control"] == {"type": "ephemeral"}
    assert req["messages"][-1]["content"][-1]["cache_control"] == {"type": "ephemeral"}
    assert req["model"] == "anthropic/claude-sonnet-5"

    # Cost / call ledger
    t = result.totals
    assert t["llm_calls_by_role"] == {"agent": 7, "critic": 4}
    assert t["tool_calls_by_kind"] == {"mcp": 4, "local": 0, "blocked": 1}
    assert t["cost_usd"] is not None and t["cost_usd"] > 0

    logs = tmp_path / "logs"
    rr = json.loads((logs / "run_result.json").read_text(encoding="utf-8"))
    assert rr["turns"] == 7 and rr["model"] == "anthropic/claude-sonnet-5" and rr["error"] is None
    records = [json.loads(line) for line in (logs / "trace.jsonl").read_text(encoding="utf-8").splitlines()]
    assert records[0]["role"] == "system" and records[1]["role"] == "user"
    calls = [c["name"] for r in records if r["role"] == "assistant" for c in r["tool_calls"]]
    assert "review.submit_md_case_decision" in calls  # trajectory with canonical names
    assert sum(1 for r in records if r["role"] == "tool") == 5
    critic_log = json.loads((logs / "critic_log.json").read_text(encoding="utf-8"))
    assert critic_log[0]["allow"] is False and "signed_off" in critic_log[0]["reason"]
    assert "api_key" not in (logs / "agent_config.json").read_text(encoding="utf-8").lower()


def test_baseline_profile_passes_everything_through(tmp_path):
    gateway = payer_gateway()
    llm = ScriptedLLM([
        tool_response([("t1", "review__submit_md_case_decision", BAD_MD)]),
        text_response("done"),
    ])
    result = asyncio.run(make_agent("e1_baseline", llm, gateway, tmp_path).run(INSTRUCTION))
    assert result.blocked_actions == 0 and result.stopped_reason == "end_turn"
    assert gateway.calls[0][1] == BAD_MD
    assert result.totals["llm_calls_by_role"] == {"agent": 2}


def test_checklist_only_profile_blocks_without_llm_critic(tmp_path):
    gateway = payer_gateway()
    llm = ScriptedLLM([
        tool_response([("t1", "review__submit_md_case_decision", BAD_MD)]),
        tool_response([("t2", "review__submit_md_case_decision", GOOD_MD)]),
        text_response("done"),
    ])
    result = asyncio.run(make_agent("e3_checklist", llm, gateway, tmp_path).run(INSTRUCTION))
    assert result.blocked_actions == 1 and [n for n, _ in gateway.calls] == ["review.submit_md_case_decision"]
    assert "critic" not in result.totals["llm_calls_by_role"]


def test_llm_error_is_recorded_and_outputs_written(tmp_path):
    class Broken:
        def create(self, **kw):
            raise ConnectionError("network unreachable")

    result = asyncio.run(make_agent("default", Broken(), payer_gateway(), tmp_path).run(INSTRUCTION))
    assert result.stopped_reason == "error" and "ConnectionError" in result.error
    rr = json.loads((tmp_path / "logs" / "run_result.json").read_text(encoding="utf-8"))
    assert rr["error"].startswith("ConnectionError") and rr["input_tokens"] == 0


def test_max_turns_stop(tmp_path):
    llm = ScriptedLLM(default=tool_response([("t", "triage__get", {"case_id": "CASE-9"})]))
    agent = make_agent("e1_baseline", llm, payer_gateway(), tmp_path)
    agent.cfg.loop.max_turns = 3
    result = asyncio.run(agent.run(INSTRUCTION))
    assert result.turns == 3 and result.stopped_reason == "max_turns"

