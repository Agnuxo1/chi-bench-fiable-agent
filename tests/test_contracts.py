import json

from fiable_agent.contracts import (
    Checklist,
    care_plan_structure_problems,
    is_mutating,
    update_case_lock,
)
from fiable_agent.roles import ROLE_CM, ROLE_PAYER, ROLE_PROVIDER
from fiable_agent.state import ObservedState


def ids(result):
    return {v.rule_id for v in result.blocking}


def ok(state, name, args=None, payload=None):
    state.record(name, args or {}, True, json.dumps(payload if payload is not None else {"ok": True}))
    update_case_lock(state, name, args or {}, True)


GOOD_ORDER = {"patient_id": "PAT-1", "ordering_practitioner_id": "PRAC-1", "service_type": "surgical",
              "site_of_service": "outpatient_hospital", "procedure_code": "43775",
              "diagnosis_codes": ["E66.01"], "description": "sleeve", "priority": "standard"}


# ---------------------------------------------------------------- provider
def test_provider_create_case_enums():
    cl, st = Checklist(ROLE_PROVIDER), ObservedState()
    assert cl.check_action("cases.create_from_order", GOOD_ORDER, st).passed
    bad = {**GOOD_ORDER, "site_of_service": "hospital", "diagnosis_codes": []}
    res = cl.check_action("cases.create_from_order", bad, st)
    assert "PROV-01a" in ids(res) and len(res.blocking) == 2


def test_provider_non_submit_needs_case_and_reason():
    cl, st = Checklist(ROLE_PROVIDER), ObservedState()
    res = cl.check_action("auth.finalize_provider_determination",
                          {"case_id": "CASE-1", "final_action": "do_not_submit"}, st)
    assert {"PROV-01b", "PROV-02/03/05"} <= ids(res)
    ok(st, "cases.create_from_order", GOOD_ORDER, {"case": {"id": "CASE-1"}})
    assert st.locked_case_id == "CASE-1"
    res = cl.check_action("auth.finalize_provider_determination",
                          {"case_id": "CASE-1", "final_action": "gather_more_evidence",
                           "override_reason": "Chart lacks 6 months of documented conservative therapy."}, st)
    assert res.passed


def test_provider_submit_path_and_finish_gate():
    cl, st = Checklist(ROLE_PROVIDER), ObservedState()
    ok(st, "cases.create_from_order", GOOD_ORDER, {"case": {"id": "CASE-1"}})
    res = cl.check_action("auth.finalize_provider_determination", {"case_id": "CASE-1", "final_action": "submit_pa"}, st)
    assert "PROV-02/03/05" in ids(res)  # missing bundle
    assert "PROV-04" in ids(cl.check_finish(st))  # no disposition
    ok(st, "docs.create_submission_bundle", {"case_id": "CASE-1", "document_ids": ["DOC-1"]})
    args = {"case_id": "CASE-1", "final_action": "submit_pa", "bundle_document_id": "DOC-B"}
    assert cl.check_action("auth.finalize_provider_determination", args, st).passed
    ok(st, "auth.finalize_provider_determination", args)
    assert "PROV-04" in ids(cl.check_finish(st))  # submit_pa without submission
    assert "PROV-03b" in ids(cl.check_action("auth.submit_authorization", {"case_id": "CASE-1"}, st))
    ok(st, "auth.submit_authorization", {"case_id": "CASE-1", "channel": "portal", "bundle_document_id": "DOC-B"})
    assert cl.check_finish(st).passed


def test_case_lock_blocks_other_cases():
    cl, st = Checklist(ROLE_PROVIDER), ObservedState()
    ok(st, "cases.create_from_order", GOOD_ORDER, {"case": {"id": "CASE-1"}})
    res = cl.check_action("docs.upload_evidence", {"case_id": "CASE-2", "kind": "chart_note"}, st)
    assert "GLOBAL-01" in ids(res)
    assert not is_mutating("chart.get_patient_chart_by_patient")


# ------------------------------------------------------------------ payer
def test_md_decision_requires_signoff_and_rationales():
    cl, st = Checklist(ROLE_PAYER), ObservedState()
    res = cl.check_action("review.submit_md_case_decision", {"case_id": "CASE-9", "decision": "deny"}, st)
    msgs = " ".join(v.message for v in res.blocking)
    assert "signed_off" in msgs and "determination_rationale" in msgs and "denial_rationale" in msgs
    good = {"case_id": "CASE-9", "decision": "deny", "signed_off": True,
            "determination_rationale": "BMI 38 without comorbidity fails criterion 1.a",
            "denial_rationale": "Criterion 1.a not met", "clinical_notes": "BMI 38"}
    assert cl.check_action("review.submit_md_case_decision", good, st).passed
    p2p = {"case_id": "CASE-9", "decision": "p2p_review"}
    assert cl.check_action("review.submit_md_case_decision", p2p, st).passed


def test_nurse_cannot_deny():
    cl, st = Checklist(ROLE_PAYER), ObservedState()
    res = cl.check_action("review.submit_nurse_case_recommendation",
                          {"case_id": "C", "recommendation": "deny", "clinical_notes": "x"}, st)
    assert "UM-01" in ids(res)


def test_finalize_must_match_upstream_unless_overridden():
    cl, st = Checklist(ROLE_PAYER), ObservedState()
    ok(st, "review.submit_md_case_decision", {"case_id": "C", "decision": "approve", "signed_off": True,
                                              "determination_rationale": "met"})
    assert "UM-03" in ids(cl.check_action("determination.finalize", {"case_id": "C", "decision": "denied"}, st))
    assert cl.check_action("determination.finalize", {"case_id": "C", "decision": "approved"}, st).passed
    over = {"case_id": "C", "decision": "denied", "overridden": True, "override_reason": "eligibility termed"}
    assert cl.check_action("determination.finalize", over, st).passed


def test_finalize_after_p2p_review_requires_outcome():
    cl, st = Checklist(ROLE_PAYER), ObservedState()
    ok(st, "review.submit_md_case_decision", {"case_id": "C", "decision": "p2p_review"})
    assert "UM-03" in ids(cl.check_action("determination.finalize", {"case_id": "C", "decision": "denied"}, st))
    assert "UM-06b" in ids(cl.check_action("p2p.record_outcome", {"request_id": "P2P-1", "outcome": "x"}, st))
    ok(st, "p2p.offer_slots", {"request_id": "P2P-1"})
    ok(st, "p2p.schedule", {"request_id": "P2P-1"})
    ok(st, "p2p.record_outcome", {"request_id": "P2P-1", "outcome": "uphold_intended_adverse_decision"})
    assert cl.check_action("determination.finalize", {"case_id": "C", "decision": "denied"}, st).passed


def test_finish_requires_matching_letter():
    cl, st = Checklist(ROLE_PAYER), ObservedState()
    ok(st, "determination.finalize", {"case_id": "C", "decision": "denied"})
    res = cl.check_finish(st)
    assert "UM-04" in ids(res) and "generate_denial" in res.render()
    ok(st, "payer_letter_center.generate_approval", {"case_id": "C"})  # wrong letter
    assert not cl.check_finish(st).passed
    ok(st, "payer_letter_center.generate_denial", {"case_id": "C"})
    assert cl.check_finish(st).passed


def test_pend_path_requires_pend_letter():
    cl, st = Checklist(ROLE_PAYER), ObservedState()
    ok(st, "review.submit_nurse_case_recommendation",
       {"case_id": "C", "recommendation": "request_info", "info_requested": "PT notes"})
    assert "generate_pend" in cl.check_finish(st).render()
    ok(st, "payer_letter_center.generate_pend", {"case_id": "C"})
    assert cl.check_finish(st).passed


# --------------------------------------------------------- case management
GOOD_PLAN = {
    "problems": [{"problem_id": "P1", "description": "AFib symptom burden", "category": "clinical",
                  "goals": [{"goal_id": "G1", "description": "fewer ED visits", "target_date": "2026-07-01",
                             "measurable_criteria": "0 ED visits for palpitations in 90 days",
                             "interventions": [{"intervention_id": "I1", "description": "pulse log coaching",
                                                "owner": "care_manager", "frequency": "weekly"}]}]}],
    "escalation_conditions": ["HR > 120 at rest -> call PCP"],
    "follow_up_cadence_kind": "weekly",
}


def test_care_plan_structure_mirrors_cm_v4():
    assert care_plan_structure_problems(GOOD_PLAN) == []
    bad = json.loads(json.dumps(GOOD_PLAN))
    goal = bad["problems"][0]["goals"][0]
    goal["target_date"] = None
    goal["interventions"][0].pop("frequency")
    goal["interventions"][0]["owner"] = ""
    bad["escalation_conditions"] = []
    issues = " ".join(care_plan_structure_problems(bad))
    for needle in ("target_date", "owner", "due_date or frequency", "escalation_conditions"):
        assert needle in issues


def test_cm_order_sections_and_finalize():
    cl, st = Checklist(ROLE_CM), ObservedState()
    review = {"case_id": "CM-CASE-1", "review": {"preliminary_risk_level": "moderate",
                                                 "key_findings": "GAD-7 12, three ED visits in 14 months.",
                                                 "primary_concerns": ["AFib"]}}
    assert "CM-01a" in ids(cl.check_action("cm_chart.submit_chart_review", review, st))
    ok(st, "cm_intake.open_case", {"referral_id": "REF-1", "program": "chronic_disease", "priority": "standard"},
       {"case_id": "CM-CASE-1"})
    assert st.locked_case_id == "CM-CASE-1"
    assert cl.check_action("cm_chart.submit_chart_review", review, st).passed
    for section in ("clinical", "medication"):
        ok(st, "cm_assessment.save_section", {"case_id": "CM-CASE-1", "payload": {"section": section}})
    res = cl.check_action("cm_assessment.complete_assessment", {"case_id": "CM-CASE-1"}, st)
    assert "CM-02" in ids(res) and "behavioral_health" in res.render()
    for section in ("behavioral_health", "sdoh"):
        ok(st, "cm_assessment.save_section", {"case_id": "CM-CASE-1", "payload": {"section": section}})
    assert cl.check_action("cm_assessment.complete_assessment", {"case_id": "CM-CASE-1"}, st).passed
    ok(st, "cm_assessment.complete_assessment", {"case_id": "CM-CASE-1"})
    ok(st, "cm_care_plan.create_draft", {"case_id": "CM-CASE-1"})
    bad_plan = json.loads(json.dumps(GOOD_PLAN))
    bad_plan["follow_up_cadence_kind"] = ""
    ok(st, "cm_care_plan.save_care_plan", {"case_id": "CM-CASE-1", "plan_data": bad_plan})
    assert "CM-03b" in ids(cl.check_action("cm_care_plan.finalize", {"case_id": "CM-CASE-1"}, st))
    assert "CM-06" in ids(cl.check_finish(st))
    ok(st, "cm_care_plan.save_care_plan", {"case_id": "CM-CASE-1", "plan_data": GOOD_PLAN})
    assert cl.check_action("cm_care_plan.finalize", {"case_id": "CM-CASE-1"}, st).passed
    ok(st, "cm_care_plan.finalize", {"case_id": "CM-CASE-1"})
    assert cl.check_finish(st).passed


def test_consent_only_with_consent_exchange():
    cl, st = Checklist(ROLE_CM), ObservedState()
    ok(st, "cm_outreach.start_outreach_call", {"case_id": "CM-CASE-1"}, {"session_id": "SES-1"})
    args = {"session_id": "SES-1", "summary": {"engagement_status": "engaged", "consent_obtained": True}}
    assert "CM-01b/04" in ids(cl.check_action("cm_outreach.end_outreach_call", args, st))
    ok(st, "cm_outreach.send_message", {"session_id": "SES-1", "message": "Hello, how are you feeling?"})
    assert "CM-01b/04" in ids(cl.check_action("cm_outreach.end_outreach_call", args, st))
    ok(st, "cm_outreach.send_message", {"session_id": "SES-1",
                                        "message": "Would it be okay if we talk about your ED visits? Do you agree "
                                                   "to enroll?"})
    assert cl.check_action("cm_outreach.end_outreach_call", args, st).passed


def test_describe_lists_role_rules_only():
    text = Checklist(ROLE_CM).describe()
    assert "CM-03b" in text and "UM-02" not in text and "GLOBAL-01" in text

