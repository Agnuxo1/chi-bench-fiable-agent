"""Per-role planner.

The role is detected from the exposed MCP tools (reliable), the MCP port
(task.toml: 8020 provider, 8100 payer, 8200 CM) or the instruction.
The phases follow the sequence of the trusted oracle
(repo/chi-bench/src/chi_bench/verifier/reference_oracle.py) and the
conventions of tool_reference.md; they contain no answers to any task.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass, field

from .state import ObservedState

ROLE_PROVIDER = "provider"
ROLE_PAYER = "payer"
ROLE_CM = "care_manager"
ROLES = (ROLE_PROVIDER, ROLE_PAYER, ROLE_CM)

PORT_ROLE = {8020: ROLE_PROVIDER, 8100: ROLE_PAYER, 8200: ROLE_CM}

NAMESPACE_ROLE = {
    "cases": ROLE_PROVIDER,
    "chart": ROLE_PROVIDER,
    "auth": ROLE_PROVIDER,
    "docs": ROLE_PROVIDER,
    "forms": ROLE_PROVIDER,
    "inbox": ROLE_PROVIDER,
    "people": ROLE_PROVIDER,
    "payer_intake_hub": ROLE_PAYER,
    "triage": ROLE_PAYER,
    "review": ROLE_PAYER,
    "determination": ROLE_PAYER,
    "payer_letter_center": ROLE_PAYER,
    "p2p": ROLE_PAYER,
    "cm_intake": ROLE_CM,
    "cm_chart": ROLE_CM,
    "cm_outreach": ROLE_CM,
    "cm_assessment": ROLE_CM,
    "cm_care_plan": ROLE_CM,
}


def detect_role(
    instruction: str = "",
    mcp_url: str | None = None,
    tool_names: Iterable[str] | None = None,
) -> str:
    votes: dict[str, int] = {r: 0 for r in ROLES}
    for name in tool_names or ():
        namespace = str(name).split(".", 1)[0].split("__", 1)[0]
        role = NAMESPACE_ROLE.get(namespace)
        if role:
            votes[role] += 1
    if any(votes.values()):
        return max(votes, key=votes.get)
    if mcp_url:
        match = re.search(r":(\d{2,5})(?:/|$)", mcp_url)
        if match and int(match.group(1)) in PORT_ROLE:
            return PORT_ROLE[int(match.group(1))]
    text = (instruction or "").lower()
    if "care-manager namespaces" in text or "cm_intake" in text or "care management" in text:
        return ROLE_CM
    if "payer namespaces" in text or "payer_intake_hub" in text:
        return ROLE_PAYER
    if "provider namespaces" in text:
        return ROLE_PROVIDER
    return ROLE_PROVIDER


@dataclass(frozen=True)
class Phase:
    name: str
    goal: str
    tools: tuple[str, ...]
    done_when: tuple[str, ...] = field(default_factory=tuple)
    optional: bool = False


ROLE_PLANS: dict[str, tuple[Phase, ...]] = {
    ROLE_PROVIDER: (
        Phase("orient", "Read the handbook SKILL.md routing, the instruction, and the patient chart/orders.",
              ("chart.search_patients", "chart.get_patient_chart_by_patient", "chart.list_candidate_orders",
               "inbox.list_messages"), ("chart.get_patient_chart_by_patient",)),
        Phase("case", "Create exactly one PA case from the right order with grounded procedure/diagnosis codes, "
              "service_type and site_of_service.", ("cases.create_from_order",), ("cases.create_from_order",)),
        Phase("requirements", "Check payer requirements/forms and the governing medical policy (handbook "
              "references/medical-library). Decide: submit_pa, gather_more_evidence or do_not_submit.",
              ("forms.list_required_forms", "docs.list_case_documents"), ()),
        Phase("evidence", "Only if submitting: upload grounded evidence, save form answers, build the "
              "submission bundle.", ("docs.upload_evidence", "forms.save_form_response",
                                     "docs.create_submission_bundle"), (), optional=True),
        Phase("disposition", "Record the provider disposition with auth.finalize_provider_determination "
              "(with a substantive override_reason).", ("auth.finalize_provider_determination",),
              ("auth.finalize_provider_determination",)),
        Phase("submit", "Only after submit_pa: auth.submit_authorization with the bundle.",
              ("auth.submit_authorization",), ("auth.submit_authorization",), optional=True),
    ),
    ROLE_PAYER: (
        Phase("orient", "Identify your seat (intake / triage / nurse / MD / P2P) from the instruction; load the "
              "case, request documents and the governing policy.",
              ("payer_intake_hub.list_intake_queue", "payer_intake_hub.list_documents", "triage.get",
               "review.get_nurse_review_data", "review.get_md_review_data", "determination.get_summary"), ()),
        Phase("intake", "If intake seat: create/complete intake (coverage, provider, diagnoses, lines) and "
              "confirm the four intake decisions.", ("payer_intake_hub.create_intake_case",
                                                     "payer_intake_hub.confirm_decisions"),
              ("payer_intake_hub.confirm_decisions",), optional=True),
        Phase("triage", "If triage applies: calculate SLA, set disposition, then route the case.",
              ("triage.calculate_sla", "triage.check_gold_card", "triage.set_disposition", "triage.route_case"),
              ("triage.route_case",), optional=True),
        Phase("nurse_review", "If nurse review applies: save per-line criteria evaluations, then submit the "
              "nurse recommendation (approve / escalate / request_info; never deny).",
              ("review.save_criteria_evaluation", "review.submit_nurse_case_recommendation"),
              ("review.submit_nurse_case_recommendation",), optional=True),
        Phase("md_review", "If MD review applies: save criteria, then submit the MD decision with "
              "determination_rationale and signed_off=true for finalizing decisions.",
              ("review.save_md_review", "review.submit_md_case_decision"),
              ("review.submit_md_case_decision",), optional=True),
        Phase("p2p", "If a peer-to-peer is pending: offer slots, schedule, hold the session, record outcome.",
              ("p2p.list", "p2p.offer_slots", "p2p.schedule", "p2p_session.send_turn",
               "p2p_session.submit_summary", "p2p.record_outcome"), (), optional=True),
        Phase("determination", "Finalize the determination consistent with the upstream decision (unless "
              "pended).", ("determination.finalize",), ("determination.finalize",), optional=True),
        Phase("letters", "Generate (and deliver if required) the provider letter matching the outcome: "
              "approval / denial / partial approval / pend.",
              ("payer_letter_center.generate_approval", "payer_letter_center.generate_denial",
               "payer_letter_center.generate_partial_approval", "payer_letter_center.generate_pend",
               "payer_letter_center.deliver"), ()),
    ),
    ROLE_CM: (
        Phase("intake", "Find the referral and open the CM case with the right program and priority.",
              ("cm_intake.list_referral_queue", "cm_intake.get_referral", "cm_intake.open_case"),
              ("cm_intake.open_case",)),
        Phase("chart_review", "Review chart data and submit a chart review with numeric anchors, risk level "
              "per the handbook risk-stratification rules and substantive key_findings.",
              ("cm_chart.get_chart_data", "cm_chart.submit_chart_review"), ("cm_chart.submit_chart_review",)),
        Phase("outreach", "Call the patient: ask permission/consent before sensitive topics, address concerns, "
              "then end the call with an honest summary (consent_obtained only if it happened).",
              ("cm_outreach.start_outreach_call", "cm_outreach.send_message", "cm_outreach.get_transcript",
               "cm_outreach.end_outreach_call"), ("cm_outreach.end_outreach_call",)),
        Phase("assessment", "Start the assessment, save all four sections (clinical, medication, "
              "behavioral_health, sdoh) grounded in chart + transcript, then complete it.",
              ("cm_assessment.start_assessment", "cm_assessment.save_section", "cm_assessment.complete_assessment"),
              ("cm_assessment.complete_assessment",)),
        Phase("care_plan", "Create the draft, save a complete plan (problems -> goals with target_date and "
              "measurable_criteria -> interventions with owner and due_date/frequency, escalation conditions, "
              "follow-up cadence) and finalize.",
              ("cm_care_plan.create_draft", "cm_care_plan.save_care_plan", "cm_care_plan.finalize"),
              ("cm_care_plan.finalize",)),
    ),
}


def phase_status(role: str, state: ObservedState) -> list[tuple[Phase, bool]]:
    out = []
    for phase in ROLE_PLANS[role]:
        done = bool(phase.done_when) and all(state.has(t) for t in phase.done_when)
        out.append((phase, done))
    return out


def render_plan(role: str, state: ObservedState | None = None) -> str:
    state = state or ObservedState()
    lines = [f"ROLE PLAN ({role}):"]
    for i, (phase, done) in enumerate(phase_status(role, state), 1):
        mark = "[x]" if done else "[ ]"
        opt = " (if applicable)" if phase.optional else ""
        lines.append(f"{mark} {i}. {phase.name}{opt}: {phase.goal}  Tools: {', '.join(phase.tools)}")
    return "\n".join(lines)

