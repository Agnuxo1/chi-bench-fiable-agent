"""Contract checklist - evaluated BEFORE every terminal/irreversible action
and before the task is considered finished.

Each rule is traced to a rule id (contracts analysis, section 6) and to the
verifier checks it protects. Rules only use information visible to the agent
(its own calls, tool docstrings and the instruction); never hidden
expectations.

Severity:
  * ``block``: the action is not sent to the MCP; the agent receives the reason.
  * ``warn``:  it is sent, but the warning reaches the critic and the log.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from .roles import ROLE_CM, ROLE_PAYER, ROLE_PROVIDER
from .state import ObservedState

FINISH = "__finish__"

# --- tool sets -------------------------------------------------

TERMINAL_TOOLS: dict[str, frozenset[str]] = {
    ROLE_PROVIDER: frozenset({
        "cases.create_from_order",
        "docs.create_submission_bundle",
        "auth.finalize_provider_determination",
        "auth.submit_authorization",
        "auth.request_peer_to_peer",
        "auth.respond_to_pend",
        "auth.file_appeal",
        "auth.file_external_appeal",
    }),
    ROLE_PAYER: frozenset({
        "payer_intake_hub.confirm_decisions",
        "triage.route_case",
        "review.submit_nurse_case_recommendation",
        "review.submit_md_case_decision",
        "determination.finalize",
        "p2p.record_outcome",
        "p2p_session.submit_summary",
        "payer_letter_center.generate_approval",
        "payer_letter_center.generate_denial",
        "payer_letter_center.generate_partial_approval",
        "payer_letter_center.generate_pend",
        "payer_letter_center.generate_notification",
        "payer_letter_center.deliver",
    }),
    ROLE_CM: frozenset({
        "cm_intake.open_case",
        "cm_intake.disenroll_case",
        "cm_chart.submit_chart_review",
        "cm_outreach.end_outreach_call",
        "cm_assessment.complete_assessment",
        "cm_care_plan.finalize",
    }),
}

_MUTATING_PREFIXES = ("save_", "add_", "update_", "remove_", "create_", "submit_", "set_",
                      "route_", "upload_", "finalize", "confirm_", "generate_", "deliver",
                      "record_", "offer_", "schedule", "start_", "end_", "complete_",
                      "open_", "disenroll", "file_", "request_", "respond_")


def is_mutating(tool: str) -> bool:
    method = tool.split(".", 1)[-1]
    return method.startswith(_MUTATING_PREFIXES)


# Enums copied from the docstrings / tool_reference.md of runtime 1.1.0.
SERVICE_TYPES = frozenset({"medical", "surgical", "behavioral_health", "rehab_therapy", "radiology",
                           "cardiology", "dme", "home_health", "skilled_nursing", "transplant",
                           "oncology", "pain_management", "other"})
SITES_OF_SERVICE = frozenset({"office", "inpatient_hospital", "outpatient_hospital",
                              "ambulatory_surgical_center", "emergency_room", "skilled_nursing_facility",
                              "home", "telehealth", "urgent_care", "rehabilitation_facility", "hospice",
                              "other"})
PROVIDER_FINAL_ACTIONS = frozenset({"submit_pa", "gather_more_evidence", "do_not_submit"})
NON_SUBMIT_ACTIONS = frozenset({"gather_more_evidence", "do_not_submit"})
NURSE_RECOMMENDATIONS = frozenset({"approve", "escalate", "request_info"})
MD_DECISIONS = frozenset({"approve", "deny", "partial_approve", "request_info", "p2p_review"})
MD_FINALIZING = frozenset({"approve", "deny", "partial_approve"})
DETERMINATION_DECISIONS = frozenset({"approved", "denied", "partially_approved"})
MD_TO_FINAL = {"approve": "approved", "deny": "denied", "partial_approve": "partially_approved"}
NURSE_TO_FINAL = {"approve": "approved"}
FINAL_TO_LETTER = {
    "approved": "payer_letter_center.generate_approval",
    "denied": "payer_letter_center.generate_denial",
    "partially_approved": "payer_letter_center.generate_partial_approval",
}
CM_SECTIONS = ("clinical", "medication", "behavioral_health", "sdoh")
RISK_LEVELS = frozenset({"high", "moderate", "low"})
_CONSENT_RE = re.compile(r"\b(consent|permission|is it (ok|okay)|would it be (ok|okay)|may i|do you agree|"
                         r"okay with|ok with|agree to)\b", re.IGNORECASE)


# --- types --------------------------------------------------------------------

@dataclass(frozen=True)
class Finding:
    severity: str  # block | warn
    message: str


@dataclass(frozen=True)
class Violation:
    rule_id: str
    severity: str
    message: str
    protects: tuple[str, ...]


@dataclass
class ChecklistResult:
    tool: str
    violations: list[Violation] = field(default_factory=list)

    @property
    def blocking(self) -> list[Violation]:
        return [v for v in self.violations if v.severity == "block"]

    @property
    def warnings(self) -> list[Violation]:
        return [v for v in self.violations if v.severity == "warn"]

    @property
    def passed(self) -> bool:
        return not self.blocking

    def render(self) -> str:
        if not self.violations:
            return "checklist OK"
        return "\n".join(f"- [{v.severity}] {v.rule_id}: {v.message}" for v in self.violations)


RuleFn = Callable[[str, dict[str, Any], ObservedState], list[Finding]]


@dataclass(frozen=True)
class Rule:
    rule_id: str
    role: str  # role or "*"
    tools: frozenset[str]  # tools that trigger it (or FINISH)
    description: str
    protects: tuple[str, ...]
    fn: RuleFn


def _blank(value: Any) -> bool:
    return value is None or (isinstance(value, str) and not value.strip()) or (
        isinstance(value, (list, dict, tuple)) and not value)


def _s(value: Any) -> str:
    return str(value or "").strip().lower()


def _block(msg: str) -> list[Finding]:
    return [Finding("block", msg)]


def _warn(msg: str) -> list[Finding]:
    return [Finding("warn", msg)]


# --- global rules ----------------------------------------------------------

def _global_case_lock(tool: str, args: dict, state: ObservedState) -> list[Finding]:
    case_id = args.get("case_id")
    if not isinstance(case_id, str) or not case_id.strip() or not is_mutating(tool):
        return []
    locked = state.locked_case_id
    if locked and case_id.strip() != locked:
        return _block(f"this task's target case is {locked}; mutating another case ({case_id}) fails the "
                      "forbidden-mutation checks. Use the target case only.")
    return []


# --- provider -----------------------------------------------------------------

def _prov_create_case(tool: str, args: dict, state: ObservedState) -> list[Finding]:
    out: list[Finding] = []
    if state.has("cases.create_from_order"):
        out += _warn("a PA case was already created in this task; creating a second case usually breaks "
                     "'exactly one service request'. Reuse the existing case unless the first was wrong.")
    if _s(args.get("service_type")) not in SERVICE_TYPES:
        out += _block(f"service_type must be one of {sorted(SERVICE_TYPES)}")
    if _s(args.get("site_of_service")) not in SITES_OF_SERVICE:
        out += _block(f"site_of_service must be one of {sorted(SITES_OF_SERVICE)}")
    codes = args.get("diagnosis_codes")
    if not isinstance(codes, list) or not any(str(c).strip() for c in codes):
        out += _block("diagnosis_codes must list at least one ICD-10 code grounded in the chart")
    if _blank(args.get("procedure_code")):
        out += _block("procedure_code (CPT/HCPCS) is required")
    return out


def _prov_needs_case(tool: str, args: dict, state: ObservedState) -> list[Finding]:
    if not state.has("cases.create_from_order") and not state.has("cases.get_case"):
        return _block("no PA case exists yet: create it with cases.create_from_order first")
    return []


def _prov_disposition(tool: str, args: dict, state: ObservedState) -> list[Finding]:
    action = _s(args.get("final_action"))
    out: list[Finding] = []
    if action not in PROVIDER_FINAL_ACTIONS:
        return _block(f"final_action must be one of {sorted(PROVIDER_FINAL_ACTIONS)}")
    if action in NON_SUBMIT_ACTIONS:
        if _blank(args.get("override_reason")):
            out += _block("non-submit dispositions need a substantive override_reason citing the chart gaps "
                          "and the policy criterion that is not met")
        if state.has("docs.create_submission_bundle") or state.has("auth.submit_authorization"):
            out += _warn("a submission bundle/packet already exists; a non-submit disposition expects no packet")
    if action == "submit_pa":
        if not state.has("docs.create_submission_bundle"):
            out += _block("submit_pa requires a submission bundle: create it with docs.create_submission_bundle")
        elif _blank(args.get("bundle_document_id")):
            out += _block("pass bundle_document_id of the submission bundle you created")
    return out


def _prov_submit(tool: str, args: dict, state: ObservedState) -> list[Finding]:
    last = state.last("auth.finalize_provider_determination")
    if last is None or _s(last.args.get("final_action")) != "submit_pa":
        return _block("record auth.finalize_provider_determination(final_action='submit_pa') before submitting")
    if _blank(args.get("bundle_document_id")):
        return _block("pass bundle_document_id to auth.submit_authorization")
    return []


def _prov_finish(tool: str, args: dict, state: ObservedState) -> list[Finding]:
    last = state.last("auth.finalize_provider_determination")
    if last is None:
        return _block("no provider disposition recorded: call auth.finalize_provider_determination with "
                      "submit_pa, gather_more_evidence or do_not_submit before finishing")
    if _s(last.args.get("final_action")) == "submit_pa" and not state.happened_after(
            "auth.submit_authorization", last):
        return _block("disposition is submit_pa but the PA was not submitted: call auth.submit_authorization")
    return []


# --- payer -------------------------------------------------------------------

def _um_intake_confirm(tool: str, args: dict, state: ObservedState) -> list[Finding]:
    out: list[Finding] = []
    for key in ("request_complete", "member_eligible"):
        if not isinstance(args.get(key), bool):
            out += _block(f"{key} must be an explicit boolean grounded in the intake review")
    if not (state.has("payer_intake_hub.get_intake_case") or state.has("payer_intake_hub.list_documents")):
        out += _warn("confirming intake decisions without having read the intake case/documents")
    return out


def _um_route(tool: str, args: dict, state: ObservedState) -> list[Finding]:
    if not state.has("triage.set_disposition"):
        return _warn("no triage.set_disposition recorded before routing (urgency/SLA/disposition)")
    return []


def _um_nurse(tool: str, args: dict, state: ObservedState) -> list[Finding]:
    rec = _s(args.get("recommendation"))
    out: list[Finding] = []
    if rec in {"deny", "denied"}:
        return _block("a nurse may not deny: use escalate (physician review) or request_info")
    if rec not in NURSE_RECOMMENDATIONS:
        out += _block(f"recommendation must be one of {sorted(NURSE_RECOMMENDATIONS)}")
    if rec == "request_info" and _blank(args.get("info_requested")):
        out += _block("request_info requires info_requested (what is missing)")
    if _blank(args.get("clinical_notes")):
        out += _block("include rationale-rich clinical_notes (criteria met/not met with chart evidence)")
    if not state.has("review.save_criteria_evaluation"):
        out += _warn("no per-line criteria evaluation saved (review.save_criteria_evaluation)")
    return out


def _um_md(tool: str, args: dict, state: ObservedState) -> list[Finding]:
    dec = _s(args.get("decision"))
    out: list[Finding] = []
    if dec not in MD_DECISIONS:
        return _block(f"decision must be one of {sorted(MD_DECISIONS)}")
    if dec in MD_FINALIZING:
        if args.get("signed_off") is not True:
            out += _block("finalizing MD decisions require signed_off=true")
        if _blank(args.get("determination_rationale")):
            out += _block("determination_rationale is required for every finalizing decision")
    if dec == "deny" and _blank(args.get("denial_rationale")):
        out += _block("deny requires denial_rationale")
    if dec == "partial_approve" and _blank(args.get("line_item_decisions")):
        out += _block("partial_approve requires line_item_decisions")
    if dec == "request_info" and (_blank(args.get("info_requested")) or _blank(args.get("info_deadline"))):
        out += _block("request_info requires info_requested and info_deadline")
    if _blank(args.get("clinical_notes")):
        out += _warn("clinical_notes (factual chart summary) is empty")
    return out


def _upstream_decision(state: ObservedState) -> tuple[str | None, str | None]:
    md = state.last("review.submit_md_case_decision")
    if md is not None:
        return "md", _s(md.args.get("decision"))
    nurse = state.last("review.submit_nurse_case_recommendation")
    if nurse is not None:
        return "nurse", _s(nurse.args.get("recommendation"))
    return None, None


def _um_finalize(tool: str, args: dict, state: ObservedState) -> list[Finding]:
    dec = _s(args.get("decision"))
    out: list[Finding] = []
    if dec not in DETERMINATION_DECISIONS:
        return _block(f"decision must be one of {sorted(DETERMINATION_DECISIONS)}")
    source, upstream = _upstream_decision(state)
    if source == "md" and upstream == "p2p_review" and not (
            state.has("p2p.record_outcome") or state.has("p2p_session.submit_summary")):
        out += _block("the MD reserved the decision for peer-to-peer: complete the P2P and record its outcome "
                      "before finalizing")
    if upstream == "request_info":
        out += _warn("upstream pended the case (request_info); a pend usually ends with a pend letter, "
                     "not a final determination")
    expected = None
    if source == "md":
        expected = MD_TO_FINAL.get(upstream or "")
    elif source == "nurse":
        expected = NURSE_TO_FINAL.get(upstream or "")
    if expected and dec != expected and not (args.get("overridden") is True
                                             and not _blank(args.get("override_reason"))):
        out += _block(f"decision '{dec}' differs from the upstream {source} decision ('{upstream}' -> "
                      f"'{expected}'); align it, or set overridden=true with an override_reason")
    return out


def _um_record_p2p(tool: str, args: dict, state: ObservedState) -> list[Finding]:
    out: list[Finding] = []
    if not state.has("p2p.offer_slots"):
        out += _block("offer P2P slots (p2p.offer_slots) before recording an outcome")
    if not state.has("p2p.schedule"):
        out += _block("schedule the P2P (p2p.schedule) before recording an outcome")
    if _blank(args.get("outcome")):
        out += _block("outcome is required")
    return out


def _um_schedule(tool: str, args: dict, state: ObservedState) -> list[Finding]:
    if not state.has("p2p.offer_slots"):
        return _block("offer slots (p2p.offer_slots) before scheduling")
    return []


def _um_finish(tool: str, args: dict, state: ObservedState) -> list[Finding]:
    out: list[Finding] = []
    final = state.last("determination.finalize")
    if final is not None:
        letter_tool = FINAL_TO_LETTER.get(_s(final.args.get("decision")))
        if letter_tool and not state.happened_after(letter_tool, final):
            out += _block(f"the determination is finalized but the provider letter was not generated: call "
                          f"{letter_tool} (finalize does not compose correspondence)")
    pended = state.last_any(("review.submit_md_case_decision", "review.submit_nurse_case_recommendation"))
    if pended is not None and _s(pended.args.get("decision") or pended.args.get("recommendation")) == \
            "request_info" and final is None:
        if not state.happened_after("payer_letter_center.generate_pend", pended):
            out += _block("the case was pended (request_info): generate the pend letter "
                          "(payer_letter_center.generate_pend)")
    md = state.last("review.submit_md_case_decision")
    if md is not None and _s(md.args.get("decision")) in MD_FINALIZING and final is None:
        out += _warn("MD decision recorded but determination not finalized; if your seat owns the "
                     "determination, call determination.finalize")
    return out


# --- case management -----------------------------------------------------------

def _require(*tools: str) -> RuleFn:
    def fn(tool: str, args: dict, state: ObservedState) -> list[Finding]:
        missing = [t for t in tools if not state.has(t)]
        if missing:
            return _block(f"workflow order: complete {', '.join(missing)} before {tool}")
        return []
    return fn


def _cm_chart_review(tool: str, args: dict, state: ObservedState) -> list[Finding]:
    review = args.get("review") if isinstance(args.get("review"), dict) else args
    out: list[Finding] = []
    if _s(review.get("preliminary_risk_level")) not in RISK_LEVELS:
        out += _block(f"preliminary_risk_level must be one of {sorted(RISK_LEVELS)}")
    if _blank(review.get("key_findings")):
        out += _block("key_findings must be a substantive 2-5 sentence clinical synthesis with chart numbers")
    if _blank(review.get("primary_concerns")):
        out += _warn("primary_concerns is empty")
    return out


def _saved_sections(state: ObservedState) -> set[str]:
    sections: set[str] = set()
    for event in state.succeeded("cm_assessment.save_section"):
        payload = event.args.get("payload")
        if isinstance(payload, dict) and payload.get("section"):
            sections.add(_s(payload.get("section")))
        elif event.args.get("section"):
            sections.add(_s(event.args.get("section")))
    return sections


def _cm_complete_assessment(tool: str, args: dict, state: ObservedState) -> list[Finding]:
    missing = [s for s in CM_SECTIONS if s not in _saved_sections(state)]
    if missing:
        return _block(f"save all assessment sections before completing; missing: {missing}")
    return []


def care_plan_structure_problems(plan: Any) -> list[str]:
    """Structural problems as checked by stages/cm_v4.py."""
    problems_out: list[str] = []
    if not isinstance(plan, dict):
        return ["plan_data is missing"]
    problems = plan.get("problems")
    if not isinstance(problems, list) or not problems:
        return ["at least one problem is required"]
    for p_i, problem in enumerate(problems, 1):
        goals = problem.get("goals") if isinstance(problem, dict) else None
        if not isinstance(goals, list) or not goals:
            problems_out.append(f"problem {p_i} has no goals")
            continue
        for g_i, goal in enumerate(goals, 1):
            if not isinstance(goal, dict):
                problems_out.append(f"problem {p_i} goal {g_i} is malformed")
                continue
            if _blank(goal.get("target_date")):
                problems_out.append(f"problem {p_i} goal {g_i} lacks target_date")
            if _blank(goal.get("measurable_criteria")):
                problems_out.append(f"problem {p_i} goal {g_i} lacks measurable_criteria")
            interventions = goal.get("interventions")
            if not isinstance(interventions, list) or not interventions:
                problems_out.append(f"problem {p_i} goal {g_i} has no interventions")
                continue
            for i_i, item in enumerate(interventions, 1):
                if not isinstance(item, dict):
                    problems_out.append(f"intervention {p_i}.{g_i}.{i_i} is malformed")
                    continue
                if _blank(item.get("owner")):
                    problems_out.append(f"intervention {p_i}.{g_i}.{i_i} lacks owner")
                if _blank(item.get("due_date")) and _blank(item.get("frequency")):
                    problems_out.append(f"intervention {p_i}.{g_i}.{i_i} lacks due_date or frequency")
    if _blank(plan.get("escalation_conditions")):
        problems_out.append("escalation_conditions is empty")
    if _blank(plan.get("follow_up_cadence_kind")):
        problems_out.append("follow_up_cadence_kind is empty")
    return problems_out


def _plan_from_args(args: dict) -> Any:
    return args.get("plan_data") if "plan_data" in args else args


def _cm_save_plan(tool: str, args: dict, state: ObservedState) -> list[Finding]:
    issues = care_plan_structure_problems(_plan_from_args(args))
    return _warn("care plan structure incomplete: " + "; ".join(issues[:8])) if issues else []


def _cm_finalize(tool: str, args: dict, state: ObservedState) -> list[Finding]:
    out = _require("cm_assessment.complete_assessment", "cm_care_plan.create_draft",
                   "cm_care_plan.save_care_plan")(tool, args, state)
    last = state.last("cm_care_plan.save_care_plan")
    if last is not None:
        issues = care_plan_structure_problems(_plan_from_args(last.args))
        if issues:
            out += _block("the last saved care plan fails the structure contract: " + "; ".join(issues[:10])
                          + ". Save a corrected complete plan, then finalize.")
    return out


def _cm_end_outreach(tool: str, args: dict, state: ObservedState) -> list[Finding]:
    out = _require("cm_outreach.start_outreach_call")(tool, args, state)
    summary = args.get("summary") if isinstance(args.get("summary"), dict) else args
    if summary.get("consent_obtained") is True:
        sent = [e for e in state.succeeded("cm_outreach.send_message")]
        if not sent:
            out += _block("consent_obtained=true but no outreach messages were exchanged")
        elif not any(_CONSENT_RE.search(str(e.args.get("message") or "")) for e in sent):
            out += _block("consent_obtained=true but you never asked for consent/permission in the call; "
                          "only mark consent when the consent script actually happened")
    if _blank(summary.get("engagement_status")):
        out += _block("engagement_status is required (engaged | tentative | refused | pending)")
    return out


def _cm_disenroll(tool: str, args: dict, state: ObservedState) -> list[Finding]:
    out = []
    if _blank(args.get("reason")):
        out += _block("disenrollment needs a reason grounded in the transcript")
    out += _warn("disenrollment is terminal and irreversible: only if the patient explicitly and repeatedly "
                 "refused after respectful consent attempts")
    return out


def _cm_finish(tool: str, args: dict, state: ObservedState) -> list[Finding]:
    if state.has("cm_intake.disenroll_case") or state.has("cm_care_plan.finalize"):
        return []
    blocked = any(e.ok and e.args.get("blocked_reason") for e in state.succeeded("cm_assessment.save_section"))
    if blocked:
        return []
    return _block("the CM case is not in a terminal state: finish the workflow through cm_care_plan.finalize")


# --- rule catalog -------------------------------------------------------

def _t(*names: str) -> frozenset[str]:
    return frozenset(names)


ALL_RULES: tuple[Rule, ...] = (
    Rule("GLOBAL-01", "*", frozenset(), "Never mutate a case other than this task's target case.",
         ("cross.forbidden_mutations", "cm.cross_stage.no_forbidden_mutations"), _global_case_lock),
    # provider
    Rule("PROV-01a", ROLE_PROVIDER, _t("cases.create_from_order"),
         "Valid enums and grounded codes when creating the PA case.",
         ("new_referral.service_type", "new_referral.site_of_service", "new_referral.procedure_code",
          "new_referral.required_diagnosis_codes"), _prov_create_case),
    Rule("PROV-01b", ROLE_PROVIDER, _t("auth.finalize_provider_determination", "docs.create_submission_bundle",
                                         "auth.submit_authorization"),
         "A PA case must exist before dispositions/bundles/submissions.",
         ("new_referral.case_created",), _prov_needs_case),
    Rule("PROV-02/03/05", ROLE_PROVIDER, _t("auth.finalize_provider_determination"),
         "Disposition: valid action; non-submit has override_reason and no packet; submit_pa has a bundle.",
         ("provider_pre_submission.final_action", "provider_pre_submission.packet_absent",
          "provider_pre_submission.required_documents_present"), _prov_disposition),
    Rule("PROV-03b", ROLE_PROVIDER, _t("auth.submit_authorization"),
         "Submit only after a submit_pa disposition, with the bundle.", ("new_referral.case_status",),
         _prov_submit),
    Rule("PROV-04", ROLE_PROVIDER, _t(FINISH), "Do not finish without a recorded provider disposition "
         "(and the submission when submit_pa).",
         ("provider_pre_submission.provider_determination_present", "new_referral.case_status",
          "cross.closed_backed_by_provider_determination"), _prov_finish),
    # payer
    Rule("UM-05a", ROLE_PAYER, _t("payer_intake_hub.confirm_decisions"),
         "Intake decisions are explicit booleans grounded in the intake review.",
         ("intake.audit", "intake.decision_request_complete", "intake.decision_member_eligible"),
         _um_intake_confirm),
    Rule("UM-05b", ROLE_PAYER, _t("triage.route_case"), "Set the triage disposition before routing.",
         ("triage.record_exists", "triage.audit"), _um_route),
    Rule("UM-01", ROLE_PAYER, _t("review.submit_nurse_case_recommendation"),
         "Nurse never denies; request_info needs info_requested; notes are rationale-rich.",
         ("nurse.no_unauthorized_denial", "nurse.recommendation_exists"), _um_nurse),
    Rule("UM-02", ROLE_PAYER, _t("review.submit_md_case_decision"),
         "Finalizing MD decisions are signed off with a rationale (denial rationale for deny).",
         ("md.signed_off", "md.rationale_present", "md.denial_rationale_present"), _um_md),
    Rule("UM-03", ROLE_PAYER, _t("determination.finalize"),
         "Determination matches the upstream decision unless explicitly overridden; P2P resolved first.",
         ("outcome.source_recommendation_consistent", "outcome.clean_determination"), _um_finalize),
    Rule("UM-06a", ROLE_PAYER, _t("p2p.schedule"), "Offer slots before scheduling.", ("p2p.audit",),
         _um_schedule),
    Rule("UM-06b", ROLE_PAYER, _t("p2p.record_outcome"), "Offer and schedule before recording the outcome.",
         ("p2p.audit", "p2p.audit_chronology", "p2p.result_recorded"), _um_record_p2p),
    Rule("UM-04", ROLE_PAYER, _t(FINISH), "Generate the letter matching the outcome before finishing.",
         ("outcome.letter_types", "outcome.target_status"), _um_finish),
    # case management
    Rule("CM-01a", ROLE_CM, _t("cm_chart.submit_chart_review"), "Open the case before the chart review; "
         "valid risk level and substantive findings.",
         ("cm.chart_review.record_exists", "cm.cross_stage.audit_actions"),
         lambda t, a, s: _require("cm_intake.open_case")(t, a, s) + _cm_chart_review(t, a, s)),
    Rule("CM-01b/04", ROLE_CM, _t("cm_outreach.end_outreach_call"),
         "End the call only after it started; consent_obtained only with an actual consent exchange.",
         ("cm.cross_stage.audit_actions", "judge.cm.outreach.quality"), _cm_end_outreach),
    Rule("CM-01c", ROLE_CM, _t("cm_assessment.start_assessment"), "Assessment follows the outreach.",
         ("cm.cross_stage.audit_actions",), _require("cm_outreach.end_outreach_call")),
    Rule("CM-02", ROLE_CM, _t("cm_assessment.complete_assessment"), "All four sections saved before completion.",
         ("cm.assessment.required_sections_present", "cm.assessment.completed"), _cm_complete_assessment),
    Rule("CM-03a", ROLE_CM, _t("cm_care_plan.save_care_plan"), "Warn early on care plan structure gaps.",
         ("cm.care_plan.goal_structure", "cm.care_plan.intervention_structure"), _cm_save_plan),
    Rule("CM-03b", ROLE_CM, _t("cm_care_plan.finalize"), "Finalize only a structurally complete plan.",
         ("cm.care_plan.finalized", "cm.care_plan.problem_count", "cm.care_plan.goal_structure",
          "cm.care_plan.intervention_structure", "cm.care_plan.escalation_conditions_present",
          "cm.care_plan.follow_up_cadence_present", "cm.cross_stage.target_status"), _cm_finalize),
    Rule("CM-05", ROLE_CM, _t("cm_intake.disenroll_case"), "Disenrollment is a last resort with a reason.",
         ("cm.cross_stage.target_status",), _cm_disenroll),
    Rule("CM-06", ROLE_CM, _t(FINISH), "Do not finish before the case reaches a terminal state.",
         ("cm.cross_stage.target_status", "cm.care_plan.finalized"), _cm_finish),
)


class Checklist:
    def __init__(self, role: str, rules: tuple[Rule, ...] = ALL_RULES):
        self.role = role
        self.rules = tuple(r for r in rules if r.role in (role, "*"))

    def _run(self, tool: str, args: dict[str, Any], state: ObservedState, rules: list[Rule]) -> ChecklistResult:
        result = ChecklistResult(tool=tool)
        for rule in rules:
            for finding in rule.fn(tool, args or {}, state):
                result.violations.append(Violation(rule.rule_id, finding.severity, finding.message, rule.protects))
        return result

    def check_action(self, tool: str, args: dict[str, Any], state: ObservedState) -> ChecklistResult:
        rules = [r for r in self.rules if tool in r.tools or (r.rule_id == "GLOBAL-01")]
        return self._run(tool, args, state, rules)

    def check_finish(self, state: ObservedState) -> ChecklistResult:
        rules = [r for r in self.rules if FINISH in r.tools]
        return self._run(FINISH, {}, state, rules)

    def is_terminal(self, tool: str) -> bool:
        return tool in TERMINAL_TOOLS.get(self.role, frozenset())

    def describe(self) -> str:
        lines = ["CONTRACT CHECKLIST (enforced before irreversible actions and before finishing):"]
        for rule in self.rules:
            lines.append(f"- {rule.rule_id}: {rule.description}")
        return "\n".join(lines)


def update_case_lock(state: ObservedState, tool: str, args: dict[str, Any], ok: bool) -> None:
    """Lock the target case after the first successful mutation."""
    if not ok or state.locked_case_id:
        return
    created = state.created_case_id() if tool in ("cases.create_from_order", "cm_intake.open_case") else None
    if created:
        state.locked_case_id = created
        return
    case_id = args.get("case_id")
    if is_mutating(tool) and isinstance(case_id, str) and case_id.strip():
        state.locked_case_id = case_id.strip()

