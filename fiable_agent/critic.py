"""Internal multi-judge critic (inspired by the BenchClaw Tribunal).

Three judges with DISTINCT rubrics review every irreversible action before it
reaches the simulator:

1. ``ContractJudge``  (deterministic) - does it satisfy the contract checklist?
                       Rubric: contracts analysis, section 6.
2. ``GroundingJudge`` (deterministic) - does every identifier / code in the
                       payload appear in the observed evidence (tool results,
                       instruction, retrieved Handbook)?
                       Rubric: "Do not fabricate evidence" (instructions).
3. ``PolicyJudge``    (LLM or heuristic) - is the clinical and policy
                       decision/reasoning justified and does it cite concrete
                       criteria?
                       Rubric: mirror of the verifier's [J] rubrics.

Aggregation: any ``block`` from a deterministic judge blocks; LLM judges block
by majority. After ``max_blocks_per_action`` blocks on the same tool, the
action is let through marked as ``forced`` (an agent blocked indefinitely
exhausts the timeout and scores 0 for sure).
Every verdict carries a reason; everything is kept in the log.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Protocol

from .contracts import ChecklistResult
from .state import ObservedState

_ICD10_RE = re.compile(r"\b[A-TV-Z][0-9][0-9AB](?:\.[0-9A-TV-Z]{1,4})?\b")
_CPT_RE = re.compile(r"\b(?:\d{4}[0-9FTU]|[A-V]\d{4})\b")
_ID_RE = re.compile(r"\b(?:CM-CASE|CM-PAT|CM-REF|CASE|PAT|P2P|DOC|LET|REF|INT|SR|ORD|PRAC|SES|CP)-[A-Z0-9_\-]{3,}\b")
_ID_KEYS = ("case_id", "request_id", "bundle_document_id", "letter_id", "referral_id", "session_id",
            "patient_id", "ordering_practitioner_id", "source_order_id", "document_ids", "review_id")
_CODE_KEYS = ("procedure_code", "diagnosis_codes", "code", "codes", "cpt", "icd10")


@dataclass
class ProposedAction:
    tool: str
    args: dict[str, Any]


@dataclass
class CriticContext:
    role: str
    instruction: str
    state: ObservedState
    checklist: ChecklistResult
    rag_text: str = ""
    recent_reasoning: str = ""


@dataclass
class Verdict:
    judge: str
    allow: bool
    reason: str
    deterministic: bool = True
    severity: str = "ok"  # ok | warn | block


@dataclass
class PanelDecision:
    allow: bool
    verdicts: list[Verdict] = field(default_factory=list)
    forced: bool = False

    @property
    def reason(self) -> str:
        blocks = [v for v in self.verdicts if not v.allow]
        if not blocks:
            return "approved by internal critic"
        return " | ".join(f"{v.judge}: {v.reason}" for v in blocks)

    def to_dict(self) -> dict[str, Any]:
        return {"allow": self.allow, "forced": self.forced, "reason": self.reason,
                "verdicts": [v.__dict__ for v in self.verdicts]}


class Judge(Protocol):
    name: str

    def review(self, action: ProposedAction, ctx: CriticContext) -> Verdict: ...


class ContractJudge:
    name = "contract"
    rubric = "Deterministic workflow contract (contracts analysis, section 6)."

    def review(self, action: ProposedAction, ctx: CriticContext) -> Verdict:
        if ctx.checklist.blocking:
            return Verdict(self.name, False, "; ".join(f"{v.rule_id} {v.message}" for v in ctx.checklist.blocking),
                           True, "block")
        if ctx.checklist.warnings:
            return Verdict(self.name, True, "; ".join(f"{v.rule_id} {v.message}" for v in ctx.checklist.warnings),
                           True, "warn")
        return Verdict(self.name, True, "contract checklist satisfied")


def _walk_values(node: Any, keys: tuple[str, ...], found: list[str], key: str = "") -> None:
    if isinstance(node, dict):
        for k, v in node.items():
            _walk_values(v, keys, found, str(k))
    elif isinstance(node, list):
        for v in node:
            _walk_values(v, keys, found, key)
    elif isinstance(node, str) and key in keys and node.strip():
        found.append(node.strip())


class GroundingJudge:
    name = "grounding"
    rubric = "Every identifier and clinical code in the payload must appear in observed evidence."

    def review(self, action: ProposedAction, ctx: CriticContext) -> Verdict:
        evidence = "\n".join([ctx.instruction or "", ctx.state.evidence_text(), ctx.rag_text or ""])
        candidates: list[str] = []
        _walk_values(action.args, _ID_KEYS + _CODE_KEYS, candidates)
        # identifiers with a known prefix that appear in free text
        free_text = json.dumps(action.args, ensure_ascii=False)
        candidates += _ID_RE.findall(free_text)
        missing = sorted({c for c in candidates if c and c not in evidence})
        if not candidates:
            return Verdict(self.name, True, "no identifiers/codes to ground")
        if missing:
            return Verdict(self.name, False,
                           f"not found in any observed tool result or the instruction: {missing[:8]}; "
                           "look them up with the read tools instead of guessing", True, "block")
        return Verdict(self.name, True, f"{len(set(candidates))} identifiers/codes grounded")


POLICY_RUBRIC = """You are an independent utilization-management / care-management compliance reviewer.
Decide whether the PROPOSED IRREVERSIBLE ACTION should be executed now.
Block ONLY for a concrete, fixable problem, e.g.:
- the decision/disposition contradicts the evidence or the governing policy criteria shown;
- a required rationale is generic (no patient-specific facts, numbers, or policy criterion);
- a mandatory prior workflow step is clearly missing;
- the role is exceeding its authority (e.g. a nurse denying).
Do NOT block for style. Answer with JSON only: {"allow": true|false, "reason": "<one or two sentences>"}"""


def _excerpt(text: str, limit: int) -> str:
    text = text or ""
    return text if len(text) <= limit else text[: limit // 2] + "\n...\n" + text[-limit // 2:]


class PolicyJudge:
    name = "policy"
    rubric = POLICY_RUBRIC

    def __init__(self, llm: Any | None = None, model: str | None = None, ledger: Any | None = None,
                 max_tokens: int = 700, evidence_chars: int = 12000):
        self.llm = llm
        self.model = model
        self.ledger = ledger
        self.max_tokens = max_tokens
        self.evidence_chars = evidence_chars

    @property
    def deterministic(self) -> bool:
        return self.llm is None

    def _heuristic(self, action: ProposedAction) -> Verdict:
        text_fields = [v for k, v in action.args.items() if isinstance(v, str) and any(
            w in k for w in ("rationale", "reason", "notes", "summary", "findings"))]
        if text_fields and max(len(t.strip()) for t in text_fields) < 60:
            return Verdict(self.name, False, "rationale text is too thin; cite patient-specific facts and the "
                                             "policy criterion", True, "block")
        return Verdict(self.name, True, "heuristic policy check passed", True)

    def review(self, action: ProposedAction, ctx: CriticContext) -> Verdict:
        if self.llm is None:
            return self._heuristic(action)
        user = (
            f"ROLE: {ctx.role}\nTASK INSTRUCTION:\n{_excerpt(ctx.instruction, 3000)}\n\n"
            f"PROPOSED ACTION: {action.tool}\nARGUMENTS:\n{json.dumps(action.args, ensure_ascii=False)[:6000]}\n\n"
            f"CONTRACT CHECKLIST:\n{ctx.checklist.render()}\n\n"
            f"HANDBOOK PASSAGES:\n{_excerpt(ctx.rag_text, 3000)}\n\n"
            f"OBSERVED EVIDENCE (tool results, truncated):\n{_excerpt(ctx.state.evidence_text(), self.evidence_chars)}"
        )
        try:
            resp = self.llm.create(model=self.model, system=self.rubric,
                                   messages=[{"role": "user", "content": user}], max_tokens=self.max_tokens)
        except Exception as exc:  # API failure: do not block (fail-open), leave a trace
            return Verdict(self.name, True, f"policy judge unavailable ({type(exc).__name__}); allowed", False, "warn")
        if self.ledger is not None:
            self.ledger.record_llm("critic", self.model or "", resp.usage, resp.latency_s)
        return self.parse(resp.text)

    def parse(self, text: str) -> Verdict:
        match = re.search(r"\{.*\}", text or "", re.DOTALL)
        if not match:
            return Verdict(self.name, True, "unparseable policy verdict; allowed", False, "warn")
        try:
            data = json.loads(match.group(0))
        except json.JSONDecodeError:
            return Verdict(self.name, True, "unparseable policy verdict; allowed", False, "warn")
        allow = data.get("allow")
        if not isinstance(allow, bool):
            return Verdict(self.name, True, "policy verdict without boolean; allowed", False, "warn")
        reason = str(data.get("reason") or "").strip() or ("ok" if allow else "blocked without reason")
        return Verdict(self.name, allow, reason, False, "ok" if allow else "block")


class CriticPanel:
    def __init__(self, judges: list[Any], max_blocks_per_action: int = 2):
        self.judges = judges
        self.max_blocks_per_action = max_blocks_per_action
        self._blocks: dict[str, int] = {}
        self.log: list[dict[str, Any]] = []

    def decide(self, action: ProposedAction, ctx: CriticContext) -> PanelDecision:
        verdicts = [j.review(action, ctx) for j in self.judges]
        det_block = any((not v.allow) and v.deterministic for v in verdicts)
        llm = [v for v in verdicts if not v.deterministic]
        llm_block = bool(llm) and sum(1 for v in llm if not v.allow) * 2 > len(llm)
        allow = not (det_block or llm_block)
        forced = False
        if not allow:
            count = self._blocks.get(action.tool, 0) + 1
            self._blocks[action.tool] = count
            if count > self.max_blocks_per_action:
                allow, forced = True, True
        decision = PanelDecision(allow, verdicts, forced)
        self.log.append({"tool": action.tool, **decision.to_dict()})
        return decision


def build_panel(names: list[str], *, llm: Any | None, model: str | None, ledger: Any | None,
                mode: str = "llm", max_blocks_per_action: int = 2, policy_max_tokens: int = 700) -> CriticPanel:
    judges: list[Any] = []
    for name in names:
        if name == "contract":
            judges.append(ContractJudge())
        elif name == "grounding":
            judges.append(GroundingJudge())
        elif name == "policy":
            judges.append(PolicyJudge(llm if mode == "llm" else None, model, ledger, policy_max_tokens))
    return CriticPanel(judges, max_blocks_per_action)

