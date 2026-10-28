"""State observed by the agent: ONLY its own calls and results.

The checklist and the critic reason over this state; never over verifier
files (expectations/rubrics), which the agent can not and must not read.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any


def parse_payload(text: str) -> Any:
    """Try to decode the text returned by an MCP tool as JSON."""
    if not isinstance(text, str):
        return text
    stripped = text.strip()
    if not stripped or stripped[0] not in "[{":
        return None
    try:
        return json.loads(stripped)
    except (json.JSONDecodeError, ValueError):
        return None


def _find_key(node: Any, keys: tuple[str, ...], depth: int = 0) -> str | None:
    if depth > 4:
        return None
    if isinstance(node, dict):
        for key in keys:
            value = node.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
        for value in node.values():
            found = _find_key(value, keys, depth + 1)
            if found:
                return found
    elif isinstance(node, list):
        for value in node[:5]:
            found = _find_key(value, keys, depth + 1)
            if found:
                return found
    return None


@dataclass
class ToolEvent:
    name: str  # canonical dotted name, e.g. "review.submit_md_case_decision"
    args: dict[str, Any]
    ok: bool
    text: str = ""
    data: Any = None
    blocked: bool = False
    index: int = 0

    @property
    def case_id(self) -> str | None:
        value = self.args.get("case_id")
        if isinstance(value, str) and value.strip():
            return value.strip()
        return None


@dataclass
class ObservedState:
    events: list[ToolEvent] = field(default_factory=list)
    locked_case_id: str | None = None

    def record(
        self,
        name: str,
        args: dict[str, Any],
        ok: bool,
        text: str = "",
        *,
        blocked: bool = False,
    ) -> ToolEvent:
        event = ToolEvent(
            name=name,
            args=dict(args or {}),
            ok=ok,
            text=text or "",
            data=parse_payload(text) if ok else None,
            blocked=blocked,
            index=len(self.events),
        )
        self.events.append(event)
        return event

    # --- queries ---------------------------------------------------------
    def succeeded(self, name: str) -> list[ToolEvent]:
        return [e for e in self.events if e.name == name and e.ok and not e.blocked]

    def has(self, name: str) -> bool:
        return bool(self.succeeded(name))

    def last(self, name: str) -> ToolEvent | None:
        hits = self.succeeded(name)
        return hits[-1] if hits else None

    def last_any(self, names: tuple[str, ...] | list[str]) -> ToolEvent | None:
        hits = [e for e in self.events if e.name in names and e.ok and not e.blocked]
        return hits[-1] if hits else None

    def happened_after(self, later: str, earlier_event: ToolEvent | None) -> bool:
        if earlier_event is None:
            return False
        return any(e.index > earlier_event.index for e in self.succeeded(later))

    def created_case_id(self) -> str | None:
        """case_id returned by case creation/opening (provider or CM)."""
        for name in ("cases.create_from_order", "cm_intake.open_case"):
            event = self.last(name)
            if event is None or not event.data:
                continue
            data = event.data
            # cases.create_from_order returns {"case": {"id": ...}, ...}
            # (see reference_oracle.py:157); CMCase carries "case_id" at the root.
            if isinstance(data, dict):
                if isinstance(data.get("case_id"), str) and data["case_id"].strip():
                    return data["case_id"].strip()
                case = data.get("case")
                if isinstance(case, dict):
                    found = _find_key(case, ("id", "case_id"))
                    if found:
                        return found
            found = _find_key(data, ("case_id",))
            if found:
                return found
        return None

    def evidence_text(self, max_chars: int = 400_000) -> str:
        """Corpus of observed evidence (results of successful tool calls)."""
        parts: list[str] = []
        total = 0
        for event in self.events:
            if not event.ok or event.blocked or not event.text:
                continue
            parts.append(event.text)
            total += len(event.text)
            if total > max_chars:
                break
        return "\n".join(parts)

    def tool_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for event in self.events:
            counts[event.name] = counts.get(event.name, 0) + 1
        return counts

