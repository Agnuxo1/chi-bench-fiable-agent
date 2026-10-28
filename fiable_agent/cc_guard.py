"""Claude Code hook entry point: the fiable action guard for the official claude-code harness.

The same reliability components used by the stand-alone loop (contract checklist,
critic panel, closure gate) run here as Claude Code hooks, so the agent itself is
the unmodified official Claude Code CLI:

    PreToolUse   checklist + critic panel on mutating/terminal MCP calls; exit 2 blocks
                 the call and returns the reason to the model
    PostToolUse  records the call and its result as observed evidence
    Stop         closure gate: refuses to stop while terminal obligations are open

Invocation (written into $CLAUDE_CONFIG_DIR/settings.json by the fiable-cc harness):

    python -m fiable_agent.cc_guard <pre|post|stop> <guard_dir>

<guard_dir> holds ``config.json`` (written by the harness) and receives
``state.json`` and ``guard_log.jsonl``. Hook payloads arrive as JSON on stdin.
Every failure inside the guard is fail-open (exit 0): the guard may slow the
agent down but must never break a trial.
"""

from __future__ import annotations

import json
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

from .contracts import FINISH, Checklist, ChecklistResult, is_mutating, update_case_lock
from .critic import CriticContext, ProposedAction, build_panel
from .prompts import ROLE_TITLES
from .roles import NAMESPACE_ROLE, render_plan
from .state import ObservedState

try:  # the guard runs inside Linux containers; fcntl is absent on Windows dev machines
    import fcntl
except ImportError:  # pragma: no cover - exercised only on Windows
    fcntl = None  # type: ignore[assignment]

BLOCK_EXIT = 2
MAX_EVENT_TEXT = 20_000

DEFAULTS: dict[str, Any] = {
    "role": "provider",
    "instruction": "",
    "checklist": True,
    "critic": False,
    "judges": ["contract", "grounding", "policy"],
    "max_blocks_per_action": 2,
    "finish_gate": True,
    "max_finish_nudges": 2,
}

_NAMESPACES = sorted(NAMESPACE_ROLE, key=len, reverse=True)


# --------------------------------------------------------------------------- names
def canonical_tool(tool_name: str) -> str:
    """Map a Claude Code tool name back to the simulator's dotted MCP name.

    ``mcp__chi_bench__review_submit_md_case_decision`` -> ``review.submit_md_case_decision``.
    Built-in tools (Read, Bash, ...) become ``local.<Name>``.
    """
    if not tool_name.startswith("mcp__"):
        return f"local.{tool_name}"
    rest = tool_name[len("mcp__"):]
    rest = rest.split("__", 1)[1] if "__" in rest else rest
    if "." in rest:
        return rest
    for ns in _NAMESPACES:
        if rest.startswith(ns + "_"):
            return f"{ns}.{rest[len(ns) + 1:]}"
    return rest


def response_text(response: Any) -> str:
    """Flatten a PostToolUse ``tool_response`` (string, content blocks or dict) to text."""
    if response is None:
        return ""
    if isinstance(response, str):
        return response
    if isinstance(response, list):
        parts = []
        for block in response:
            if isinstance(block, dict) and isinstance(block.get("text"), str):
                parts.append(block["text"])
            else:
                parts.append(json.dumps(block, ensure_ascii=False, default=str))
        return "\n".join(parts)
    if isinstance(response, dict):
        for key in ("content", "result", "output", "stdout"):
            if key in response:
                inner = response_text(response[key])
                if inner:
                    return inner
        return json.dumps(response, ensure_ascii=False, default=str)
    return str(response)


def response_ok(response: Any, text: str) -> bool:
    if isinstance(response, dict) and (response.get("is_error") or response.get("isError")):
        return False
    head = text.lstrip()[:200].lower()
    return not (head.startswith("error") or '"error"' in head[:40])


# --------------------------------------------------------------------------- persistence
class GuardStore:
    def __init__(self, guard_dir: Path):
        self.dir = guard_dir
        self.config = {**DEFAULTS, **self._read_json(guard_dir / "config.json", {})}
        self.state_path = guard_dir / "state.json"
        self.lock_path = guard_dir / "state.lock"

    @staticmethod
    def _read_json(path: Path, default: Any) -> Any:
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return default

    @contextmanager
    def locked(self) -> Iterator[dict[str, Any]]:
        self.dir.mkdir(parents=True, exist_ok=True)
        with open(self.lock_path, "a+") as lock:
            if fcntl is not None:
                fcntl.flock(lock, fcntl.LOCK_EX)
            data = self._read_json(self.state_path, {})
            data.setdefault("events", [])
            data.setdefault("blocks", {})
            data.setdefault("finish_nudges", 0)
            yield data
            tmp = self.state_path.with_suffix(".tmp")
            tmp.write_text(json.dumps(data, ensure_ascii=False, default=str), encoding="utf-8")
            tmp.replace(self.state_path)

    def log(self, record: dict[str, Any]) -> None:
        try:
            with open(self.dir / "guard_log.jsonl", "a", encoding="utf-8") as fh:
                fh.write(json.dumps({"t": round(time.time(), 3), **record}, ensure_ascii=False, default=str) + "\n")
        except OSError:
            pass


def rebuild_state(data: dict[str, Any]) -> ObservedState:
    state = ObservedState()
    for ev in data["events"]:
        state.record(ev["name"], ev.get("args") or {}, bool(ev.get("ok")), ev.get("text") or "",
                     blocked=bool(ev.get("blocked")))
    state.locked_case_id = data.get("locked_case_id")
    return state


def append_event(data: dict[str, Any], name: str, args: dict[str, Any], ok: bool, text: str,
                 blocked: bool = False) -> None:
    data["events"].append({"name": name, "args": args, "ok": ok, "text": text[:MAX_EVENT_TEXT],
                           "blocked": blocked})


# --------------------------------------------------------------------------- hooks
def on_pre(payload: dict[str, Any], store: GuardStore) -> tuple[int, str]:
    cfg = store.config
    tool = canonical_tool(str(payload.get("tool_name") or ""))
    args = payload.get("tool_input") or {}
    if not isinstance(args, dict) or tool.startswith("local.") or not cfg["checklist"]:
        return 0, ""
    checklist = Checklist(cfg["role"])
    terminal = checklist.is_terminal(tool)
    if not (terminal or is_mutating(tool)):
        return 0, ""
    with store.locked() as data:
        state = rebuild_state(data)
        result = checklist.check_action(tool, args, state)
        limit = int(cfg["max_blocks_per_action"])
        count = int(data["blocks"].get(tool, 0))
        if cfg["critic"] and terminal:
            panel = build_panel(cfg["judges"], llm=None, model=None, ledger=None, mode="heuristic",
                                max_blocks_per_action=limit)
            panel._blocks[tool] = count
            decision = panel.decide(ProposedAction(tool, args),
                                    CriticContext(cfg["role"], cfg["instruction"], state, result))
            allow, reason, forced = decision.allow, decision.reason, decision.forced
            verdicts = [v.__dict__ for v in decision.verdicts]
        else:
            allow, reason, forced, verdicts = not result.blocking, "; ".join(
                f"{v.rule_id} {v.message}" for v in result.blocking), False, []
            if not allow and count >= limit:
                allow, forced = True, True
        if not allow:
            data["blocks"][tool] = count + 1
            append_event(data, tool, args, False, f"blocked: {reason}", blocked=True)
    store.log({"hook": "pre", "tool": tool, "allow": allow, "forced": forced, "reason": reason,
               "verdicts": verdicts, "warnings": [w.rule_id for w in result.warnings]})
    if allow:
        return 0, ""
    final = count + 1 >= limit
    tail = (" This is the final warning for this action; retrying it unchanged will execute it as-is."
            if final else "")
    return BLOCK_EXIT, (f"BLOCKED_BY_INTERNAL_CHECKLIST: '{tool}' was NOT executed. Reason: {reason}. "
                        f"Fix the arguments or complete the missing step, then retry.{tail}")


def on_post(payload: dict[str, Any], store: GuardStore) -> tuple[int, str]:
    tool = canonical_tool(str(payload.get("tool_name") or ""))
    args = payload.get("tool_input") or {}
    if not isinstance(args, dict):
        args = {"input": args}
    response = payload.get("tool_response")
    text = response_text(response)
    ok = response_ok(response, text)
    with store.locked() as data:
        append_event(data, tool, args, ok, text)
        state = rebuild_state(data)
        update_case_lock(state, tool, args, ok)
        data["locked_case_id"] = state.locked_case_id
    return 0, ""


def on_stop(payload: dict[str, Any], store: GuardStore) -> tuple[int, str]:
    cfg = store.config
    if not (cfg["checklist"] and cfg["finish_gate"]):
        return 0, ""
    with store.locked() as data:
        state = rebuild_state(data)
        result: ChecklistResult = Checklist(cfg["role"]).check_finish(state)
        nudges = int(data["finish_nudges"])
        block = bool(result.blocking) and nudges < int(cfg["max_finish_nudges"])
        if block:
            data["finish_nudges"] = nudges + 1
    store.log({"hook": "stop", "block": block, "open": [v.rule_id for v in result.blocking]})
    if not block:
        return 0, ""
    reasons = "\n".join(f"- {v.rule_id}: {v.message}" for v in result.blocking)
    return BLOCK_EXIT, ("Not finished yet. The contract checklist reports outstanding obligations:\n" + reasons +
                        "\nComplete them with the tools, then give your final summary.")


HOOKS = {"pre": on_pre, "post": on_post, "stop": on_stop}


# --------------------------------------------------------------------------- prompt
def build_append_prompt(role: str, *, with_checklist: bool) -> str:
    """System-prompt addendum for the Claude Code harness (role plan + operating principles)."""
    guard = ("Irreversible and mutating tool calls are reviewed by an internal contract checklist. If a call comes "
             "back BLOCKED, read the reason, fix the payload or perform the missing step, and retry; never repeat "
             "a blocked call unchanged.\n") if with_checklist else ""
    return (
        f"You are an autonomous {ROLE_TITLES.get(role, role)} in a simulator of U.S. healthcare operations. "
        "No human will answer questions: never stop to ask.\n"
        "OPERATING PRINCIPLES (reliability by construction)\n"
        "1. Ground everything: every code, date, identifier and decision must come from the chart, case data, "
        "request documents, transcripts or the Managed-Care Operations Handbook. Never invent identifiers.\n"
        "2. Policy first: open the handbook index (SKILL.md under /workspace/skills/), go to your role's sub-book, "
        "and read the governing criteria before deciding. In rationales name the criterion, the patient-specific "
        "facts and numbers, and the handbook section.\n"
        "3. Follow the role plan below in order; skip phases that do not apply to your seat, never mandatory ones.\n"
        f"{guard}"
        "4. Stay on the target case; never modify other cases or read simulator source, tests or expectations.\n"
        "5. Be efficient: batch independent reads and do not re-fetch data you already have.\n"
        "6. Finish only when the workflow has reached its terminal state and every obligation is met.\n\n"
        + render_plan(role)
    )


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if len(argv) != 2 or argv[0] not in HOOKS:
        print("usage: python -m fiable_agent.cc_guard <pre|post|stop> <guard_dir>", file=sys.stderr)
        return 0
    store = GuardStore(Path(argv[1]))
    try:
        payload = json.loads(sys.stdin.read() or "{}")
        code, message = HOOKS[argv[0]](payload, store)
    except Exception as exc:  # noqa: BLE001 - fail open, keep a trace
        store.log({"hook": argv[0], "error": f"{type(exc).__name__}: {exc}"})
        return 0
    if message:
        print(message, file=sys.stderr)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
