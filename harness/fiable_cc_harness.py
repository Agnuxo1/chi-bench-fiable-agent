"""Harbor harness ``fiable-cc``: the official Claude Code harness plus the fiable action guard.

Installed next to the other in-tree harnesses
(``src/chi_bench/experiment/agents/fiable_cc_harness.py``) by
``scripts/install_into_chi_bench.py``. The agent is the unmodified Claude Code
CLI driven by Harbor's ``ClaudeCode`` harness; this subclass only adds:

* ``--append-system-prompt``: operating principles + the role plan for the seat
  detected from the task instruction;
* ``--settings <guard_dir>/settings.json``: PreToolUse / PostToolUse / Stop hooks
  that run ``fiable_agent.cc_guard`` inside the agent container;
* an upload of the ``fiable_agent`` package to ``/opt/fiable`` so the hooks do
  not depend on how the agent image was built.

Ablation arms (``FIABLE_CC_ARM`` in the experiment's ``agent_env``):

    stock      identical to the official claude-code harness (control)
    plan       + system-prompt addendum (principles + role plan)
    checklist  + contract-checklist hooks and closure gate
    critic     + three-judge critic panel (contract, grounding, heuristic policy) on terminal calls

Credentials are untouched: ``ANTHROPIC_API_KEY`` / ``CLAUDE_CODE_OAUTH_TOKEN`` are
handled by the parent harness exactly as for ``claude-code``.
"""

from __future__ import annotations

import json
import os
import shlex
from pathlib import Path
from typing import TYPE_CHECKING

from harbor.agents.installed.claude_code import ClaudeCode
from harbor.models.trial.paths import EnvironmentPaths

if TYPE_CHECKING:
    from harbor.environments.base import BaseEnvironment
    from harbor.models.agent.context import AgentContext

ARMS = ("stock", "plan", "checklist", "critic")
DEFAULT_ARM = "checklist"
PACKAGE_DIR = Path(__file__).resolve().parent / "fiable_agent"
REMOTE_ROOT = "/opt/fiable"
GUARD_DIR = (EnvironmentPaths.agent_dir / "fiable").as_posix()
PYTHON = '"$( [ -x /workspace/.venv/bin/python ] && echo /workspace/.venv/bin/python || command -v python3 )"'


def hook_command(event: str) -> str:
    return f"PYTHONPATH={REMOTE_ROOT} {PYTHON} -m fiable_agent.cc_guard {event} {GUARD_DIR}"


def guard_settings() -> dict:
    def entry(event: str, matcher: str | None) -> list[dict]:
        hook = {"type": "command", "command": hook_command(event), "timeout": 30}
        return [{"matcher": matcher, "hooks": [hook]}] if matcher is not None else [{"hooks": [hook]}]

    return {"hooks": {
        "PreToolUse": entry("pre", "mcp__.*"),
        "PostToolUse": entry("post", ".*"),
        "Stop": entry("stop", None),
    }}


class FiableClaudeCodeHarness(ClaudeCode):
    @staticmethod
    def name() -> str:
        return "fiable-cc"

    def _arm(self) -> str:
        getter = getattr(self, "_get_env", None)
        value = (getter("FIABLE_CC_ARM") if getter else None) or os.environ.get("FIABLE_CC_ARM") or DEFAULT_ARM
        value = value.strip().lower()
        if value not in ARMS:
            raise ValueError(f"FIABLE_CC_ARM must be one of {ARMS}, got {value!r}")
        return value

    def build_cli_flags(self) -> str:
        flags = super().build_cli_flags() or ""
        extra = getattr(self, "_fiable_extra_flags", "")
        return f"{flags} {extra}".strip()

    async def run(self, instruction: str, environment: "BaseEnvironment", context: "AgentContext") -> None:
        from chi_bench.experiment.agents.fiable_agent.cc_guard import build_append_prompt
        from chi_bench.experiment.agents.fiable_agent.roles import detect_role

        arm = self._arm()
        self._fiable_extra_flags = ""
        if arm != "stock":
            mcp_url = next((s.url for s in self.mcp_servers if getattr(s, "url", None)), None)
            role = detect_role(instruction, mcp_url)
            prompt = build_append_prompt(role, with_checklist=arm in ("checklist", "critic"))
            flags = [f"--append-system-prompt {shlex.quote(prompt)}"]
            if arm in ("checklist", "critic"):
                config = {"arm": arm, "role": role, "instruction": instruction, "checklist": True,
                          "critic": arm == "critic", "finish_gate": True}
                await self.exec_as_root(environment, command=f"mkdir -p {REMOTE_ROOT} {GUARD_DIR}")
                await environment.upload_dir(PACKAGE_DIR, f"{REMOTE_ROOT}/fiable_agent")
                write = (f"printf '%s' {shlex.quote(json.dumps(config))} > {GUARD_DIR}/config.json && "
                         f"printf '%s' {shlex.quote(json.dumps(guard_settings()))} > {GUARD_DIR}/settings.json")
                await self.exec_as_root(environment, command=f"{write} && chmod -R a+rwX {GUARD_DIR} "
                                                             f"&& chmod -R a+rX {REMOTE_ROOT}")
                flags.append(f"--settings {GUARD_DIR}/settings.json")
            self._fiable_extra_flags = " ".join(flags)
        await super().run(instruction, environment, context)
