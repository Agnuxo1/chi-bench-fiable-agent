"""Harbor harness for fiable_agent (installed at
``repo/chi-bench/src/chi_bench/experiment/agents/fiable_agent_harness.py``
by ``scripts/install_into_chi_bench.py``).

Contract: docs/extending.md section 2 (BaseInstalledAgent). Reference model:
openai_agents_harness.py (same run/tee/populate_context_post_run pattern).

* The ``fiable_agent`` package is baked into the ``chi-bench:agent`` image
  (the installer adds a COPY to the ``agent`` target of the Dockerfile), inside
  ``chi_bench.experiment.agents`` so it is importable with the editable venv.
* Credentials: ``ANTHROPIC_API_KEY`` arrives through Harbor (``_extra_env``)
  because the model is ``anthropic/...`` (runner.py:_selected_agent_env). It is
  never printed.
* Cost: ``populate_context_post_run`` copies n_input/n_output/n_cache tokens and
  cost to ``AgentContext`` -> ``result.json.agent_result`` -> aggregator.py.
"""

from __future__ import annotations

import json
import logging
import os
import shlex
from typing import TYPE_CHECKING, Any

from harbor.agents.installed.base import BaseInstalledAgent, with_prompt_template

if TYPE_CHECKING:
    from harbor.environments.base import BaseEnvironment
    from harbor.models.agent.context import AgentContext

logger = logging.getLogger(__name__)

RUNNER_MODULE = "chi_bench.experiment.agents.fiable_agent.runner"
_FORWARD_KEYS = ("ANTHROPIC_API_KEY", "FIABLE_AGENT_CONFIG")


class FiableAgentHarness(BaseInstalledAgent):
    SUPPORTS_ATIF: bool = True

    @staticmethod
    def name() -> str:
        return "fiable-agent"

    def get_version_command(self) -> str | None:
        return ("python -c 'import chi_bench.experiment.agents.fiable_agent as m; print(m.__version__)' "
                "2>/dev/null")

    async def install(self, environment: "BaseEnvironment") -> None:
        # No network install: dependencies (anthropic, mcp, pyyaml) are already in /workspace/.venv.
        await self.exec_as_root(
            environment,
            command="python -c 'import chi_bench.experiment.agents.fiable_agent.runner' "
                    "|| (echo 'fiable_agent missing from image: run install_into_chi_bench.py "
                    "and cb docker build' >&2; exit 1)",
        )

    @with_prompt_template
    async def run(self, instruction: str, environment: "BaseEnvironment", context: "AgentContext") -> None:
        mcp_url = next((s.url for s in self.mcp_servers if getattr(s, "url", None)), "")
        if not mcp_url:
            raise RuntimeError("no MCP server URL in task config (task.toml [[environment.mcp_servers]])")
        env: dict[str, str] = {"FIABLE_AGENT_LOGS_DIR": "/logs/agent"}
        if self.model_name:
            env["FIABLE_AGENT_MODEL"] = self.model_name
        for key in _FORWARD_KEYS:
            value = self._extra_env.get(key) or os.environ.get(key)
            if value:
                env[key] = value
        await self.exec_as_agent(
            environment,
            command=(
                f"echo {shlex.quote(instruction)} > /tmp/instruction.md && "
                f"python -m {RUNNER_MODULE} --instruction-file /tmp/instruction.md "
                f"--mcp-url {shlex.quote(mcp_url)} 2>&1 | tee /logs/agent/run_log.txt"
            ),
            env=env,
        )

    def populate_context_post_run(self, context: "AgentContext") -> None:
        result: dict[str, Any] = {}
        path = self.logs_dir / "run_result.json"
        if path.exists():
            try:
                result = json.loads(path.read_text())
            except (OSError, json.JSONDecodeError) as exc:
                logger.debug("cannot read %s: %s", path, exc)
        if result.get("input_tokens") is not None:
            context.n_input_tokens = result["input_tokens"]
        if result.get("output_tokens") is not None:
            context.n_output_tokens = result["output_tokens"]
        if result.get("cached_tokens") is not None:
            context.n_cache_tokens = result["cached_tokens"]
        if result.get("cost_usd") is not None:
            context.cost_usd = result["cost_usd"]

        trace_path = self.logs_dir / "trace.jsonl"
        if not trace_path.exists():
            return
        try:
            from harbor.models.trajectories.agent import Agent

            from chi_bench.experiment.agents.openai_agents_harness import _build_atif_trajectory, _read_trace

            trajectory = _build_atif_trajectory(
                records=_read_trace(trace_path),
                model_name=result.get("model") or self.model_name,
                final_output=result.get("final_output"),
                error=result.get("error"),
                agent_version=self.version() or "unknown",
                cost_usd=result.get("cost_usd"),
            )
            trajectory.agent = Agent(name=self.name(), version=self.version() or "unknown",
                                     model_name=result.get("model") or self.model_name)
            (self.logs_dir / "trajectory.json").write_text(
                json.dumps(trajectory.to_json_dict(), indent=2, ensure_ascii=False), encoding="utf-8")
        except Exception as exc:  # noqa: BLE001 - the trajectory does not affect the reward
            logger.debug("failed to build ATIF trajectory: %s", exc)

