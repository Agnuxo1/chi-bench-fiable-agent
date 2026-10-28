"""Main loop of the "reliability by construction" agent.

Flow per turn: LLM -> (tool_use)* -> for each call:
  local tool   -> runs in the container (Handbook / workspace)
  mutating     -> checklist (target-case lock, per-tool rules)
  terminal     -> checklist + 3-judge critic -> MCP only if approved
When trying to finish: checklist finish gate (outstanding obligations).

Artifacts in ``logs_dir`` (default /logs/agent):
  run_result.json  (schema compatible with openai_agents_runner, read by the harness)
  trace.jsonl      (format converted to ATIF v1.2 by openai_agents_harness._build_atif_trajectory)
  ledger.json/.jsonl, critic_log.json, agent_config.json
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from .config import AgentConfig
from .contracts import Checklist, ChecklistResult, is_mutating, update_case_lock
from .critic import CriticContext, CriticPanel, ProposedAction, build_panel
from .ledger import CostLedger, load_prices
from .prompts import blocked_message, build_system_prompt, finish_nudge
from .rag import NullRetriever, Retriever, format_hits
from .roles import detect_role, render_plan
from .state import ObservedState
from .tools import LocalToolRoots, LocalTools, ToolGateway, ToolNameMap, ToolOutcome

_EPHEMERAL = {"type": "ephemeral"}


@dataclass
class RunResult:
    role: str
    final_output: str | None
    error: str | None
    turns: int
    stopped_reason: str
    totals: dict[str, Any] = field(default_factory=dict)
    blocked_actions: int = 0


class FiableAgent:
    def __init__(
        self,
        config: AgentConfig,
        llm: Any,
        gateway: ToolGateway,
        *,
        retriever: Retriever | None = None,
        ledger: CostLedger | None = None,
        logs_dir: str | Path | None = None,
        local_roots: LocalToolRoots | None = None,
        critic_llm: Any | None = None,
        clock: Callable[[], float] = time.monotonic,
    ):
        self.cfg = config
        self.llm = llm
        self.critic_llm = critic_llm or llm
        self.gateway = gateway
        self.retriever = retriever or NullRetriever()
        self.ledger = ledger or CostLedger(prices=load_prices(config.prices_file))
        self.logs_dir = Path(logs_dir) if logs_dir else None
        self.local = LocalTools(self.retriever, local_roots or LocalToolRoots(handbook=config.rag.handbook_dir),
                                top_k=config.rag.top_k)
        self.clock = clock
        self.state = ObservedState()
        self.role = ""
        self.instruction = ""
        self.checklist: Checklist | None = None
        self.panel: CriticPanel | None = None
        self.trace: list[dict[str, Any]] = []
        self._checklist_blocks: dict[str, int] = {}
        self._last_reasoning = ""

    # ------------------------------------------------------------------ util
    def _with_cache(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if not self.cfg.loop.prompt_caching or not messages:
            return messages
        out = list(messages)
        last = dict(out[-1])
        content = last.get("content")
        if isinstance(content, str):
            content = [{"type": "text", "text": content}]
        if isinstance(content, list) and content:
            content = list(content)
            content[-1] = {**content[-1], "cache_control": _EPHEMERAL}
            last["content"] = content
            out[-1] = last
        return out

    def _action_rag(self, tool: str, args: dict[str, Any]) -> str:
        if not self.retriever.available:
            return ""
        query = f"{tool.replace('.', ' ').replace('_', ' ')} {json.dumps(args, ensure_ascii=False)[:1500]}"
        return format_hits(self.retriever.search(query, self.cfg.rag.top_k), 3000)

    # -------------------------------------------------------------- dispatch
    async def _dispatch(self, tool: str, args: dict[str, Any]) -> ToolOutcome:
        if tool in self.local.names:
            out = self.local.call(tool, args)
            self.state.record(tool, args, out.ok, out.text)
            self.ledger.record_tool(tool, "local", out.ok)
            return out

        assert self.checklist is not None
        terminal = self.checklist.is_terminal(tool)
        checked = self.cfg.checklist.enabled and (terminal or is_mutating(tool))
        result = self.checklist.check_action(tool, args, self.state) if checked else ChecklistResult(tool)

        allow, reason, forced_next = True, "", False
        if self.panel is not None and terminal:
            decision = self.panel.decide(
                ProposedAction(tool, args),
                CriticContext(self.role, self.instruction, self.state, result,
                              self._action_rag(tool, args), self._last_reasoning),
            )
            allow, reason = decision.allow, decision.reason
            forced_next = self.panel._blocks.get(tool, 0) >= self.panel.max_blocks_per_action
        elif result.blocking:
            count = self._checklist_blocks.get(tool, 0) + 1
            self._checklist_blocks[tool] = count
            limit = self.cfg.critic.max_blocks_per_action
            allow = count > limit
            reason = "; ".join(f"{v.rule_id} {v.message}" for v in result.blocking)
            forced_next = count >= limit

        if not allow:
            msg = blocked_message(tool, reason, forced_next)
            self.state.record(tool, args, False, msg, blocked=True)
            self.ledger.record_tool(tool, "blocked", False)
            return ToolOutcome(False, msg)

        out = await self.gateway.call(tool, args)
        self.state.record(tool, args, out.ok, out.text)
        update_case_lock(self.state, tool, args, out.ok)
        self.ledger.record_tool(tool, "mcp", out.ok)
        if out.ok and result.warnings:
            out = ToolOutcome(True, out.text + "\n[internal checklist warnings] " +
                              "; ".join(f"{v.rule_id} {v.message}" for v in result.warnings), out.data)
        return out

    # ------------------------------------------------------------------- run
    async def run(self, instruction: str, mcp_url: str | None = None) -> RunResult:
        cfg = self.cfg
        t0 = self.clock()
        self.instruction = instruction
        specs = await self.gateway.list_tools()
        self.role = detect_role(instruction, mcp_url, [s.name for s in specs])
        self.checklist = Checklist(self.role)
        if cfg.critic.enabled:
            self.panel = build_panel(cfg.critic.judges, llm=self.critic_llm, model=cfg.critic_model,
                                     ledger=self.ledger, mode=cfg.critic.mode,
                                     max_blocks_per_action=cfg.critic.max_blocks_per_action,
                                     policy_max_tokens=cfg.critic.policy_max_tokens)

        names = ToolNameMap()
        tools_param: list[dict[str, Any]] = []
        for spec in specs:
            api = names.register(spec.name)
            tools_param.append({"name": api, "description": (spec.description or spec.name)[:4000],
                                "input_schema": spec.input_schema or {"type": "object", "properties": {}}})
        for spec in self.local.specs():
            names.register(spec["name"])
            tools_param.append(dict(spec))
        if cfg.loop.prompt_caching and tools_param:
            tools_param[-1] = {**tools_param[-1], "cache_control": _EPHEMERAL}

        rag_text = ""
        if cfg.rag.enabled and cfg.rag.inject_on_start and self.retriever.available:
            rag_text = format_hits(self.retriever.search(instruction[:3000], cfg.rag.top_k), cfg.rag.max_inject_chars)
        system_text = build_system_prompt(self.role, render_plan(self.role),
                                          self.checklist.describe() if cfg.checklist.enabled else "", rag_text)
        system: Any = [{"type": "text", "text": system_text,
                        **({"cache_control": _EPHEMERAL} if cfg.loop.prompt_caching else {})}]

        messages: list[dict[str, Any]] = [{"role": "user", "content": instruction}]
        self.trace = [{"role": "system", "content": system_text}, {"role": "user", "content": instruction}]
        final_output: str | None = None
        error: str | None = None
        stopped = "max_turns"
        turns = 0
        nudges = 0
        extra = dict(cfg.model.extra_request_params or {})
        if cfg.model.temperature is not None:
            extra["temperature"] = cfg.model.temperature

        try:
            while turns < cfg.loop.max_turns:
                if self.clock() - t0 > cfg.loop.wall_clock_budget_s:
                    stopped = "time_budget"
                    break
                resp = self.llm.create(model=cfg.agent_model, system=system, messages=self._with_cache(messages),
                                       tools=tools_param, max_tokens=cfg.model.max_output_tokens, **extra)
                turns += 1
                call = self.ledger.record_llm("agent", cfg.agent_model, resp.usage, resp.latency_s)
                messages.append({"role": "assistant", "content": resp.content})
                if resp.reasoning:
                    self._last_reasoning = resp.reasoning[-4000:]
                    self.trace.append({"role": "reasoning", "content": resp.reasoning})
                uses = resp.tool_uses
                self.trace.append({
                    "role": "assistant", "content": resp.text, "model": cfg.agent_model,
                    "usage": {"input_tokens": call.n_input, "output_tokens": call.output,
                              "cached_tokens": call.cache_read},
                    "tool_calls": [{"id": u.get("id"), "name": names.canonical(u.get("name", "")),
                                    "arguments": u.get("input") or {}} for u in uses],
                })
                if not uses:
                    if resp.stop_reason == "max_tokens":
                        messages.append({"role": "user", "content": "Continue."})
                        continue
                    if cfg.checklist.enabled and cfg.checklist.enforce_finish_gate:
                        fin = self.checklist.check_finish(self.state)
                        if fin.blocking and nudges < cfg.checklist.max_finish_nudges:
                            nudges += 1
                            msg = finish_nudge(fin.render())
                            messages.append({"role": "user", "content": msg})
                            self.trace.append({"role": "user", "content": msg})
                            continue
                    final_output = resp.text
                    stopped = "end_turn"
                    break
                results: list[dict[str, Any]] = []
                for use in uses:
                    tool = names.canonical(use.get("name", ""))
                    outcome = await self._dispatch(tool, use.get("input") or {})
                    text = outcome.text or ""
                    limit = cfg.loop.max_tool_result_chars
                    if len(text) > limit:
                        text = text[:limit] + f"\n[truncated {len(outcome.text) - limit} chars]"
                    block: dict[str, Any] = {"type": "tool_result", "tool_use_id": use.get("id"), "content": text}
                    if not outcome.ok:
                        block["is_error"] = True
                    results.append(block)
                    self.trace.append({"role": "tool", "tool_call_id": use.get("id"), "content": text})
                messages.append({"role": "user", "content": results})
        except Exception as exc:  # noqa: BLE001 - recorded and returned
            error = f"{type(exc).__name__}: {exc}"
            stopped = "error"

        blocked = sum(1 for e in self.state.events if e.blocked)
        totals = self._write_outputs(final_output, error, turns, stopped, blocked)
        return RunResult(self.role, final_output, error, turns, stopped, totals, blocked)

    # --------------------------------------------------------------- outputs
    def _write_outputs(self, final_output: str | None, error: str | None, turns: int, stopped: str,
                       blocked: int) -> dict[str, Any]:
        extra = {"role": self.role, "stopped_reason": stopped, "blocked_actions": blocked,
                 "profile": self.cfg.profile, "agent_model": self.cfg.agent_model,
                 "critic_model": self.cfg.critic_model, "locked_case_id": self.state.locked_case_id,
                 "tool_counts": self.state.tool_counts()}
        if self.logs_dir is None:
            return {**self.ledger.totals(), **extra}
        self.ledger.write(self.logs_dir, model=self.cfg.agent_model, turns=turns,
                          final_output=final_output, error=error, extra=extra)
        with (self.logs_dir / "trace.jsonl").open("w", encoding="utf-8") as fh:
            for rec in self.trace:
                fh.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")
        (self.logs_dir / "critic_log.json").write_text(
            json.dumps(self.panel.log if self.panel else [], indent=2, default=str), encoding="utf-8")
        (self.logs_dir / "agent_config.json").write_text(json.dumps(self.cfg.to_dict(), indent=2), encoding="utf-8")
        if final_output:
            (self.logs_dir / "run_output.txt").write_text(final_output, encoding="utf-8")
        return {**self.ledger.totals(), **extra}

