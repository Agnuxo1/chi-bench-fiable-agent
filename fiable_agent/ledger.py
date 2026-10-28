"""Per-task cost and call-count ledger (tie-breakers 3 and 4 of Rules section 12).

Conventions (aligned with repo/chi-bench):
  * ``n_input_tokens`` INCLUDES the cache (repo CLAUDE.md; aggregator.py subtracts
    ``n_cache_tokens`` before applying the base price).
  * Cache write price: prices.yaml does not model it; aggregator.py charges it
    as normal input. The same formula is replicated here so that our own cost
    matches the one ``cb submission prepare`` will compute.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml

# USD per 1M tokens - copied from repo/chi-bench/configs/prices.yaml @7998d113.
DEFAULT_PRICES: dict[str, dict[str, float]] = {
    "anthropic/claude-sonnet-5": {"input": 3.00, "output": 15.00, "cache": 0.30},
    "anthropic/claude-opus-5": {"input": 5.00, "output": 25.00, "cache": 0.50},
    "anthropic/claude-opus-4-7": {"input": 5.00, "output": 25.00, "cache": 0.50},
    "anthropic/claude-opus-4-6": {"input": 5.00, "output": 25.00, "cache": 0.50},
    "anthropic/claude-sonnet-4-6": {"input": 3.00, "output": 15.00, "cache": 0.30},
    "anthropic/claude-haiku-4-5": {"input": 1.00, "output": 5.00, "cache": 0.10},
}


def load_prices(path: str | Path | None) -> dict[str, dict[str, float]]:
    if not path:
        return dict(DEFAULT_PRICES)
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    prices = data.get("prices", data)
    merged = dict(DEFAULT_PRICES)
    merged.update({k: v for k, v in prices.items() if isinstance(v, dict)})
    return merged


def _get(usage: Any, key: str) -> int:
    value = usage.get(key) if isinstance(usage, dict) else getattr(usage, key, None)
    return int(value) if isinstance(value, (int, float)) and value >= 0 else 0


@dataclass
class LLMCall:
    role: str  # agent | critic | other
    model: str
    uncached_input: int
    cache_read: int
    cache_write: int
    output: int
    latency_s: float = 0.0

    @property
    def n_input(self) -> int:  # Harbor convention: includes cache
        return self.uncached_input + self.cache_read + self.cache_write


@dataclass
class ToolCallRecord:
    name: str
    kind: str  # mcp | local | blocked
    ok: bool


@dataclass
class CostLedger:
    prices: dict[str, dict[str, float]] = field(default_factory=lambda: dict(DEFAULT_PRICES))
    llm_calls: list[LLMCall] = field(default_factory=list)
    tool_calls: list[ToolCallRecord] = field(default_factory=list)
    started_at: float = field(default_factory=time.time)

    def record_llm(self, role: str, model: str, usage: Any, latency_s: float = 0.0) -> LLMCall:
        """``usage`` in Anthropic format: input_tokens (uncached), cache_read_input_tokens,
        cache_creation_input_tokens, output_tokens."""
        call = LLMCall(
            role=role,
            model=model,
            uncached_input=_get(usage, "input_tokens"),
            cache_read=_get(usage, "cache_read_input_tokens"),
            cache_write=_get(usage, "cache_creation_input_tokens"),
            output=_get(usage, "output_tokens"),
            latency_s=latency_s,
        )
        self.llm_calls.append(call)
        return call

    def record_tool(self, name: str, kind: str, ok: bool) -> None:
        self.tool_calls.append(ToolCallRecord(name, kind, ok))

    def cost_of(self, call: LLMCall) -> float | None:
        price = self.prices.get(call.model)
        if not price:
            return None
        inp, out = float(price["input"]), float(price["output"])
        cache = float(price.get("cache", inp * 0.1))
        return (
            (call.n_input - call.cache_read) / 1e6 * inp
            + call.output / 1e6 * out
            + call.cache_read / 1e6 * cache
        )

    def totals(self) -> dict[str, Any]:
        n_input = sum(c.n_input for c in self.llm_calls)
        n_cache = sum(c.cache_read for c in self.llm_calls)
        n_output = sum(c.output for c in self.llm_calls)
        costs = [self.cost_of(c) for c in self.llm_calls]
        cost = None if any(v is None for v in costs) else round(sum(v for v in costs if v is not None), 6)
        by_model: dict[str, dict[str, Any]] = {}
        for call, c in zip(self.llm_calls, costs):
            entry = by_model.setdefault(call.model, {"calls": 0, "n_input_tokens": 0, "n_cache_tokens": 0,
                                                     "n_output_tokens": 0, "cost_usd": 0.0})
            entry["calls"] += 1
            entry["n_input_tokens"] += call.n_input
            entry["n_cache_tokens"] += call.cache_read
            entry["n_output_tokens"] += call.output
            entry["cost_usd"] = None if (c is None or entry["cost_usd"] is None) else entry["cost_usd"] + c
        by_role: dict[str, int] = {}
        for call in self.llm_calls:
            by_role[call.role] = by_role.get(call.role, 0) + 1
        kinds: dict[str, int] = {"mcp": 0, "local": 0, "blocked": 0}
        for t in self.tool_calls:
            kinds[t.kind] = kinds.get(t.kind, 0) + 1
        return {
            "n_input_tokens": n_input,
            "n_cache_tokens": n_cache,
            "n_output_tokens": n_output,
            "cost_usd": cost,
            "llm_calls": len(self.llm_calls),
            "llm_calls_by_role": by_role,
            "tool_calls_total": len(self.tool_calls),
            "tool_calls_by_kind": kinds,
            "cost_by_model": by_model,
            "models": sorted(by_model),
            "wall_clock_s": round(time.time() - self.started_at, 3),
        }

    def write(
        self,
        logs_dir: str | Path,
        *,
        model: str,
        turns: int,
        final_output: str | None,
        error: str | None,
        extra: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        logs = Path(logs_dir)
        logs.mkdir(parents=True, exist_ok=True)
        totals = self.totals()
        # Same schema as openai_agents_runner.write_result, for the harness.
        run_result = {
            "input_tokens": totals["n_input_tokens"],
            "output_tokens": totals["n_output_tokens"],
            "cached_tokens": totals["n_cache_tokens"],
            "cost_usd": totals["cost_usd"],
            "turns": turns,
            "final_output": final_output,
            "model": model,
            "error": error,
            "mixed_models": len(totals["models"]) > 1,
        }
        (logs / "run_result.json").write_text(json.dumps(run_result, indent=2), encoding="utf-8")
        summary = {**totals, **(extra or {})}
        (logs / "ledger.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
        with (logs / "ledger.jsonl").open("w", encoding="utf-8") as fh:
            for call in self.llm_calls:
                fh.write(json.dumps({"type": "llm", **asdict(call)}) + "\n")
            for tool in self.tool_calls:
                fh.write(json.dumps({"type": "tool", **asdict(tool)}) + "\n")
        return run_result

