"""Summarize a fiable-agent trials directory (Harbor): official reward + own ledger.

    python experiments/summarize_ledger.py repo/chi-bench/logs/experiments/e4_critic_um_x3 [--csv out.csv]

Reads per trial:
  result.json                  (official verifier reward; Harbor tokens)
  agent/ledger.json            (written by fiable_agent: calls by kind, blocks)
  agent/critic_log.json        (critic verdicts; forced overrides)
Modifies nothing (Rules section 7: do not alter results or trajectories).
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path


def _reward(result: dict) -> float | None:
    vr = result.get("verifier_result")
    if not isinstance(vr, dict):
        return None
    rb = vr.get("rewards")
    value = rb.get("reward") if isinstance(rb, dict) else vr.get("reward")
    try:
        return float(value or 0.0)
    except (TypeError, ValueError):
        return None


def _task(result: dict, trial_dir: Path) -> str:
    name = result.get("task_name") or (result.get("task") or {}).get("path") or ""
    return Path(str(name)).name or trial_dir.name.split("__", 1)[0]


def collect(root: Path) -> list[dict]:
    rows = []
    for result_path in sorted(root.rglob("result.json")):
        trial = result_path.parent
        try:
            result = json.loads(result_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        reward = _reward(result)
        if reward is None:
            continue
        ledger, critic = {}, []
        if (trial / "agent" / "ledger.json").exists():
            ledger = json.loads((trial / "agent" / "ledger.json").read_text(encoding="utf-8"))
        if (trial / "agent" / "critic_log.json").exists():
            critic = json.loads((trial / "agent" / "critic_log.json").read_text(encoding="utf-8"))
        kinds = ledger.get("tool_calls_by_kind") or {}
        started = (result.get("agent_execution") or {}).get("started_at") or ""
        rows.append({
            "task": _task(result, trial),
            "trial": trial.name,
            "started_at": started,
            "reward": reward,
            "cost_usd": ledger.get("cost_usd"),
            "tool_calls_total": ledger.get("tool_calls_total"),
            "mcp_calls": kinds.get("mcp"),
            "local_calls": kinds.get("local"),
            "blocked_calls": kinds.get("blocked"),
            "llm_calls_agent": (ledger.get("llm_calls_by_role") or {}).get("agent"),
            "llm_calls_critic": (ledger.get("llm_calls_by_role") or {}).get("critic"),
            "forced_overrides": sum(1 for c in critic if c.get("forced")),
            "stopped_reason": ledger.get("stopped_reason"),
            "role": ledger.get("role"),
        })
    return rows


def summarize(rows: list[dict]) -> dict:
    by_task: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_task[row["task"]].append(row)
    first = [sorted(v, key=lambda r: (r["started_at"], r["trial"]))[0]["reward"] for v in by_task.values()]
    ordered = {k: sorted(v, key=lambda r: (r["started_at"], r["trial"])) for k, v in by_task.items()}
    pow3 = [1.0 if len(v) >= 3 and all(r["reward"] == 1.0 for r in v[:3]) else 0.0 for v in ordered.values()]

    def mean(key: str) -> float | None:
        vals = [r[key] for r in rows if isinstance(r.get(key), (int, float))]
        return round(statistics.mean(vals), 4) if vals else None

    return {
        "n_tasks": len(by_task),
        "n_trials": len(rows),
        "pass_at_1_first_attempt": round(statistics.mean(first), 4) if first else None,
        "pass_pow_3": round(statistics.mean(pow3), 4) if by_task and all(len(v) >= 3 for v in by_task.values())
        else None,
        "mean_cost_usd_ledger": mean("cost_usd"),
        "mean_tool_calls": mean("tool_calls_total"),
        "mean_blocked_calls": mean("blocked_calls"),
        "mean_llm_calls_critic": mean("llm_calls_critic"),
        "forced_overrides_total": sum(r["forced_overrides"] for r in rows),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("trials_root", type=Path)
    parser.add_argument("--csv", type=Path)
    args = parser.parse_args()
    rows = collect(args.trials_root)
    print(json.dumps(summarize(rows), indent=2))
    if args.csv and rows:
        with args.csv.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


if __name__ == "__main__":
    main()

