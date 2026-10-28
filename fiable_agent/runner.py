"""Entry point inside the agent container (invoked by the harness).

    python -m chi_bench.experiment.agents.fiable_agent.runner \
        --instruction-file /tmp/instruction.md --mcp-url http://localhost:8100/mcp

Environment:
  ANTHROPIC_API_KEY     required (forwarded by Harbor; never printed)
  FIABLE_AGENT_CONFIG   profile name / YAML path (default: configs/default.yaml)
  FIABLE_AGENT_MODEL    model override (the harness passes submission.model)
  FIABLE_AGENT_LOGS_DIR output directory (default: /logs/agent)
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from pathlib import Path

from .agent import FiableAgent
from .config import API_KEY_ENV, has_api_key, load_config
from .ledger import CostLedger, load_prices
from .llm import AnthropicClient
from .rag import build_retriever
from .tools import MCPGateway


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="fiable_agent runner")
    parser.add_argument("--instruction-file", required=True)
    parser.add_argument("--mcp-url", required=True)
    parser.add_argument("--config", default=None, help="YAML path or packaged profile name")
    parser.add_argument("--logs-dir", default=os.environ.get("FIABLE_AGENT_LOGS_DIR", "/logs/agent"))
    return parser.parse_args(argv)


async def _amain(args: argparse.Namespace) -> int:
    cfg = load_config(args.config)
    if not has_api_key():
        print(f"fatal: {API_KEY_ENV} is not set in the agent environment", file=sys.stderr)
        return 2
    instruction = Path(args.instruction_file).read_text(encoding="utf-8")
    retriever = build_retriever(cfg.rag)
    ledger = CostLedger(prices=load_prices(cfg.prices_file))
    llm = AnthropicClient()
    print(f"fiable_agent profile={cfg.profile} model={cfg.agent_model} critic={cfg.critic_model} "
          f"rag={retriever.name}:{'on' if retriever.available else 'off'} mcp={args.mcp_url}")
    try:
        async with MCPGateway(args.mcp_url) as gateway:
            agent = FiableAgent(cfg, llm, gateway, retriever=retriever, ledger=ledger, logs_dir=args.logs_dir)
            result = await agent.run(instruction, args.mcp_url)
    except Exception as exc:  # e.g. MCP unreachable: leave a consistent run_result.json
        error = f"{type(exc).__name__}: {exc}"
        ledger.write(args.logs_dir, model=cfg.agent_model, turns=0, final_output=None, error=error)
        print(f"agent error: {error}", file=sys.stderr)
        return 1
    print(f"role={result.role} turns={result.turns} stop={result.stopped_reason} "
          f"blocked={result.blocked_actions} cost_usd={result.totals.get('cost_usd')} "
          f"tool_calls={result.totals.get('tool_calls_total')}")
    if result.final_output:
        print("---\n" + result.final_output)
    if result.error:
        print(f"agent error: {result.error}", file=sys.stderr)
        return 1
    return 0


def main(argv: list[str] | None = None) -> None:
    sys.exit(asyncio.run(_amain(parse_args(argv))))


if __name__ == "__main__":
    main()

