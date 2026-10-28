"""fiable_agent - "reliability by construction" agent for chi-Bench (IEEE Big Data Cup 2026).

Modules:
  - roles.py      per-role planner (provider / payer / care manager)
  - contracts.py  checklist of verifier contracts
  - critic.py     internal 3-judge critic that blocks irreversible actions
  - rag.py        Handbook retrieval (local BM25, optional FAISS, null stub)
  - ledger.py     per-task cost / call ledger (tie-breakers)
  - agent.py      tool-use loop over MCP
  - runner.py     entry point inside the agent container

All internal imports are relative so the package works both as ``fiable_agent``
(local tests) and as ``chi_bench.experiment.agents.fiable_agent`` (inside the
``chi-bench:agent`` image).
"""

__version__ = "0.2.0"

