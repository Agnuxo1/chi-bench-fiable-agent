"""Tests need no network or keys: all LLM and MCP calls are in-memory doubles."""

import sys
from pathlib import Path

import pytest

AGENTE = Path(__file__).resolve().parents[1]
if str(AGENTE) not in sys.path:
    sys.path.insert(0, str(AGENTE))


@pytest.fixture(autouse=True)
def _no_real_credentials(monkeypatch):
    # Guarantees that no test can accidentally use a real key.
    for key in ("ANTHROPIC_API_KEY", "FIABLE_AGENT_CONFIG", "FIABLE_AGENT_MODEL"):
        monkeypatch.delenv(key, raising=False)
