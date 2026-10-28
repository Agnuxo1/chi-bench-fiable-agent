import json

import pytest

from fiable_agent.ledger import CostLedger, load_prices


def aggregator_cost(n_input, n_output, n_cache, price):
    """Formula from repo/chi-bench/src/chi_bench/aggregator.py:_cost."""
    return (max(n_input - n_cache, 0) / 1e6 * price["input"] + n_output / 1e6 * price["output"]
            + n_cache / 1e6 * price["cache"])


def test_usage_normalization_and_cost_matches_aggregator():
    ledger = CostLedger()
    usage = {"input_tokens": 1200, "cache_read_input_tokens": 50_000, "cache_creation_input_tokens": 3000,
             "output_tokens": 800}
    ledger.record_llm("agent", "anthropic/claude-sonnet-5", usage)
    ledger.record_llm("agent", "anthropic/claude-sonnet-5", {"input_tokens": 500, "output_tokens": 100})
    t = ledger.totals()
    assert t["n_input_tokens"] == 1200 + 50_000 + 3000 + 500
    assert t["n_cache_tokens"] == 50_000
    assert t["n_output_tokens"] == 900
    price = load_prices(None)["anthropic/claude-sonnet-5"]
    expected = aggregator_cost(t["n_input_tokens"], t["n_output_tokens"], t["n_cache_tokens"], price)
    assert t["cost_usd"] == pytest.approx(expected, rel=1e-9)


def test_unknown_model_price_gives_null_cost():
    ledger = CostLedger()
    ledger.record_llm("agent", "anthropic/claude-unknown", {"input_tokens": 1, "output_tokens": 1})
    assert ledger.totals()["cost_usd"] is None


def test_tool_counts_and_files(tmp_path):
    ledger = CostLedger()
    ledger.record_llm("agent", "anthropic/claude-haiku-4-5", {"input_tokens": 10, "output_tokens": 2})
    ledger.record_llm("critic", "anthropic/claude-sonnet-5", {"input_tokens": 10, "output_tokens": 2})
    ledger.record_tool("review.get_md_review_data", "mcp", True)
    ledger.record_tool("handbook_search", "local", True)
    ledger.record_tool("determination.finalize", "blocked", False)
    rr = ledger.write(tmp_path, model="anthropic/claude-haiku-4-5", turns=3, final_output="done", error=None)
    assert set(rr) >= {"input_tokens", "output_tokens", "cached_tokens", "cost_usd", "turns", "model", "error"}
    assert rr["mixed_models"] is True  # warning: aggregator.py prices everything at a single price
    summary = json.loads((tmp_path / "ledger.json").read_text(encoding="utf-8"))
    assert summary["tool_calls_by_kind"] == {"mcp": 1, "local": 1, "blocked": 1}
    assert summary["llm_calls_by_role"] == {"agent": 1, "critic": 1}
    lines = (tmp_path / "ledger.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 5


def test_load_prices_from_chi_bench_format(tmp_path):
    p = tmp_path / "prices.yaml"
    p.write_text("prices:\n  anthropic/claude-new:\n    input: 2.0\n    output: 10.0\n    cache: 0.2\n",
                 encoding="utf-8")
    prices = load_prices(p)
    assert prices["anthropic/claude-new"]["output"] == 10.0
    assert "anthropic/claude-sonnet-5" in prices

