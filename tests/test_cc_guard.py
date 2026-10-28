"""Claude Code hook guard (fiable_agent.cc_guard): name mapping, blocking, evidence, closure gate."""

from __future__ import annotations

import io
import json
import sys

import pytest

from fiable_agent import cc_guard
from fiable_agent.cc_guard import GuardStore, build_append_prompt, canonical_tool, main


def make_store(tmp_path, **config):
    cfg = {"role": "payer", "instruction": "", "checklist": True, "critic": False,
           "max_blocks_per_action": 2, "finish_gate": True, "max_finish_nudges": 2, **config}
    (tmp_path / "config.json").write_text(json.dumps(cfg), encoding="utf-8")
    return GuardStore(tmp_path)


def pre(store, tool, args):
    return cc_guard.on_pre({"tool_name": tool, "tool_input": args}, store)


@pytest.mark.parametrize("raw, expected", [
    ("mcp__chi_bench__review_submit_md_case_decision", "review.submit_md_case_decision"),
    ("mcp__healthverse__payer_intake_hub_confirm_decisions", "payer_intake_hub.confirm_decisions"),
    ("mcp__chi_bench__cm_care_plan_finalize", "cm_care_plan.finalize"),
    ("mcp__chi_bench__review.submit_md_case_decision", "review.submit_md_case_decision"),
    ("Read", "local.Read"),
])
def test_canonical_tool(raw, expected):
    assert canonical_tool(raw) == expected


def test_response_text_flattens_content_blocks():
    blocks = [{"type": "text", "text": '{"case_id": "C-1"}'}, {"type": "text", "text": "done"}]
    assert cc_guard.response_text(blocks) == '{"case_id": "C-1"}\ndone'
    assert cc_guard.response_text({"content": blocks}).startswith('{"case_id"')


def test_pre_blocks_then_releases_after_limit(tmp_path):
    store = make_store(tmp_path)
    bad = {"case_id": "CASE-9", "decision": "deny"}
    tool = "mcp__chi_bench__review_submit_md_case_decision"
    code, msg = pre(store, tool, bad)
    assert code == cc_guard.BLOCK_EXIT and "UM-02" in msg and "NOT executed" in msg
    code, msg = pre(store, tool, bad)
    assert code == cc_guard.BLOCK_EXIT and "final warning" in msg
    assert pre(store, tool, bad) == (0, "")  # bounded: never stalls a trial


def test_pre_allows_compliant_and_read_only_calls(tmp_path):
    store = make_store(tmp_path)
    good = {"case_id": "CASE-9", "decision": "deny", "signed_off": True,
            "determination_rationale": "BMI 38 without comorbidity fails criterion 1.a",
            "denial_rationale": "Criterion 1.a not met", "clinical_notes": "BMI 38"}
    assert pre(store, "mcp__chi_bench__review_submit_md_case_decision", good) == (0, "")
    assert pre(store, "mcp__chi_bench__review_get_case", {"case_id": "CASE-9"}) == (0, "")
    assert pre(store, "Read", {"file_path": "/workspace/skills/x.md"}) == (0, "")


def test_checklist_disabled_is_a_no_op(tmp_path):
    store = make_store(tmp_path, checklist=False)
    assert pre(store, "mcp__chi_bench__review_submit_md_case_decision", {"case_id": "C"}) == (0, "")


def test_post_records_evidence_and_case_lock(tmp_path):
    store = make_store(tmp_path)
    cc_guard.on_post({"tool_name": "mcp__chi_bench__review_submit_nurse_case_recommendation",
                      "tool_input": {"case_id": "CASE-1", "recommendation": "approve"},
                      "tool_response": [{"type": "text", "text": '{"ok": true}'}]}, store)
    data = json.loads((tmp_path / "state.json").read_text(encoding="utf-8"))
    assert data["events"][0]["name"] == "review.submit_nurse_case_recommendation"
    assert data["locked_case_id"] == "CASE-1"
    # GLOBAL-01: a mutation on another case is now blocked
    code, msg = pre(store, "mcp__chi_bench__review_submit_md_case_decision",
                    {"case_id": "CASE-2", "decision": "p2p_review"})
    assert code == cc_guard.BLOCK_EXIT and "GLOBAL-01" in msg


def test_stop_gate_is_bounded(tmp_path):
    store = make_store(tmp_path, role="provider", max_finish_nudges=1)
    code, msg = cc_guard.on_stop({"stop_hook_active": False}, store)
    assert code == cc_guard.BLOCK_EXIT and "PROV-04" in msg
    assert cc_guard.on_stop({"stop_hook_active": True}, store) == (0, "")


def test_critic_arm_uses_heuristic_panel(tmp_path):
    store = make_store(tmp_path, critic=True)
    thin = {"case_id": "CASE-9", "decision": "deny", "signed_off": True, "determination_rationale": "no",
            "denial_rationale": "no", "clinical_notes": "x"}
    code, msg = pre(store, "mcp__chi_bench__review_submit_md_case_decision", thin)
    assert code == cc_guard.BLOCK_EXIT and "policy" in msg
    log = (tmp_path / "guard_log.jsonl").read_text(encoding="utf-8").splitlines()
    assert json.loads(log[-1])["verdicts"]


def test_main_fails_open_on_bad_payload(tmp_path, monkeypatch, capsys):
    make_store(tmp_path)
    monkeypatch.setattr(sys, "stdin", io.StringIO("{not json"))
    assert main(["pre", str(tmp_path)]) == 0
    assert "error" in (tmp_path / "guard_log.jsonl").read_text(encoding="utf-8")


def test_append_prompt_mentions_plan_and_guard():
    text = build_append_prompt("payer", with_checklist=True)
    assert "ROLE PLAN (payer)" in text and "BLOCKED" in text
    assert "BLOCKED" not in build_append_prompt("payer", with_checklist=False)
