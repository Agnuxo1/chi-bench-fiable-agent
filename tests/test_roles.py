from fiable_agent.roles import ROLE_CM, ROLE_PAYER, ROLE_PROVIDER, detect_role, render_plan
from fiable_agent.state import ObservedState


def test_detect_role_by_tools_url_and_instruction():
    assert detect_role(tool_names=["review.submit_md_case_decision", "triage.get"]) == ROLE_PAYER
    assert detect_role(tool_names=["cm_intake.open_case"]) == ROLE_CM
    assert detect_role(tool_names=["cases.create_from_order", "auth.submit_authorization"]) == ROLE_PROVIDER
    assert detect_role(mcp_url="http://localhost:8100/mcp") == ROLE_PAYER
    assert detect_role(mcp_url="http://localhost:8200/mcp") == ROLE_CM
    assert detect_role(mcp_url="http://localhost:8020/mcp") == ROLE_PROVIDER
    assert detect_role("- Use only payer namespaces: `payer_intake_hub`, `triage`") == ROLE_PAYER
    assert detect_role("- Use only care-manager namespaces: `cm_intake`") == ROLE_CM


def test_render_plan_marks_completed_phases():
    state = ObservedState()
    state.record("cm_intake.open_case", {"referral_id": "REF-1"}, True, '{"case_id": "CM-CASE-1"}')
    plan = render_plan(ROLE_CM, state)
    assert "[x] 1. intake" in plan
    assert "[ ] 2. chart_review" in plan

