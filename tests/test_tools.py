import re

from fiable_agent.rag import BM25Retriever, Chunk
from fiable_agent.tools import LocalToolRoots, LocalTools, ToolNameMap, to_api_name

API_RE = re.compile(r"^[a-zA-Z0-9_-]{1,64}$")


def test_dotted_names_are_sanitized_and_roundtrip():
    m = ToolNameMap()
    for name in ("review.submit_md_case_decision", "payer_letter_center.generate_partial_approval",
                 "cm_care_plan.finalize", "x" * 80 + ".y"):
        api = m.register(name)
        assert API_RE.match(api), api
        assert m.canonical(api) == name
    assert to_api_name("auth.submit_authorization") == "auth__submit_authorization"


def _roots(tmp_path):
    hb = tmp_path / "skills" / "handbook"
    ws = tmp_path / "logs" / "artifacts" / "workspaces"
    hb.mkdir(parents=True)
    ws.mkdir(parents=True)
    (hb / "SKILL.md").write_text("# Skill\nRoute to sub-books.", encoding="utf-8")
    return LocalToolRoots(handbook=str(hb), readable=(str(tmp_path / "skills"), str(ws)), writable=(str(ws),)), hb, ws


def test_read_allowed_and_denied(tmp_path):
    roots, hb, ws = _roots(tmp_path)
    tools = LocalTools(roots=roots)
    ok = tools.call("read_file", {"path": str(hb / "SKILL.md")})
    assert ok.ok and "Route to sub-books" in ok.text
    secret = tmp_path / "other" / "expectations.json"
    secret.parent.mkdir()
    secret.write_text("{}", encoding="utf-8")
    assert not tools.call("read_file", {"path": str(secret)}).ok
    sneaky = ws / "CASE-1" / "tests"
    sneaky.mkdir(parents=True)
    (sneaky / "x.md").write_text("no", encoding="utf-8")
    assert not tools.call("read_file", {"path": str(sneaky / "x.md")}).ok
    assert not tools.call("read_file", {"path": str(hb / ".." / ".." / "other" / "expectations.json")}).ok


def test_write_only_under_workspace(tmp_path):
    roots, hb, ws = _roots(tmp_path)
    tools = LocalTools(roots=roots)
    out = tools.call("write_workspace_file", {"path": str(ws / "CASE-1" / "payer" / "notes.md"), "content": "hi"})
    assert out.ok and (ws / "CASE-1" / "payer" / "notes.md").read_text(encoding="utf-8") == "hi"
    assert not tools.call("write_workspace_file", {"path": str(hb / "evil.md"), "content": "x"}).ok


def test_handbook_search_with_and_without_index(tmp_path):
    roots, _, _ = _roots(tmp_path)
    assert "unavailable" in LocalTools(roots=roots).call("handbook_search", {"query": "pend"}).text
    idx = BM25Retriever([Chunk("um/pend.md", "Pend letters", "A pend letter lists missing information.")])
    out = LocalTools(idx, roots=roots).call("handbook_search", {"query": "pend letter"})
    assert out.ok and "[um/pend.md § Pend letters]" in out.text

