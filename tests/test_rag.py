"""RAG with a SYNTHETIC corpus (it neither contains nor derives from the real Handbook)."""

from fiable_agent.config import RagConfig
from fiable_agent.rag import (
    BM25Retriever,
    NullRetriever,
    build_retriever,
    format_hits,
    split_markdown,
    tokenize,
)


def _corpus(tmp_path):
    root = tmp_path / "handbook"
    (root / "references" / "payer").mkdir(parents=True)
    (root / "references" / "care-manager").mkdir(parents=True)
    (root / "SKILL.md").write_text("# Router\nRead the sub-book for your role.\n", encoding="utf-8")
    (root / "references" / "payer" / "letters.md").write_text(
        "# Letters\n## Denial letter\nA denial letter states the criterion not met and appeal rights.\n"
        "## Pend letter\nA pend letter lists the missing documentation and the deadline.\n", encoding="utf-8")
    (root / "references" / "care-manager" / "risk.md").write_text(
        "# Risk stratification\n## Moderate band\nPredictive score between 0.40 and 0.70 is moderate.\n",
        encoding="utf-8")
    return root


def test_tokenize_and_split():
    assert "gad-7" in tokenize("GAD-7 = 12")
    chunks = split_markdown("a.md", "# T\n" + ("word " * 1000), chunk_chars=500)
    assert len(chunks) >= 9 and all(len(c.text) <= 500 for c in chunks)


def test_bm25_ranks_relevant_section_first(tmp_path):
    idx = BM25Retriever.from_directory(_corpus(tmp_path), chunk_chars=400)
    hits = idx.search("pend letter missing documentation deadline", k=2)
    assert hits and hits[0].chunk.section == "Pend letter"
    hits = idx.search("predictive score moderate risk", k=1)
    assert hits[0].chunk.doc_id == "references/care-manager/risk.md"
    assert idx.search("zzzz-unknown-term") == []
    text = format_hits(hits)
    assert text.startswith("[references/care-manager/risk.md § Moderate band]")


def test_build_retriever_stub_when_handbook_missing(tmp_path):
    r = build_retriever(RagConfig(handbook_dir=str(tmp_path / "missing")))
    assert isinstance(r, NullRetriever) and not r.available and r.search("x") == []
    assert isinstance(build_retriever(RagConfig(enabled=False)), NullRetriever)


def test_build_retriever_bm25_and_faiss_fallback(tmp_path):
    root = _corpus(tmp_path)
    r = build_retriever(RagConfig(handbook_dir=str(root)))
    assert r.name == "bm25" and r.available
    # faiss without an embedding function -> BM25 (no implicit downloads)
    r2 = build_retriever(RagConfig(handbook_dir=str(root), backend="faiss"))
    assert r2.name == "bm25"

