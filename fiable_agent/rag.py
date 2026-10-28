"""RAG over the Managed-Care Operations Handbook, with an interchangeable interface.

* ``NullRetriever``  - stub that works without the Handbook (returns []).
* ``BM25Retriever``  - local lexical index in pure Python (no dependencies).
                       Built AT RUNTIME from the Handbook mounted in the
                       container; no derived index is distributed
                       (restricted license: Rules section 5).
* ``FaissRetriever`` - dense, optional: requires ``faiss`` and an injected
                       embedding function (e.g. a local model on a GPU).

Citations: each hit carries ``doc_id`` (relative path) and ``section``
(heading), so the agent cites section identifiers instead of copying text.
"""

from __future__ import annotations

import logging
import math
import re
from collections import Counter
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

logger = logging.getLogger(__name__)

_TOKEN_RE = re.compile(r"[a-z0-9][a-z0-9_.\-]*")
_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)$")
_STOPWORDS = frozenset(
    "a an and are as at be by for from has have in is it its of on or that the this to was were will with "
    "you your we our they their not no if then than into per via".split()
)


def tokenize(text: str) -> list[str]:
    tokens = []
    for tok in _TOKEN_RE.findall(text.lower()):
        tok = tok.strip(".-_")
        if tok and tok not in _STOPWORDS and len(tok) > 1:
            tokens.append(tok)
    return tokens


@dataclass(frozen=True)
class Chunk:
    doc_id: str
    section: str
    text: str


@dataclass(frozen=True)
class Hit:
    chunk: Chunk
    score: float


class Retriever(Protocol):
    name: str

    @property
    def available(self) -> bool: ...

    def search(self, query: str, k: int = 4) -> list[Hit]: ...


class NullRetriever:
    """Stub without a Handbook: the agent keeps working, just without RAG context."""

    name = "null"

    def __init__(self, reason: str = "handbook not available"):
        self.reason = reason

    @property
    def available(self) -> bool:
        return False

    def search(self, query: str, k: int = 4) -> list[Hit]:
        return []


def split_markdown(doc_id: str, text: str, chunk_chars: int = 1800) -> list[Chunk]:
    """Split by headings and, if needed, by size (respecting paragraphs)."""
    sections: list[tuple[str, list[str]]] = []
    current_title = Path(doc_id).stem
    current: list[str] = []
    for line in text.splitlines():
        match = _HEADING_RE.match(line)
        if match:
            if current and "".join(current).strip():
                sections.append((current_title, current))
            current_title = match.group(2).strip() or current_title
            current = [line]
        else:
            current.append(line)
    if current and "".join(current).strip():
        sections.append((current_title, current))

    chunks: list[Chunk] = []
    for title, lines in sections:
        body = "\n".join(lines).strip()
        if len(body) <= chunk_chars:
            chunks.append(Chunk(doc_id, title, body))
            continue
        buf = ""
        for para in re.split(r"\n\s*\n", body):
            if buf and len(buf) + len(para) + 2 > chunk_chars:
                chunks.append(Chunk(doc_id, title, buf.strip()))
                buf = ""
            if len(para) > chunk_chars:  # giant paragraph: hard cut
                for i in range(0, len(para), chunk_chars):
                    chunks.append(Chunk(doc_id, title, para[i:i + chunk_chars]))
                continue
            buf = f"{buf}\n\n{para}" if buf else para
        if buf.strip():
            chunks.append(Chunk(doc_id, title, buf.strip()))
    return chunks


def iter_markdown_chunks(root: str | Path, chunk_chars: int = 1800, max_files: int = 5000) -> Iterable[Chunk]:
    root = Path(root)
    files = sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in {".md", ".markdown", ".txt"})
    for path in files[:max_files]:
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        yield from split_markdown(path.relative_to(root).as_posix(), text, chunk_chars)


class BM25Retriever:
    name = "bm25"

    def __init__(self, chunks: Sequence[Chunk], k1: float = 1.5, b: float = 0.75):
        self.chunks = list(chunks)
        self.k1 = k1
        self.b = b
        self._tfs: list[Counter[str]] = []
        self._lens: list[int] = []
        df: Counter[str] = Counter()
        for chunk in self.chunks:
            toks = tokenize(f"{chunk.section} {chunk.doc_id} {chunk.text}")
            tf = Counter(toks)
            self._tfs.append(tf)
            self._lens.append(len(toks))
            df.update(tf.keys())
        n = max(len(self.chunks), 1)
        self._avgdl = (sum(self._lens) / n) if self._lens else 0.0
        self._idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}
        self._postings: dict[str, list[int]] = {}
        for i, tf in enumerate(self._tfs):
            for term in tf:
                self._postings.setdefault(term, []).append(i)

    @classmethod
    def from_directory(cls, root: str | Path, chunk_chars: int = 1800, max_files: int = 5000) -> "BM25Retriever":
        return cls(list(iter_markdown_chunks(root, chunk_chars, max_files)))

    @property
    def available(self) -> bool:
        return bool(self.chunks)

    def search(self, query: str, k: int = 4) -> list[Hit]:
        terms = [t for t in tokenize(query) if t in self._idf]
        if not terms:
            return []
        scores: dict[int, float] = {}
        for term in set(terms):
            idf = self._idf[term]
            for i in self._postings.get(term, ()):
                tf = self._tfs[i][term]
                denom = tf + self.k1 * (1 - self.b + self.b * self._lens[i] / (self._avgdl or 1.0))
                scores[i] = scores.get(i, 0.0) + idf * tf * (self.k1 + 1) / denom
        ranked = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))[:k]
        return [Hit(self.chunks[i], s) for i, s in ranked]


EmbedFn = Callable[[list[str]], "object"]  # returns an (n, d) float32 array


class FaissRetriever:
    """Local dense index. ``embed_fn`` is injected (no implicit downloads)."""

    name = "faiss"

    def __init__(self, chunks: Sequence[Chunk], embed_fn: EmbedFn):
        try:
            import faiss  # type: ignore
            import numpy as np
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise ImportError("faiss backend requires 'faiss-cpu' (or faiss-gpu) and numpy") from exc
        self._np = np
        self.chunks = list(chunks)
        self.embed_fn = embed_fn
        vectors = np.asarray(embed_fn([f"{c.section}\n{c.text}" for c in self.chunks]), dtype="float32")
        faiss.normalize_L2(vectors)
        self._faiss = faiss
        self.index = faiss.IndexFlatIP(vectors.shape[1])
        self.index.add(vectors)

    @property
    def available(self) -> bool:
        return bool(self.chunks)

    def search(self, query: str, k: int = 4) -> list[Hit]:
        q = self._np.asarray(self.embed_fn([query]), dtype="float32")
        self._faiss.normalize_L2(q)
        scores, ids = self.index.search(q, min(k, len(self.chunks)))
        return [Hit(self.chunks[int(i)], float(s)) for s, i in zip(scores[0], ids[0]) if i >= 0]


def build_retriever(cfg, embed_fn: EmbedFn | None = None) -> Retriever:
    """Build the retriever according to ``RagConfig``; never raises on a missing Handbook."""
    if not cfg.enabled or cfg.backend == "null":
        return NullRetriever("rag disabled by config")
    root = Path(cfg.handbook_dir)
    if not root.is_dir():
        logger.warning("handbook dir not found (%s); using NullRetriever", root)
        return NullRetriever(f"handbook dir not found: {root}")
    chunks = list(iter_markdown_chunks(root, cfg.chunk_chars, cfg.max_files))
    if not chunks:
        return NullRetriever("handbook dir is empty")
    if cfg.backend == "faiss":
        if embed_fn is None:
            logger.warning("faiss backend requested without embed_fn; falling back to BM25")
        else:
            try:
                return FaissRetriever(chunks, embed_fn)
            except ImportError as exc:
                logger.warning("%s; falling back to BM25", exc)
    return BM25Retriever(chunks)


def format_hits(hits: Sequence[Hit], max_chars: int = 6000) -> str:
    out: list[str] = []
    used = 0
    for hit in hits:
        header = f"[{hit.chunk.doc_id} § {hit.chunk.section}]"
        room = max_chars - used - len(header) - 2
        if room <= 80:
            break
        body = hit.chunk.text[:room]
        out.append(f"{header}\n{body}")
        used += len(header) + len(body) + 2
    return "\n\n".join(out)

