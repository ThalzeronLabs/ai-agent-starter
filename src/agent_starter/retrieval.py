"""Document loading, chunking and retrieval: the "R" in RAG.

Everything the agent needs from retrieval is the small `Retriever` protocol below. The default implementation,
`BM25Index`, has zero external dependencies: BM25 is a classic lexical ranking function (the one search engines used
before embeddings). It needs no API key, no vector database and no GPU, ranks well on technical docs where exact
terms matter, and is fully deterministic, which makes the whole agent testable offline.

`agent_starter.embeddings.EmbeddingIndex` is an optional, local alternative (``pip install "agent-starter[embeddings]"``)
that implements the same protocol; `build_retriever` picks one by name.
"""

from __future__ import annotations

import math
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, runtime_checkable

TEXT_SUFFIXES = {".md", ".txt", ".rst"}

_STOPWORDS = {
    # English
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "how", "in", "is", "it", "of", "on", "or",
    "that", "the", "this", "to", "was", "what", "when", "where", "which", "who", "why", "with", "do", "does", "can",
    # Portuguese
    "o", "os", "as", "um", "uma", "de", "do", "da", "dos", "das", "e", "em", "no", "na", "nos", "nas", "que",
    "para", "por", "com", "se", "ao", "como", "qual", "quais",
}


@dataclass(frozen=True)
class Chunk:
    source: str
    index: int
    heading: str
    text: str

    @property
    def ref(self) -> str:
        """A short citation tag the model can quote, e.g. `refunds.md#2`."""
        return f"{self.source}#{self.index}"


def tokenize(text: str) -> list[str]:
    """Lowercase, strip accents (so "cancelação" matches "cancelacao"), split on non-alphanumerics, drop stopwords."""
    folded = unicodedata.normalize("NFKD", text.lower())
    folded = "".join(ch for ch in folded if not unicodedata.combining(ch))
    return [t for t in re.findall(r"[a-z0-9]+", folded) if t not in _STOPWORDS and len(t) > 1]


def chunk_document(source: str, text: str, max_chars: int = 900) -> list[Chunk]:
    """Splits a document by Markdown headings, then packs paragraphs into chunks of at most ~max_chars."""
    chunks: list[Chunk] = []
    heading = ""
    buffer: list[str] = []

    def flush() -> None:
        body = "\n\n".join(buffer).strip()
        if body:
            chunks.append(Chunk(source, len(chunks), heading, body))
        buffer.clear()

    for block in re.split(r"\n\s*\n", text):
        block = block.strip()
        if not block:
            continue
        match = re.match(r"^#{1,6}\s+(.*)", block)
        if match:
            flush()
            heading = match.group(1).strip()
            rest = block[match.end():].strip()
            if rest:
                buffer.append(rest)
            continue
        if buffer and sum(len(b) for b in buffer) + len(block) > max_chars:
            flush()
        buffer.append(block)
    flush()
    return chunks


def load_folder(folder: str | Path) -> dict[str, str]:
    """Reads every .md/.txt/.rst file under `folder` (recursively). Keys are paths relative to the folder."""
    root = Path(folder)
    if not root.is_dir():
        raise FileNotFoundError(f"No such folder: {root}")
    return {
        path.relative_to(root).as_posix(): path.read_text(encoding="utf-8", errors="replace")
        for path in sorted(root.rglob("*"))
        if path.is_file() and path.suffix.lower() in TEXT_SUFFIXES
    }


@runtime_checkable
class Retriever(Protocol):
    """What the tools need from a search backend. Implement these three members to plug in your own."""

    documents: dict[str, str]
    """Full text of every document, keyed by its name (the part of a citation tag before `#`)."""

    chunks: list[Chunk]
    """Every searchable passage."""

    def search(self, query: str, k: int = 4) -> list[tuple[Chunk, float]]:
        """Top-k `(chunk, score)` pairs for `query`, best first. May return fewer than k, or none."""
        ...


RETRIEVERS = ("bm25", "embeddings")


def build_retriever(folder: str | Path, kind: str = "bm25", **kwargs) -> Retriever:
    """Index `folder` with the retriever named `kind` (one of `RETRIEVERS`). Extra kwargs go to its constructor."""
    if kind == "bm25":
        return BM25Index.from_folder(folder, **kwargs)
    if kind == "embeddings":
        from .embeddings import EmbeddingIndex  # optional extra: imported only when asked for

        return EmbeddingIndex.from_folder(folder, **kwargs)
    raise ValueError(f"Unknown retriever {kind!r}; choose one of {', '.join(RETRIEVERS)}")


class BM25Index:
    """Okapi BM25 over chunks. `k1` controls term-frequency saturation, `b` length normalization."""

    def __init__(self, documents: dict[str, str], k1: float = 1.5, b: float = 0.75, max_chars: int = 900):
        self.documents = documents
        self.chunks = [c for source, text in documents.items() for c in chunk_document(source, text, max_chars)]
        self.k1, self.b = k1, b
        self._terms = [Counter(tokenize(f"{c.heading} {c.text}")) for c in self.chunks]
        self._lengths = [sum(t.values()) for t in self._terms]
        self._avg_len = (sum(self._lengths) / len(self._lengths)) if self._lengths else 0.0
        df: Counter[str] = Counter()
        for terms in self._terms:
            df.update(terms.keys())
        n = len(self.chunks)
        self._idf = {term: math.log(1 + (n - freq + 0.5) / (freq + 0.5)) for term, freq in df.items()}

    @classmethod
    def from_folder(cls, folder: str | Path, **kwargs) -> "BM25Index":
        return cls(load_folder(folder), **kwargs)

    def search(self, query: str, k: int = 4) -> list[tuple[Chunk, float]]:
        """Top-k chunks for `query`, best first. Chunks that share no term with the query are never returned."""
        if k <= 0:
            return []
        query_terms = tokenize(query)
        if not query_terms or not self.chunks:
            return []
        scored = []
        for i, terms in enumerate(self._terms):
            score = 0.0
            for term in query_terms:
                tf = terms.get(term, 0)
                if tf == 0:
                    continue
                norm = tf + self.k1 * (1 - self.b + self.b * self._lengths[i] / (self._avg_len or 1))
                score += self._idf.get(term, 0.0) * tf * (self.k1 + 1) / norm
            if score > 0:
                scored.append((self.chunks[i], score))
        scored.sort(key=lambda pair: pair[1], reverse=True)
        return scored[:k]
