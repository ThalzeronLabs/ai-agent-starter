"""Optional dense retrieval: a local embedding model behind the same `Retriever` protocol as `BM25Index`.

Nothing here calls a paid API. By default the vectors come from a sentence-transformers model that runs on your
machine (``pip install "agent-starter[embeddings]"``; the model is downloaded once from Hugging Face and cached).
Any other encoder works too: pass a function that maps a list of strings to a list of vectors.

Search is brute-force cosine similarity in pure Python, which is plenty for a few thousand chunks. For more than
that, store the vectors in a vector index (FAISS, pgvector, ...) and keep the same `search` signature.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from pathlib import Path

from .retrieval import Chunk, chunk_document, load_folder

Encoder = Callable[[Sequence[str]], Sequence[Sequence[float]]]

DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def sentence_transformers_encoder(model_name: str = DEFAULT_EMBEDDING_MODEL) -> Encoder:
    """An `Encoder` backed by a local sentence-transformers model. Requires the `embeddings` extra."""
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as error:  # pragma: no cover - exercised only without the extra installed
        raise ImportError(
            'Embeddings need the optional extra: pip install "agent-starter[embeddings]"'
        ) from error
    model = SentenceTransformer(model_name)

    def encode(texts: Sequence[str]) -> list[list[float]]:
        return model.encode(list(texts), normalize_embeddings=True).tolist()

    return encode


def _unit(vector: Sequence[float]) -> list[float]:
    norm = math.sqrt(sum(x * x for x in vector))
    return [x / norm for x in vector] if norm else [0.0 for _ in vector]


class EmbeddingIndex:
    """Cosine-similarity search over chunk embeddings.

    Args:
        documents: Document name -> full text, as returned by `load_folder`.
        encoder: Maps texts to vectors. Defaults to `sentence_transformers_encoder(model_name)`.
        model_name: Which sentence-transformers model to load when no `encoder` is given.
        max_chars: Soft chunk size, as in `BM25Index`.
        min_score: Chunks whose cosine similarity is at or below this are never returned, so an unrelated
            question can still come back empty and the agent can say "not covered".
    """

    def __init__(
        self,
        documents: dict[str, str],
        encoder: Encoder | None = None,
        model_name: str = DEFAULT_EMBEDDING_MODEL,
        max_chars: int = 900,
        min_score: float = 0.2,
    ):
        self.documents = documents
        self.chunks = [c for source, text in documents.items() for c in chunk_document(source, text, max_chars)]
        self.min_score = min_score
        self._encode = encoder or sentence_transformers_encoder(model_name)
        texts = [f"{c.heading}\n{c.text}" if c.heading else c.text for c in self.chunks]
        self._vectors = [_unit(v) for v in self._encode(texts)] if texts else []

    @classmethod
    def from_folder(cls, folder: str | Path, **kwargs) -> "EmbeddingIndex":
        return cls(load_folder(folder), **kwargs)

    def search(self, query: str, k: int = 4) -> list[tuple[Chunk, float]]:
        """Top-k chunks by cosine similarity to `query`, best first."""
        if not query.strip() or not self.chunks:
            return []
        q = _unit(self._encode([query])[0])
        if any(len(vector) != len(q) for vector in self._vectors):
            raise ValueError("Query and chunk embeddings must have matching dimensions")
        scored = []
        for chunk, vector in zip(self.chunks, self._vectors):
            score = sum(a * b for a, b in zip(q, vector))
            if score > self.min_score:
                scored.append((chunk, score))
        scored.sort(key=lambda pair: pair[1], reverse=True)
        return scored[:k]

