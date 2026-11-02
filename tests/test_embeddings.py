"""The embedding adapter, tested with a tiny deterministic encoder (no model download, no network)."""

import hashlib
import math
from pathlib import Path

import pytest

from agent_starter.embeddings import EmbeddingIndex
from agent_starter.retrieval import BM25Index, Retriever, build_retriever, tokenize
from agent_starter.tools import read_document_impl, search_docs_impl

DOCS = Path(__file__).resolve().parents[1] / "docs-example"
DIM = 256


def test_cache_reuses_only_unchanged_chunks(tmp_path):
    calls = []

    def encoder(texts):
        calls.append(list(texts))
        return hashing_encoder(texts)

    docs = {"a.md": "# First\n\nalpha", "b.md": "# Second\n\nbeta"}
    kwargs = dict(encoder=encoder, cache_dir=tmp_path, cache_key="hash-v1")
    first = EmbeddingIndex(docs, **kwargs)
    second = EmbeddingIndex(docs, **kwargs)
    assert len(calls) == 1
    assert first._vectors == second._vectors
    EmbeddingIndex({**docs, "b.md": "# Second\n\ngamma"}, **kwargs)
    assert calls[-1] == ["Second\ngamma"]
    assert len(calls) == 2
    EmbeddingIndex(docs, **{**kwargs, "cache_key": "hash-v2"})
    assert len(calls[-1]) == 2


def test_injected_encoder_without_identity_is_not_cached(tmp_path):
    calls = []
    def encoder(texts):
        calls.append(list(texts))
        return hashing_encoder(texts)
    for _ in range(2):
        EmbeddingIndex({"a.md": "alpha"}, encoder=encoder, cache_dir=tmp_path)
    assert len(calls) == 2
    assert list(tmp_path.iterdir()) == []


def test_damaged_cache_is_rebuilt(tmp_path):
    calls = []
    def encoder(texts):
        calls.append(list(texts))
        return hashing_encoder(texts)
    kwargs = dict(encoder=encoder, cache_dir=tmp_path, cache_key="v1")
    first = EmbeddingIndex({"a.md": "alpha"}, **kwargs)
    next(tmp_path.glob("*.json")).write_text("broken", encoding="utf-8")
    second = EmbeddingIndex({"a.md": "alpha"}, **kwargs)
    assert len(calls) == 2
    assert first._vectors == second._vectors


def hashing_encoder(texts):
    """Bag-of-words hashed into a fixed-size vector: crude, but deterministic and dependency-free."""
    vectors = []
    for text in texts:
        vector = [0.0] * DIM
        for token in tokenize(text):
            vector[int(hashlib.md5(token.encode()).hexdigest(), 16) % DIM] += 1.0
        vectors.append(vector)
    return vectors


@pytest.fixture(scope="module")
def index():
    return EmbeddingIndex.from_folder(DOCS, encoder=hashing_encoder)


def test_both_backends_satisfy_the_retriever_protocol(index):
    assert isinstance(index, Retriever)
    assert isinstance(BM25Index({}), Retriever)


def test_same_chunks_as_bm25_so_citation_tags_are_stable(index):
    assert [c.ref for c in index.chunks] == [c.ref for c in BM25Index.from_folder(DOCS).chunks]


def test_search_ranks_by_cosine_similarity(index):
    hits = index.search("data stored European Union Frankfurt", k=3)
    assert hits[0][0].ref == "security.md#2"
    scores = [s for _, s in hits]
    assert scores == sorted(scores, reverse=True)
    assert all(-1.0 - 1e-9 <= s <= 1.0 + 1e-9 for s in scores)


def test_min_score_filters_unrelated_passages(index):
    assert index.search("zzz qqq") == []
    assert index.search("   ") == []


def test_k_limits_results(index):
    assert len(index.search("plan", k=2)) <= 2


def test_empty_index_is_safe():
    assert EmbeddingIndex({}, encoder=hashing_encoder).search("anything") == []


def test_zero_vectors_do_not_crash():
    index = EmbeddingIndex({"a.md": "# A\n\nText."}, encoder=lambda texts: [[0.0, 0.0] for _ in texts])
    assert index.search("anything") == []
    assert math.isfinite(len(index.chunks))


def test_tools_work_unchanged_on_top_of_embeddings(index):
    assert search_docs_impl(index, "students teachers discount").startswith("[plans.md#4]")
    assert "Refund window" in read_document_impl(index, "refunds.md")


def test_build_retriever_selects_by_name():
    assert isinstance(build_retriever(DOCS, "bm25"), BM25Index)
    assert isinstance(build_retriever(DOCS, "embeddings", encoder=hashing_encoder), EmbeddingIndex)
    with pytest.raises(ValueError, match="Unknown retriever"):
        build_retriever(DOCS, "nope")

