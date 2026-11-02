import pytest
from agent_starter.retrieval import BM25Index
from agent_starter.embeddings import EmbeddingIndex

@pytest.mark.parametrize("k", [0, -1, -10])
def test_nonpositive_k_returns_no_passages(k):
    docs = {"a.md": "alpha", "b.md": "alpha beta"}
    for index in (BM25Index(docs), EmbeddingIndex(docs, encoder=lambda texts: [[1.0, 0.0] for _ in texts])):
        assert index.search("alpha", k=k) == []
