import pytest
from agent_starter.embeddings import EmbeddingIndex

def test_dimension_mismatch_is_not_silently_truncated():
    def encoder(texts):
        return [[1.0, 0.0] if text != "query" else [1.0] for text in texts]
    index = EmbeddingIndex({"a.md": "alpha"}, encoder=encoder)
    with pytest.raises(ValueError, match="matching dimensions"):
        index.search("query")
