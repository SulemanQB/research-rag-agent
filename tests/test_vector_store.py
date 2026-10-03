import pytest

from src.vector_store import LocalVectorStore


class _FakeEmbeddingModel:
    def encode(self, texts):
        return [[1.0, 0.0] for _ in texts]

    def encode_query(self, text):
        return [1.0, 0.0]


def test_empty_store_returns_empty_results(tmp_path):
    store = LocalVectorStore(
        persist_directory=str(tmp_path / "chroma"),
        collection_name="empty_collection",
        embedding_model=_FakeEmbeddingModel(),
    )

    result = store.query("anything")

    assert result["documents"] == [[]]
    assert result["metadatas"] == [[]]


def test_query_rejects_non_positive_top_k(tmp_path):
    store = LocalVectorStore(
        persist_directory=str(tmp_path / "chroma"),
        collection_name="validation_collection",
        embedding_model=_FakeEmbeddingModel(),
    )

    with pytest.raises(ValueError, match="top_k must be positive"):
        store.query("anything", top_k=0)
