from src.retriever import rank_documents


def test_rank_documents_preserves_metadata():
    docs = [
        {"id": "doc-a", "text": "federated learning is common", "metadata": {"source": "paper_a.pdf", "page": 2}},
        {"id": "doc-b", "text": "clustering adapts updates", "metadata": {"source": "paper_b.pdf", "page": 4}},
    ]

    ranked = rank_documents(docs, query="federated learning", top_k=2)

    assert ranked[0]["metadata"]["source"] == "paper_a.pdf"
    assert ranked[0]["metadata"]["page"] == 2


def test_rank_documents_accepts_top_k_limit():
    docs = [
        {"id": "d1", "text": "alpha beta gamma", "metadata": {"source": "s1.pdf", "page": 1}},
        {"id": "d2", "text": "alpha beta delta", "metadata": {"source": "s2.pdf", "page": 2}},
        {"id": "d3", "text": "omega theta", "metadata": {"source": "s3.pdf", "page": 3}},
    ]

    ranked = rank_documents(docs, query="alpha", top_k=2)

    assert len(ranked) == 2
    assert ranked[0]["id"] in {"d1", "d2"}
