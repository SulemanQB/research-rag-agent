from src.chunker import chunk_text


def test_chunk_text_splits_into_multiple_chunks():
    text = "word " * 2000
    chunks = chunk_text(text, chunk_size=200, overlap=50)

    assert len(chunks) > 1
    assert all("text" in chunk for chunk in chunks)
    assert all(len(chunk["text"]) <= 250 for chunk in chunks)


def test_chunk_text_preserves_overlap():
    text = "alpha beta gamma delta epsilon zeta eta theta iota kappa " * 50
    chunks = chunk_text(text, chunk_size=80, overlap=20)

    assert len(chunks) > 1
    assert chunks[0]["text"] != chunks[1]["text"]
    assert chunks[0]["metadata"]["chunk_id"]
