from pathlib import Path

import src.ingest as ingest


class _FakePage:
    def __init__(self, text: str):
        self.text = text

    def extract_text(self) -> str:
        return self.text


class _FakeReader:
    def __init__(self, path: str):
        self.pages = [_FakePage("first page"), _FakePage("second page")]


def test_extract_pdf_pages_preserves_page_boundaries(monkeypatch, tmp_path: Path):
    monkeypatch.setattr(ingest, "PdfReader", _FakeReader)

    pages = ingest.extract_pdf_pages(tmp_path / "paper.pdf")

    assert pages == ["first page", "second page"]


def test_text_document_record_has_single_page(tmp_path: Path):
    document_path = tmp_path / "notes.md"
    document_path.write_text("local notes", encoding="utf-8")

    record = ingest.build_document_record(document_path)

    assert record["page_count"] == 1
    assert record["pages"] == ["local notes"]


def test_ingest_only_resets_when_requested(monkeypatch, tmp_path: Path):
    document_path = tmp_path / "notes.md"
    document_path.write_text("local notes", encoding="utf-8")

    class _FakeStore:
        reset_calls = 0

        def reset(self):
            self.reset_calls += 1

        def add_documents(self, chunks):
            return len(chunks)

    store = _FakeStore()
    monkeypatch.setattr(ingest, "LocalVectorStore", lambda: store)

    ingest.ingest_corpus(tmp_path, reset=False)
    ingest.ingest_corpus(tmp_path, reset=True)

    assert store.reset_calls == 1
