from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Dict, List, Tuple

from pypdf import PdfReader

from src.chunker import chunk_text
from src.vector_store import LocalVectorStore


def find_corpus_files(corpus_dir: str | Path) -> List[Path]:
    """Return supported corpus files below a directory in stable order."""
    root = Path(corpus_dir)
    if not root.exists():
        return []
    files = []
    for path in root.rglob("*"):
        if path.is_file() and path.suffix.lower() in {".pdf", ".txt", ".md"}:
            files.append(path)
    return sorted(files)


def _hash_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:16]


def _read_text_file(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def extract_pdf_pages(path: Path) -> List[str]:
    """Extract one text string per PDF page, preserving page boundaries."""
    reader = PdfReader(str(path))
    pages: List[str] = []
    for page in reader.pages:
        content = page.extract_text() or ""
        pages.append(content.strip())
    return pages


def extract_pdf_text(path: Path) -> Tuple[str, int]:
    """Return page-marked PDF text and its page count."""
    pages = extract_pdf_pages(path)
    text = "\n".join(f"\n--- Page {index} ---\n{page}" for index, page in enumerate(pages, start=1))
    return text, len(pages)


def build_document_record(file_path: Path) -> Dict[str, Any]:
    """Read one supported file into text, pages, and provenance metadata."""
    file_name = file_path.name
    if file_path.suffix.lower() == ".pdf":
        pages = extract_pdf_pages(file_path)
        return {
            "source": str(file_path),
            "filename": file_name,
            "title": file_name,
            "page_count": len(pages),
            "pages": pages,
            "text": "\n".join(pages),
            "file_type": "pdf",
        }
    text = _read_text_file(file_path)
    return {
        "source": str(file_path),
        "filename": file_name,
        "title": file_name,
        "page_count": 1,
        "pages": [text],
        "text": text,
        "file_type": file_path.suffix.lower().lstrip("."),
    }
def ingest_corpus(corpus_dir: str | Path | None = None, reset: bool = False) -> Dict[str, Any]:
    """Extract, chunk, embed, and persist the local research corpus."""
    project_root = Path(__file__).resolve().parents[1]
    corpus_root = Path(corpus_dir) if corpus_dir else project_root / "data" / "corpus"
    files = find_corpus_files(corpus_root)

    if not files:
        return {
            "documents_found": 0,
            "pages_extracted": 0,
            "chunks_created": 0,
            "chunks_stored": 0,
            "document_names": [],
            "status": "NO_CORPUS",
        }

    documents: List[Dict[str, Any]] = []
    failures: List[str] = []
    for path in files:
        try:
            documents.append(build_document_record(path))
        except Exception as exc:
            failures.append(f"{path.name}: {exc}")
            print(f"Skipping {path.name}: {exc}")

    if not documents:
        return {
            "documents_found": 0,
            "pages_extracted": 0,
            "chunks_created": 0,
            "chunks_stored": 0,
            "document_names": [],
            "errors": failures,
            "status": "NO_READABLE_CORPUS",
        }

    total_pages = sum(doc["page_count"] for doc in documents)
    all_chunks: List[Dict[str, Any]] = []
    name_counts: Dict[str, int] = {}
    for document in documents:
        name_counts[document["filename"]] = name_counts.get(document["filename"], 0) + 1

    for document in documents:
        source_name = document["filename"]
        if name_counts[source_name] > 1:
            source_name = str(Path(document["source"]).relative_to(corpus_root))
        for page_number, page_text in enumerate(document["pages"], start=1):
            chunks = chunk_text(page_text, chunk_size=1000, overlap=150, source=source_name)
            for chunk_index, chunk in enumerate(chunks):
                preview = chunk["text"][:100]
                identity = f"{document['source']}:{page_number}:{chunk_index}:{preview}"
                chunk_id = f"{_hash_text(identity)}_{page_number:03d}_{chunk_index:03d}"
                chunk["id"] = chunk_id
                chunk["metadata"].update({
                    "source": source_name,
                    "filename": document["filename"],
                    "document_path": document["source"],
                    "page": page_number,
                    "title": document["title"],
                    "chunk_id": chunk_id,
                })
                all_chunks.append(chunk)

    vector_store = LocalVectorStore()
    if reset:
        vector_store.reset()

    stored = vector_store.add_documents(all_chunks)
    print(f"Found {len(documents)} documents.")
    print(f"Extracted {total_pages} pages.")
    print(f"Created {len(all_chunks)} chunks.")
    print("Generating embeddings...")
    print(f"Stored {stored} chunks in Chroma.")

    return {
        "documents_found": len(documents),
        "documents_failed": len(failures),
        "pages_extracted": total_pages,
        "chunks_created": len(all_chunks),
        "chunks_stored": stored,
        "document_names": [doc["filename"] for doc in documents],
        "errors": failures,
        "status": "OK",
    }
