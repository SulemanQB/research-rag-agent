from __future__ import annotations

from typing import Any, Dict, List


def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 150, source: str = "document") -> List[Dict[str, Any]]:
    """Split text into overlapping chunks with stable positional metadata."""
    if not text or not text.strip():
        return []
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if overlap < 0:
        raise ValueError("overlap must be non-negative")
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    normalized = "\n".join(line.rstrip() for line in text.splitlines()).strip()
    if not normalized:
        return []

    chunks: List[Dict[str, Any]] = []
    start = 0
    chunk_index = 0
    while start < len(normalized):
        end = min(len(normalized), start + chunk_size)
        segment = normalized[start:end].strip()
        if not segment:
            break

        chunk_id = f"{source.replace(' ', '_')}_{chunk_index:03d}"
        chunk = {
            "id": chunk_id,
            "text": segment,
            "metadata": {
                "source": source,
                "chunk_id": chunk_id,
                "start": start,
                "end": end,
            },
        }
        chunks.append(chunk)

        if end >= len(normalized):
            break
        start = max(0, end - overlap)
        chunk_index += 1

    return chunks
