from __future__ import annotations

import os
from typing import List, Sequence

from sentence_transformers import SentenceTransformer


class EmbeddingModel:
    """Small wrapper around the configured local Sentence Transformer."""

    def __init__(self, model_name: str | None = None):
        self.model_name = model_name or os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
        self.model = SentenceTransformer(self.model_name)

    def encode(self, texts: Sequence[str]) -> List[List[float]]:
        """Return normalized embeddings for a sequence of texts."""
        embeddings = self.model.encode(list(texts), normalize_embeddings=True)
        if hasattr(embeddings, "tolist"):
            return embeddings.tolist()
        return [list(item) for item in embeddings]

    def encode_query(self, text: str) -> List[float]:
        """Return the normalized embedding for one query."""
        return self.encode([text])[0]


def get_embedding_model(model_name: str | None = None) -> EmbeddingModel:
    """Create an embedding model using an optional model-name override."""
    return EmbeddingModel(model_name=model_name)
