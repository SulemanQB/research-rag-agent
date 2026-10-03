from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import chromadb

from src.embeddings import EmbeddingModel


class LocalVectorStore:
    """Persistent Chroma collection backed by the configured local path."""

    def __init__(self, persist_directory: Optional[str] = None, collection_name: Optional[str] = None, embedding_model: Optional[EmbeddingModel] = None):
        project_root = Path(__file__).resolve().parents[1]
        configured_path = persist_directory or os.getenv("CHROMA_PATH")
        self.persist_directory = configured_path or str(project_root / "chroma_db")
        self.collection_name = collection_name or os.getenv("COLLECTION_NAME", "research_corpus")
        self.embedding_model = embedding_model or EmbeddingModel()
        self.client = chromadb.PersistentClient(path=self.persist_directory)
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def reset(self) -> None:
        """Delete and recreate the configured collection."""
        collection_names = {
            getattr(collection, "name", collection)
            for collection in self.client.list_collections()
        }
        if self.collection_name in collection_names:
            self.client.delete_collection(self.collection_name)
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def add_documents(self, chunks: List[Dict[str, Any]]) -> int:
        """Embed and upsert chunk records, returning the number stored."""
        if not chunks:
            return 0
        ids = [chunk["id"] for chunk in chunks]
        texts = [chunk["text"] for chunk in chunks]
        metadatas = [chunk["metadata"] for chunk in chunks]
        embeddings = self.embedding_model.encode(texts)
        self.collection.upsert(documents=texts, embeddings=embeddings, metadatas=metadatas, ids=ids)
        return len(chunks)

    def query(self, query_text: str, top_k: int = 3) -> Dict[str, Any]:
        """Return the nearest chunks and metadata for a query."""
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        if self.collection.count() == 0:
            return {"documents": [[]], "metadatas": [[]], "distances": [[]]}
        embedding = self.embedding_model.encode_query(query_text)
        return self.collection.query(
            query_embeddings=[embedding],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )
