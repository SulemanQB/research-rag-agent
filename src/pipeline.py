from __future__ import annotations

from typing import Any, Dict, List

from src.generator import LLMClient, build_extractive_answer, build_general_prompt, build_rag_prompt
from src.ingest import ingest_corpus
from src.router import route_query
from src.vector_store import LocalVectorStore


def _format_sources(results: List[Dict[str, Any]]) -> List[str]:
    sources: List[str] = []
    for result in results:
        meta = result.get("metadata", result)
        source = meta.get("source") or meta.get("filename") or "unknown"
        page = meta.get("page") or meta.get("chunk_id") or "n/a"
        citation = f"{source} — page {page}"
        if citation not in sources:
            sources.append(citation)
    return sources


def answer_question(question: str, top_k: int = 3) -> Dict[str, Any]:
    """Route, retrieve, and answer one question with source metadata."""
    route_info = route_query(question)
    route = route_info["route"]

    if route == "CORPUS_SEARCH":
        vector_store = LocalVectorStore()
        query_result = vector_store.query(question, top_k=top_k)
        documents = query_result.get("documents", [[]])[0]
        metadata = query_result.get("metadatas", [[]])[0]
        formatted_context = []
        for idx, (text, meta) in enumerate(zip(documents, metadata), start=1):
            source_name = meta.get("source") or meta.get("filename") or "unknown"
            page = meta.get("page") or 1
            formatted_context.append(f"[{idx}] Source: {source_name}, page {page}\n{text}\n")
        context = "\n".join(formatted_context)

        if not context.strip():
            return {
                "route": route,
                "reason": route_info["reason"],
                "sources": [],
                "answer": "The corpus does not contain enough information to answer this question.",
            }

        llm = LLMClient()
        try:
            answer = llm.generate(build_rag_prompt(question, context))
        except RuntimeError as exc:
            fallback_answer = build_extractive_answer(question, context)
            return {
                "route": route,
                "reason": route_info["reason"],
                "sources": _format_sources(metadata),
                "answer": f"{fallback_answer}\n\nModel status: {exc}",
            }

        return {
            "route": route,
            "reason": route_info["reason"],
            "sources": _format_sources(metadata),
            "answer": answer,
        }

    llm = LLMClient()
    try:
        answer = llm.generate(build_general_prompt(question))
    except RuntimeError as exc:
        return {
            "route": route,
            "reason": route_info["reason"],
            "sources": [],
            "answer": str(exc),
        }

    return {
        "route": route,
        "reason": route_info["reason"],
        "sources": [],
        "answer": answer,
    }


def run_ingestion(reset: bool = False) -> Dict[str, Any]:
    """Run corpus ingestion using the project-local data directory."""
    return ingest_corpus(reset=reset)
