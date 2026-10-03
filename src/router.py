from __future__ import annotations

from typing import Dict, List

CORPUS_HINTS = [
    "paper",
    "papers",
    "corpus",
    "document",
    "documents",
    "research",
    "study",
    "article",
    "survey",
    "abstract",
    "pdf",
    "notes",
    "according to",
    "in my corpus",
    "uploaded",
    "retrieved",
    "local documents",
    "what does my",
    "nist",
    "ai risk management",
    "generative ai",
    "risk management framework",
]

GENERAL_HINTS = [
    "capital of",
    "who is",
    "what is the meaning of",
    "history",
    "math",
    "python",
    "weather",
    "population",
    "language",
    "date",
    "time",
    "formula",
]


def route_query(query: str) -> Dict[str, str]:
    """Select corpus retrieval or general answering using transparent heuristics."""
    q = (query or "").strip().lower()
    if not q:
        return {"route": "GENERAL_ANSWER", "reason": "No query text was provided."}

    corpus_match = any(hint in q for hint in CORPUS_HINTS)
    general_match = any(hint in q for hint in GENERAL_HINTS)

    if corpus_match and not general_match:
        return {
            "route": "CORPUS_SEARCH",
            "reason": "The query refers to information expected to exist in the local research corpus.",
        }
    if corpus_match and general_match:
        return {
            "route": "CORPUS_SEARCH",
            "reason": "The query is asking for corpus-grounded research information as well as general knowledge.",
        }
    return {
        "route": "GENERAL_ANSWER",
        "reason": "The query is a general factual question and does not clearly require corpus retrieval.",
    }
