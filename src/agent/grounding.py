"""Lexical groundedness check used by the reflection node and the eval."""

from __future__ import annotations

import re

# Tokens that do not carry a factual claim. The boilerplate words are included
# so the extractive RAG label does not dominate the overlap score.
STOPWORDS = frozenset(
    {
        "a",
        "an",
        "and",
        "answer",
        "are",
        "as",
        "at",
        "be",
        "been",
        "but",
        "by",
        "can",
        "context",
        "could",
        "did",
        "do",
        "does",
        "extracted",
        "for",
        "from",
        "had",
        "has",
        "have",
        "how",
        "if",
        "in",
        "into",
        "is",
        "it",
        "its",
        "llm",
        "not",
        "of",
        "on",
        "or",
        "page",
        "retrieved",
        "source",
        "sources",
        "than",
        "that",
        "the",
        "their",
        "them",
        "these",
        "this",
        "those",
        "to",
        "unavailable",
        "was",
        "were",
        "what",
        "when",
        "where",
        "which",
        "who",
        "with",
    }
)

_TOKEN = re.compile(r"[a-z0-9]+(?:\.[0-9]+)?")


def content_tokens(text: str) -> list[str]:
    """Return lowercased content tokens, keeping decimals such as ``17.5``."""
    return [token for token in _TOKEN.findall((text or "").lower()) if token not in STOPWORDS and len(token) > 1]


def score_groundedness(answer: str, evidence_text: str) -> float:
    """Return the fraction of answer content tokens that appear in the evidence.

    The score is 0 when the answer has no content tokens or the evidence is empty.
    It does not call a model.
    """
    answer_tokens = content_tokens(answer)
    if not answer_tokens:
        return 0.0
    evidence_tokens = set(content_tokens(evidence_text))
    if not evidence_tokens:
        return 0.0
    supported = sum(1 for token in answer_tokens if token in evidence_tokens)
    return supported / len(answer_tokens)


def evidence_text(evidence: list[dict]) -> str:
    """Join tool evidence into one string for the overlap check."""
    return "\n".join(str(item.get("text") or "") for item in evidence)


def refine_retrieval_query(question: str) -> str:
    """Build a shorter follow-up query from the question's content words."""
    tokens = content_tokens(question)
    refined = " ".join(tokens)
    if refined and refined != (question or "").strip().lower():
        return refined
    return ((question or "").strip() + " details").strip()
