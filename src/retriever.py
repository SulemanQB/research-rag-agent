from __future__ import annotations

import re
from typing import Any, Dict, List


def _tokenize(text: str) -> List[str]:
    return re.findall(r"\w+", (text or "").lower())


def _score_document(query: str, document: Dict[str, Any]) -> float:
    text = str(document.get("text", ""))
    query_terms = _tokenize(query)
    doc_terms = _tokenize(text)
    if not query_terms:
        return 0.0
    overlap = sum(1 for term in query_terms if term in doc_terms)
    return float(overlap) / float(len(query_terms))


def rank_documents(documents: List[Dict[str, Any]], query: str, top_k: int = 3) -> List[Dict[str, Any]]:
    scored = []
    for document in documents:
        score = _score_document(query, document)
        scored.append({**document, "_score": score})
    ranked = sorted(scored, key=lambda item: item["_score"], reverse=True)
    result = []
    for item in ranked[:max(1, top_k)]:
        item.copy()
        item.pop("_score", None)
        result.append(item)
    return result
