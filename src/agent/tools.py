"""Tools the document agent can call.

The retriever adapts either the lexical ranker or the existing Chroma store.
The calculator and glossary are local. arXiv search is off unless
``AGENT_WEB_SEARCH`` is enabled.
"""

from __future__ import annotations

import ast
import operator
import os
import re
from typing import Any, Mapping, Protocol
from urllib.parse import urlencode
from urllib.request import urlopen
from xml.etree import ElementTree

from src.agent.glossary import GLOSSARY
from src.retriever import document_overlap


class DocumentRetriever(Protocol):
    """Return passage dicts with ``text``, ``source``, and ``page``."""

    def retrieve(self, query: str, top_k: int = 3) -> list[dict[str, Any]]:
        """Rank passages for a query."""


class LexicalRetriever:
    """In-memory retriever backed by the project's token-overlap score.

    Passages with no query overlap are dropped. Only passages tied with the
    best overlap are kept, and at most ``top_k`` of those are returned.
    """

    def __init__(self, documents: list[dict[str, Any]]):
        self.documents = documents

    def retrieve(self, query: str, top_k: int = 3) -> list[dict[str, Any]]:
        scored = []
        for document in self.documents:
            scored.append((document_overlap(query, document), document))
        best = max((score for score, _document in scored), default=0.0)
        if best <= 0:
            return []
        ranked = sorted(
            ((score, document) for score, document in scored if score >= best - 1e-9),
            key=lambda item: item[0],
            reverse=True,
        )
        results = []
        for _score, document in ranked[: max(1, top_k)]:
            metadata = document.get("metadata") or {}
            results.append(
                {
                    "text": document.get("text", ""),
                    "source": metadata.get("source") or document.get("source") or "unknown",
                    "page": metadata.get("page") or document.get("page") or 1,
                }
            )
        return results


class VectorStoreRetriever:
    """Adapter over ``LocalVectorStore`` for the agent retrieve tool."""

    def __init__(self, store: Any | None = None):
        self.store = store

    def retrieve(self, query: str, top_k: int = 3) -> list[dict[str, Any]]:
        store = self.store
        if store is None:
            from src.vector_store import LocalVectorStore

            store = LocalVectorStore()
            self.store = store
        raw = store.query(query, top_k=top_k)
        documents = (raw.get("documents") or [[]])[0] or []
        metadatas = (raw.get("metadatas") or [[]])[0] or []
        results = []
        for text, meta in zip(documents, metadatas):
            meta = meta or {}
            results.append(
                {
                    "text": text or "",
                    "source": meta.get("source") or meta.get("filename") or "unknown",
                    "page": meta.get("page") or 1,
                }
            )
        return results


_BIN_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_UNARY_OPS = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}
_EXPRESSION = re.compile(
    r"\d+(?:\.\d+)?(?:(?:\s*[+*/%]\s*|\s+-\s+)\d+(?:\.\d+)?)+"
)
_MAX_ABS_EXPONENT = 8


def extract_expression(text: str) -> str:
    """Pull a single arithmetic expression out of a question."""
    cleaned = (text or "").replace("×", "*").replace("÷", "/")
    match = _EXPRESSION.search(cleaned)
    if not match:
        raise ValueError("no arithmetic expression found")
    return match.group(0).strip()


def _eval_node(node: ast.AST) -> float:
    if isinstance(node, ast.Expression):
        return _eval_node(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
        return float(node.value)
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPS:
        return float(_UNARY_OPS[type(node.op)](_eval_node(node.operand)))
    if isinstance(node, ast.BinOp) and type(node.op) in _BIN_OPS:
        left = _eval_node(node.left)
        right = _eval_node(node.right)
        if isinstance(node.op, ast.Pow) and abs(right) > _MAX_ABS_EXPONENT:
            raise ValueError("exponent is too large")
        return float(_BIN_OPS[type(node.op)](left, right))
    raise ValueError("unsupported expression")


def evaluate_arithmetic(expression: str) -> float:
    """Evaluate a numeric expression with arithmetic operators only."""
    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError as exc:
        raise ValueError("unsupported expression") from exc
    return _eval_node(tree)


def format_number(value: float) -> str:
    """Render a calculation without binary float noise."""
    rounded = round(float(value), 10)
    if rounded == int(rounded):
        return str(int(rounded))
    return f"{rounded:.10f}".rstrip("0").rstrip(".")


def calculate(expression: str) -> dict[str, Any]:
    """Evaluate ``expression`` and return a tool observation."""
    try:
        value = evaluate_arithmetic(expression)
    except (ValueError, ZeroDivisionError, OverflowError) as exc:
        message = "division by zero" if isinstance(exc, ZeroDivisionError) else str(exc)
        return {
            "expression": expression,
            "error": message,
            "text": f"Calculation error: {message}",
        }
    rendered = format_number(value)
    return {
        "expression": expression,
        "result": rendered,
        "text": f"{expression} = {rendered}",
    }


def _term_in_text(term: str, text: str) -> bool:
    return re.search(rf"\b{re.escape(term)}\b", text) is not None


def lookup_term(term: str, glossary: Mapping[str, Mapping[str, str]] | None = None) -> dict[str, Any]:
    """Return the glossary entry whose name appears in ``term``."""
    entries = dict(glossary or GLOSSARY)
    query = (term or "").strip().lower()
    known = ", ".join(entry["term"] for entry in entries.values())
    if not query:
        return {
            "found": False,
            "term": term,
            "text": f"No glossary entry matched an empty term. Known terms: {known}.",
            "source": "glossary",
        }

    matches = [key for key in entries if _term_in_text(key, query) or key == query]
    if not matches:
        matches = [key for key in entries if _term_in_text(query, key)]
    if not matches:
        return {
            "found": False,
            "term": term,
            "text": f"No glossary entry matched {term.strip() or term!r}. Known terms: {known}.",
            "source": "glossary",
        }

    key = sorted(matches, key=len, reverse=True)[0]
    entry = entries[key]
    text = entry["definition"]
    return {"found": True, "term": entry["term"], "text": text, "source": "glossary"}


def extract_lookup_term(question: str, glossary: Mapping[str, Mapping[str, str]] | None = None) -> str:
    """Choose a glossary key mentioned in the question, or the raw question."""
    entries = glossary or GLOSSARY
    text = question or ""
    matches = [key for key in entries if _term_in_text(key, text.lower())]
    if matches:
        return sorted(matches, key=len, reverse=True)[0]
    return text.strip()


def env_flag(name: str, default: bool = False) -> bool:
    """Read a boolean environment flag."""
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


_ARXIV_NS = {"atom": "http://www.w3.org/2005/Atom"}


def search_arxiv(query: str, *, enabled: bool, max_results: int = 3, timeout: float = 20) -> dict[str, Any]:
    """Search the public arXiv API, or explain that the tool is switched off."""
    cleaned = (query or "").strip()
    if not enabled:
        return {
            "enabled": False,
            "query": cleaned,
            "results": [],
            "message": "arXiv search is disabled. Set AGENT_WEB_SEARCH=1 to enable it.",
        }
    if not cleaned:
        return {
            "enabled": True,
            "query": cleaned,
            "results": [],
            "message": "arXiv search needs a non-empty query.",
        }
    bounded = min(max(1, int(max_results)), 5)
    url = "https://export.arxiv.org/api/query?" + urlencode(
        {"search_query": f"all:{cleaned}", "start": 0, "max_results": bounded}
    )
    try:
        with urlopen(url, timeout=timeout) as response:
            payload = response.read()
    except Exception as exc:
        return {
            "enabled": True,
            "query": cleaned,
            "results": [],
            "error": str(exc),
            "message": f"arXiv search failed: {exc}",
        }

    try:
        root = ElementTree.fromstring(payload)
    except ElementTree.ParseError as exc:
        return {
            "enabled": True,
            "query": cleaned,
            "results": [],
            "error": str(exc),
            "message": "arXiv search returned a response that was not valid Atom XML.",
        }

    results = []
    for entry in root.findall("atom:entry", _ARXIV_NS):
        title = " ".join((entry.findtext("atom:title", default="", namespaces=_ARXIV_NS) or "").split())
        summary = " ".join((entry.findtext("atom:summary", default="", namespaces=_ARXIV_NS) or "").split())
        raw_id = (entry.findtext("atom:id", default="", namespaces=_ARXIV_NS) or "").strip()
        results.append({"id": raw_id, "title": title, "summary": summary[:400]})
    if not results:
        message = f"arXiv search returned no results for {cleaned}."
    else:
        message = f"arXiv search returned {len(results)} result(s) for {cleaned}."
    return {"enabled": True, "query": cleaned, "results": results, "message": message}
