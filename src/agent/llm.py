"""Chat backends for the document agent.

``mock`` is deterministic and is the default. ``openai`` calls an
OpenAI-compatible chat endpoint, which can be a hosted model or a local server
such as Ollama or llama.cpp. Both backends use the same action schema.
"""

from __future__ import annotations

import json
import os
import re
from typing import Any, Protocol

from src.agent.tools import extract_expression, extract_lookup_term
from src.generator import build_extractive_answer


ALLOWED_ACTIONS = {"retrieve", "calculate", "lookup", "search", "draft", "finalize"}

_EXTRACTIVE_PREFIX = "LLM unavailable; answer extracted from retrieved context: "

_JSON_OBJECT = re.compile(r"\{.*\}", re.DOTALL)

CORPUS_HINTS = (
    "according",
    "corpus",
    "document",
    "documents",
    "framework",
    "local",
    "note",
    "notes",
    "paper",
    "papers",
    "nist",
    "risk",
)

DECIDE_INSTRUCTIONS = """You are the routing step of a document question-answering agent.
Choose exactly one next action and return a JSON object with keys action, arguments, and thought.
Allowed actions:
- retrieve: arguments {"query": string}. Search the local document corpus.
- calculate: arguments {"expression": string}. Evaluate arithmetic.
- lookup: arguments {"term": string}. Read the local glossary.
- search: arguments {"query": string}. Search arXiv. The tool may be disabled.
- draft: arguments {}. Write the answer from evidence already collected.
- finalize: arguments {}. Stop because more tools will not help.
Do not answer the user in this step. Return JSON only.
"""

DRAFT_INSTRUCTIONS = """You are drafting an answer from tool evidence.
Use only the evidence below. If it does not support an answer, say so.
Include the relevant numbers, definitions, and document wording.
Do not add facts that are not in the evidence.
"""


class ChatModel(Protocol):
    """Planning, routing, and drafting interface used by the graph."""

    def plan(self, question: str) -> str:
        """Return a short plan for the question."""

    def decide(self, question: str, tool_calls: list[dict[str, Any]], *, web_search_enabled: bool) -> dict[str, Any]:
        """Return an action dict with ``action``, ``arguments``, and ``thought``."""

    def draft(self, question: str, evidence: list[dict[str, Any]], tool_calls: list[dict[str, Any]]) -> str:
        """Return a draft answer grounded in tool evidence."""


def looks_like_math(question: str) -> bool:
    """True when the question contains an explicit arithmetic expression."""
    try:
        extract_expression(question)
    except ValueError:
        return False
    return True


def looks_like_lookup(question: str) -> bool:
    """True when the question asks for the local glossary."""
    text = (question or "").lower()
    return any(phrase in text for phrase in ("glossary", "look up", "lookup", "define the term"))


def wants_web(question: str) -> bool:
    """True when the question asks for arXiv or web search."""
    text = (question or "").lower()
    return any(phrase in text for phrase in ("arxiv", "search the web", "search online", "web search"))


def needs_documents(question: str) -> bool:
    """True when the question should retrieve local documents before drafting."""
    text = (question or "").lower()
    if wants_web(text) and not any(hint in text for hint in ("local", "corpus", "notes", "according")):
        return False
    if any(hint in text for hint in CORPUS_HINTS):
        return True
    if looks_like_math(text) or looks_like_lookup(text) or wants_web(text):
        return False
    return True


def choose_action(
    question: str,
    tool_calls: list[dict[str, Any]],
    *,
    web_search_enabled: bool,
) -> dict[str, Any]:
    """Deterministic routing policy shared by the mock model and JSON fallback."""
    called = {call.get("name") for call in tool_calls}
    if looks_like_math(question) and "calculate" not in called:
        try:
            expression = extract_expression(question)
        except ValueError:
            expression = question
        return {
            "action": "calculate",
            "arguments": {"expression": expression},
            "thought": "The question includes an arithmetic expression.",
        }
    if looks_like_lookup(question) and "lookup" not in called:
        return {
            "action": "lookup",
            "arguments": {"term": extract_lookup_term(question)},
            "thought": "The question asks for a glossary definition.",
        }
    if wants_web(question) and "search" not in called:
        thought = "The question asks for an external literature search."
        if not web_search_enabled:
            thought += " Search is currently disabled; the tool will report that."
        return {
            "action": "search",
            "arguments": {"query": question},
            "thought": thought,
        }
    if needs_documents(question) and "retrieve" not in called:
        return {
            "action": "retrieve",
            "arguments": {"query": question},
            "thought": "The question should be answered from local documents.",
        }
    return {
        "action": "draft",
        "arguments": {},
        "thought": "Enough tool evidence is available to draft an answer.",
    }


def draft_from_evidence(
    question: str,
    evidence: list[dict[str, Any]],
    tool_calls: list[dict[str, Any]],
) -> str:
    """Build a draft that quotes tool observations and retrieved sentences."""
    lines: list[str] = []
    for call in tool_calls:
        observation = call.get("observation") or {}
        name = call.get("name")
        if name == "calculate":
            lines.append(str(observation.get("text") or ""))
        elif name == "lookup":
            lines.append(str(observation.get("text") or ""))
        elif name == "search":
            if observation.get("message"):
                lines.append(str(observation["message"]))
            for item in observation.get("results") or []:
                title = item.get("title") or "Untitled"
                identifier = item.get("id") or "unknown id"
                lines.append(f"{title} ({identifier})")
    passages = [str(item.get("text") or "") for item in evidence if item.get("tool") == "retrieve" and item.get("text")]
    if passages:
        extracted = build_extractive_answer(question, "\n".join(passages))
        if extracted.startswith(_EXTRACTIVE_PREFIX):
            extracted = extracted[len(_EXTRACTIVE_PREFIX) :]
        lines.append(extracted)
    cleaned = [line.strip() for line in lines if line and line.strip()]
    if not cleaned:
        return "The available evidence does not contain enough information to answer this question."
    return "\n".join(cleaned)


class MockChatModel:
    """Deterministic stand-in that emits the agent action schema without a model call."""

    def __init__(self, web_search_enabled: bool = False):
        self.web_search_enabled = web_search_enabled

    def plan(self, question: str) -> str:
        first = choose_action(question, [], web_search_enabled=self.web_search_enabled)
        steps = [first["action"]]
        if first["action"] != "draft":
            steps.append("draft")
        steps.append("reflect")
        return "Plan: " + ", then ".join(steps) + ". Keep the answer tied to tool evidence."

    def decide(self, question: str, tool_calls: list[dict[str, Any]], *, web_search_enabled: bool) -> dict[str, Any]:
        return choose_action(question, tool_calls, web_search_enabled=web_search_enabled)

    def draft(self, question: str, evidence: list[dict[str, Any]], tool_calls: list[dict[str, Any]]) -> str:
        return draft_from_evidence(question, evidence, tool_calls)


def parse_action(text: str) -> dict[str, Any]:
    """Parse one action object from a model response."""
    match = _JSON_OBJECT.search(text or "")
    if not match:
        raise ValueError("model response did not contain a JSON object")
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError as exc:
        raise ValueError("model response contained invalid JSON") from exc
    if not isinstance(data, dict):
        raise ValueError("model response JSON was not an object")
    action = str(data.get("action") or "").strip().lower()
    if action not in ALLOWED_ACTIONS:
        raise ValueError(f"model response used an unknown action: {action or '<empty>'}")
    arguments = data.get("arguments") or {}
    if not isinstance(arguments, dict):
        raise ValueError("model response arguments were not an object")
    return {"action": action, "arguments": arguments, "thought": str(data.get("thought") or "")}


class OpenAICompatibleChat:
    """Action model backed by the project's OpenAI-compatible ``LLMClient``."""

    def __init__(self, client: Any | None = None, web_search_enabled: bool = False):
        self.web_search_enabled = web_search_enabled
        self._client = client

    @property
    def client(self) -> Any:
        if self._client is None:
            from src.generator import LLMClient

            self._client = LLMClient()
        return self._client

    def _complete(self, prompt: str) -> str:
        return self.client.generate(prompt, temperature=0)

    def plan(self, question: str) -> str:
        prompt = (
            "Write a one-sentence plan for answering the question. "
            "Name which of retrieve, calculate, lookup, or search you expect to use.\n\n"
            f"Question: {question}"
        )
        return self._complete(prompt).strip()

    def decide(self, question: str, tool_calls: list[dict[str, Any]], *, web_search_enabled: bool) -> dict[str, Any]:
        called = [
            {"name": call.get("name"), "arguments": call.get("arguments"), "observation": call.get("observation")}
            for call in tool_calls
        ]
        prompt = (
            f"{DECIDE_INSTRUCTIONS}\n"
            f"Web search enabled: {web_search_enabled}\n"
            f"Question: {question}\n"
            f"Tool calls so far: {json.dumps(called, default=str)}\n"
        )
        raw = self._complete(prompt)
        try:
            return parse_action(raw)
        except ValueError:
            decision = choose_action(question, tool_calls, web_search_enabled=web_search_enabled)
            decision["thought"] = (
                "Model output was not valid JSON; used the deterministic action policy. " + decision["thought"]
            )
            return decision

    def draft(self, question: str, evidence: list[dict[str, Any]], tool_calls: list[dict[str, Any]]) -> str:
        prompt = (
            f"{DRAFT_INSTRUCTIONS}\n"
            f"Question: {question}\n\n"
            f"Evidence:\n{json.dumps(evidence, default=str)}\n"
        )
        text = self._complete(prompt).strip()
        if not text:
            return draft_from_evidence(question, evidence, tool_calls)
        return text


def get_chat_model(backend: str | None = None, *, web_search_enabled: bool = False) -> ChatModel:
    """Return the configured chat backend.

    ``AGENT_LLM_BACKEND`` accepts ``mock``, ``openai``, and ``auto``.
    ``auto`` uses the OpenAI-compatible client only when an API key is set or
    ``LLM_BASE_URL`` points at localhost. Otherwise it uses the mock.
    """
    selected = (backend if backend is not None else os.getenv("AGENT_LLM_BACKEND", "auto")).strip().lower()
    if selected == "mock":
        return MockChatModel(web_search_enabled=web_search_enabled)
    if selected in {"openai", "hosted", "local"}:
        return OpenAICompatibleChat(web_search_enabled=web_search_enabled)
    if selected not in {"", "auto"}:
        raise ValueError("AGENT_LLM_BACKEND must be mock, openai, or auto")

    api_key = os.getenv("LLM_API_KEY", "").strip()
    base_url = os.getenv("LLM_BASE_URL", "")
    if api_key or "localhost" in base_url or "127.0.0.1" in base_url:
        return OpenAICompatibleChat(web_search_enabled=web_search_enabled)
    return MockChatModel(web_search_enabled=web_search_enabled)
