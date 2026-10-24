"""State graph for planning, tool use, reflection, and cited answers.

Control flow:

    plan -> decide -> retrieve | calculate | lookup | search | draft | finalize
    retrieve/calculate/lookup/search -> decide
    draft -> reflect
    reflect -> retrieve (when the draft is not grounded and budget remains)
    reflect -> finalize
    a retrieve requested by reflect goes back to draft

``max_steps`` bounds how many times ``decide`` can select a tool. Reflection
retries also consume a step. The graph then finalizes with the evidence it has.
"""

from __future__ import annotations

import os
from typing import Any

from langgraph.graph import END, START, StateGraph

from src.agent.grounding import evidence_text, refine_retrieval_query, score_groundedness
from src.agent.llm import ChatModel, get_chat_model
from src.agent.state import AgentState
from src.agent.tools import (
    DocumentRetriever,
    VectorStoreRetriever,
    calculate,
    env_flag,
    lookup_term,
    search_arxiv,
)

DEFAULT_GROUNDED_THRESHOLD = 0.5
DEFAULT_MAX_STEPS = 6
DEFAULT_MAX_REFLECTIONS = 1


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or not str(raw).strip():
        return default
    return int(raw)


def initial_state(
    question: str,
    *,
    max_steps: int,
    max_reflections: int,
    top_k: int,
) -> AgentState:
    return {
        "question": question,
        "plan": "",
        "next_action": "",
        "action_args": {},
        "step_count": 0,
        "max_steps": max_steps,
        "max_reflections": max_reflections,
        "reflect_count": 0,
        "grounded": False,
        "groundedness_score": 0.0,
        "reflection_notes": "",
        "reflection_route": "",
        "retrieval_query": question,
        "resume_after_tool": "decide",
        "draft": "",
        "answer": "",
        "top_k": top_k,
        "last_thought": "",
        "tool_calls": [],
        "evidence": [],
        "trace": [],
        "citations": [],
    }


def _format_citations(evidence: list[dict[str, Any]]) -> list[str]:
    citations: list[str] = []
    for item in evidence:
        source = item.get("source")
        if not source:
            continue
        if item.get("tool") == "search" and not item.get("cite"):
            continue
        page = item.get("page")
        label = f"{source} — page {page}" if page else str(source)
        if label not in citations:
            citations.append(label)
    return citations


def _append_citations(answer: str, citations: list[str]) -> str:
    if not citations or "Citations:" in answer:
        return answer
    lines = "\n".join(f"- {citation}" for citation in citations)
    return answer.rstrip() + "\n\nCitations:\n" + lines


class DocumentAgent:
    """LangGraph agent closed over a chat model, retriever, and tool settings."""

    def __init__(
        self,
        chat_model: ChatModel,
        retriever: DocumentRetriever,
        *,
        web_search_enabled: bool = False,
        grounded_threshold: float = DEFAULT_GROUNDED_THRESHOLD,
        search_fn: Any = search_arxiv,
        glossary: dict | None = None,
    ):
        self.chat_model = chat_model
        self.retriever = retriever
        self.web_search_enabled = web_search_enabled
        self.grounded_threshold = grounded_threshold
        self.search_fn = search_fn
        self.glossary = glossary

    def plan_node(self, state: AgentState) -> dict[str, Any]:
        plan = self.chat_model.plan(state["question"])
        return {"plan": plan, "trace": ["plan"], "retrieval_query": state["question"]}

    def decide_node(self, state: AgentState) -> dict[str, Any]:
        if state["step_count"] >= state["max_steps"]:
            return {
                "next_action": "finalize",
                "action_args": {},
                "resume_after_tool": "decide",
                "last_thought": "Step limit reached, so the agent will answer with the evidence collected so far.",
                "trace": ["decide"],
            }
        decision = self.chat_model.decide(
            state["question"],
            list(state.get("tool_calls") or []),
            web_search_enabled=self.web_search_enabled,
        )
        action = decision.get("action") or "finalize"
        if action not in {"retrieve", "calculate", "lookup", "search", "draft", "finalize"}:
            action = "finalize"
        return {
            "step_count": state["step_count"] + 1,
            "next_action": action,
            "action_args": decision.get("arguments") or {},
            "resume_after_tool": "decide",
            "last_thought": decision.get("thought") or "",
            "trace": ["decide"],
        }

    def retrieve_node(self, state: AgentState) -> dict[str, Any]:
        if state.get("resume_after_tool") == "draft" and state.get("retrieval_query"):
            query = state["retrieval_query"]
        else:
            query = (state.get("action_args") or {}).get("query") or state.get("retrieval_query") or state["question"]
        top_k = state.get("top_k") or 3
        documents = self.retriever.retrieve(str(query), top_k=top_k)
        evidence = [
            {
                "tool": "retrieve",
                "text": document.get("text", ""),
                "source": document.get("source") or "unknown",
                "page": document.get("page") or 1,
            }
            for document in documents
        ]
        return {
            "evidence": evidence,
            "retrieval_query": str(query),
            "tool_calls": [
                {
                    "name": "retrieve",
                    "arguments": {"query": str(query), "top_k": top_k},
                    "observation": {"query": str(query), "passages": len(evidence)},
                }
            ],
            "trace": ["retrieve"],
        }

    def calculate_node(self, state: AgentState) -> dict[str, Any]:
        expression = str((state.get("action_args") or {}).get("expression") or "")
        observation = calculate(expression)
        return {
            "evidence": [
                {
                    "tool": "calculate",
                    "text": observation["text"],
                    "source": "calculator",
                    "page": None,
                }
            ],
            "tool_calls": [{"name": "calculate", "arguments": {"expression": expression}, "observation": observation}],
            "trace": ["calculate"],
        }

    def lookup_node(self, state: AgentState) -> dict[str, Any]:
        term = str((state.get("action_args") or {}).get("term") or state["question"])
        observation = lookup_term(term, self.glossary)
        return {
            "evidence": [
                {
                    "tool": "lookup",
                    "text": observation["text"],
                    "source": observation.get("source") or "glossary",
                    "page": None,
                }
            ],
            "tool_calls": [{"name": "lookup", "arguments": {"term": term}, "observation": observation}],
            "trace": ["lookup"],
        }

    def search_node(self, state: AgentState) -> dict[str, Any]:
        query = str((state.get("action_args") or {}).get("query") or state["question"])
        observation = self.search_fn(query, enabled=self.web_search_enabled)
        evidence: list[dict[str, Any]] = []
        if observation.get("message"):
            evidence.append({"tool": "search", "text": observation["message"], "source": "", "page": None})
        for item in observation.get("results") or []:
            identifier = item.get("id") or "arxiv"
            text = f"{item.get('title', '')}. {item.get('summary', '')}".strip()
            evidence.append(
                {
                    "tool": "search",
                    "text": text,
                    "source": identifier,
                    "page": None,
                    "cite": True,
                }
            )
        return {
            "evidence": evidence,
            "tool_calls": [{"name": "search", "arguments": {"query": query}, "observation": observation}],
            "trace": ["search"],
        }

    def draft_node(self, state: AgentState) -> dict[str, Any]:
        draft = self.chat_model.draft(
            state["question"],
            list(state.get("evidence") or []),
            list(state.get("tool_calls") or []),
        )
        return {"draft": draft, "trace": ["draft"]}

    def reflect_node(self, state: AgentState) -> dict[str, Any]:
        score = score_groundedness(state.get("draft") or "", evidence_text(list(state.get("evidence") or [])))
        grounded = score >= self.grounded_threshold
        notes = f"groundedness={score:.2f}; threshold={self.grounded_threshold:.2f}"
        can_retry = (
            not grounded
            and state["reflect_count"] < state["max_reflections"]
            and state["step_count"] < state["max_steps"]
        )
        if can_retry:
            refined = refine_retrieval_query(state["question"])
            return {
                "grounded": False,
                "groundedness_score": score,
                "reflection_notes": notes + "; re-retrieving with a shorter query",
                "reflect_count": state["reflect_count"] + 1,
                "step_count": state["step_count"] + 1,
                "retrieval_query": refined,
                "action_args": {"query": refined},
                "resume_after_tool": "draft",
                "reflection_route": "retrieve",
                "trace": ["reflect"],
            }
        ending = "; accepting the draft" if grounded else "; stopping without further retrieval"
        return {
            "grounded": grounded,
            "groundedness_score": score,
            "reflection_notes": notes + ending,
            "reflection_route": "finalize",
            "trace": ["reflect"],
        }

    def finalize_node(self, state: AgentState) -> dict[str, Any]:
        draft = (state.get("draft") or "").strip()
        if not draft:
            draft = self.chat_model.draft(
                state["question"],
                list(state.get("evidence") or []),
                list(state.get("tool_calls") or []),
            )
        citations = _format_citations(list(state.get("evidence") or []))
        return {
            "draft": draft,
            "answer": _append_citations(draft, citations),
            "citations": citations,
            "trace": ["finalize"],
        }


def route_decision(state: AgentState) -> str:
    """Map the decision node's action to the next graph node."""
    action = state.get("next_action") or "finalize"
    if action in {"retrieve", "calculate", "lookup", "search", "draft", "finalize"}:
        return action
    return "finalize"


def route_after_retrieve(state: AgentState) -> str:
    """Return to drafting when retrieval was requested by reflection."""
    if state.get("resume_after_tool") == "draft":
        return "draft"
    return "decide"


def route_reflection(state: AgentState) -> str:
    """Re-retrieve or finish, depending on the reflection node."""
    if state.get("reflection_route") == "retrieve":
        return "retrieve"
    return "finalize"


def build_agent_graph(
    chat_model: ChatModel,
    retriever: DocumentRetriever,
    *,
    web_search_enabled: bool = False,
    grounded_threshold: float = DEFAULT_GROUNDED_THRESHOLD,
    search_fn: Any = search_arxiv,
    glossary: dict | None = None,
):
    """Compile the document agent graph."""
    agent = DocumentAgent(
        chat_model,
        retriever,
        web_search_enabled=web_search_enabled,
        grounded_threshold=grounded_threshold,
        search_fn=search_fn,
        glossary=glossary,
    )
    builder = StateGraph(AgentState)
    builder.add_node("plan", agent.plan_node)
    builder.add_node("decide", agent.decide_node)
    builder.add_node("retrieve", agent.retrieve_node)
    builder.add_node("calculate", agent.calculate_node)
    builder.add_node("lookup", agent.lookup_node)
    builder.add_node("search", agent.search_node)
    builder.add_node("draft", agent.draft_node)
    builder.add_node("reflect", agent.reflect_node)
    builder.add_node("finalize", agent.finalize_node)

    builder.add_edge(START, "plan")
    builder.add_edge("plan", "decide")
    builder.add_conditional_edges(
        "decide",
        route_decision,
        {
            "retrieve": "retrieve",
            "calculate": "calculate",
            "lookup": "lookup",
            "search": "search",
            "draft": "draft",
            "finalize": "finalize",
        },
    )
    builder.add_conditional_edges(
        "retrieve",
        route_after_retrieve,
        {"decide": "decide", "draft": "draft"},
    )
    builder.add_edge("calculate", "decide")
    builder.add_edge("lookup", "decide")
    builder.add_edge("search", "decide")
    builder.add_edge("draft", "reflect")
    builder.add_conditional_edges(
        "reflect",
        route_reflection,
        {"retrieve": "retrieve", "finalize": "finalize"},
    )
    builder.add_edge("finalize", END)
    return builder.compile()


def run_agent(
    question: str,
    *,
    chat_model: ChatModel | None = None,
    retriever: DocumentRetriever | None = None,
    max_steps: int | None = None,
    max_reflections: int | None = None,
    top_k: int = 3,
    web_search_enabled: bool | None = None,
    grounded_threshold: float = DEFAULT_GROUNDED_THRESHOLD,
    backend: str | None = None,
) -> AgentState:
    """Run the agent on one question and return the final state."""
    if not str(question or "").strip():
        raise ValueError("question must not be empty")
    if top_k <= 0:
        raise ValueError("top_k must be positive")

    resolved_steps = _env_int("AGENT_MAX_STEPS", DEFAULT_MAX_STEPS) if max_steps is None else max_steps
    resolved_reflections = (
        _env_int("AGENT_MAX_REFLECTIONS", DEFAULT_MAX_REFLECTIONS)
        if max_reflections is None
        else max_reflections
    )
    if resolved_steps <= 0:
        raise ValueError("max_steps must be positive")
    if resolved_reflections < 0:
        raise ValueError("max_reflections must be non-negative")

    if web_search_enabled is None:
        web_search_enabled = env_flag("AGENT_WEB_SEARCH", default=False)
    model = chat_model or get_chat_model(backend, web_search_enabled=web_search_enabled)
    store = retriever or VectorStoreRetriever()
    graph = build_agent_graph(
        model,
        store,
        web_search_enabled=web_search_enabled,
        grounded_threshold=grounded_threshold,
    )
    recursion_limit = max(resolved_steps * 6, 20)
    return graph.invoke(
        initial_state(
            question,
            max_steps=resolved_steps,
            max_reflections=resolved_reflections,
            top_k=top_k,
        ),
        config={"recursion_limit": recursion_limit},
    )
