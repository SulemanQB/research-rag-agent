"""Shared state for the document agent graph."""

from __future__ import annotations

import operator
from typing import Annotated, Any, TypedDict


class AgentState(TypedDict):
    """Fields passed between graph nodes.

    ``tool_calls``, ``evidence``, and ``trace`` append across nodes.
    Every other field is replaced by the node that writes it.
    """

    question: str
    plan: str
    next_action: str
    action_args: dict[str, Any]
    step_count: int
    max_steps: int
    max_reflections: int
    reflect_count: int
    grounded: bool
    groundedness_score: float
    reflection_notes: str
    reflection_route: str
    retrieval_query: str
    resume_after_tool: str
    draft: str
    answer: str
    top_k: int
    last_thought: str
    tool_calls: Annotated[list[dict[str, Any]], operator.add]
    evidence: Annotated[list[dict[str, Any]], operator.add]
    trace: Annotated[list[str], operator.add]
    citations: list[str]
