"""Compare plain extractive RAG with the LangGraph agent on the fixture set.

The retriever is the in-memory lexical ranker, and the agent uses the mock
chat backend. No API key and no Chroma index are required. The script prints
the measured summary; it does not read or write a stored score.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from src.agent.graph import build_agent_graph, initial_state
from src.agent.grounding import score_groundedness
from src.agent.llm import MockChatModel
from src.agent.tools import LexicalRetriever
from src.generator import build_extractive_answer


def load_fixture_documents(folder: Path | None = None) -> list[dict[str, Any]]:
    """Load the committed markdown fixtures as retriever documents."""
    root = folder or Path(__file__).resolve().parent / "fixtures"
    documents = []
    for path in sorted(root.glob("*")):
        if path.suffix.lower() not in {".md", ".txt"} or not path.is_file():
            continue
        documents.append(
            {
                "id": path.name,
                "text": path.read_text(encoding="utf-8").strip(),
                "metadata": {"source": path.name, "page": 1},
            }
        )
    return documents


def load_agent_questions(path: Path | None = None) -> list[dict[str, Any]]:
    """Load the comparison questions."""
    target = path or Path(__file__).resolve().parent / "agent_questions.json"
    payload = json.loads(target.read_text(encoding="utf-8"))
    return list(payload["questions"])


def _accurate(answer: str, facts: list[str]) -> bool:
    haystack = (answer or "").lower()
    return all(fact.lower() in haystack for fact in facts)


def _mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)


def answer_plain_rag(question: str, retriever: LexicalRetriever, top_k: int = 3) -> dict[str, Any]:
    """One retrieval plus the project's extractive answer. No tool routing."""
    documents = retriever.retrieve(question, top_k=top_k)
    evidence = "\n".join(document.get("text", "") for document in documents)
    if not evidence.strip():
        answer = "The corpus does not contain enough information to answer this question."
    else:
        answer = build_extractive_answer(question, evidence)
    return {
        "answer": answer,
        "tool_calls": 1,
        "groundedness": score_groundedness(answer, evidence),
        "reflections": 0,
    }


def run_comparison(
    questions: list[dict[str, Any]] | None = None,
    documents: list[dict[str, Any]] | None = None,
    *,
    top_k: int = 3,
    max_steps: int = 6,
) -> dict[str, Any]:
    """Time one pass of plain RAG and one pass of the agent on the same questions."""
    question_rows = questions if questions is not None else load_agent_questions()
    corpus = documents if documents is not None else load_fixture_documents()
    retriever = LexicalRetriever(corpus)
    graph = build_agent_graph(MockChatModel(web_search_enabled=False), retriever, web_search_enabled=False)
    rows = []

    for item in question_rows:
        question = item["question"]
        facts = list(item.get("expected_facts") or [])

        plain_started = time.perf_counter()
        plain = answer_plain_rag(question, retriever, top_k=top_k)
        plain_ms = (time.perf_counter() - plain_started) * 1000

        agent_started = time.perf_counter()
        agent_state = graph.invoke(
            initial_state(question, max_steps=max_steps, max_reflections=1, top_k=top_k),
            config={"recursion_limit": max(max_steps * 6, 20)},
        )
        agent_ms = (time.perf_counter() - agent_started) * 1000
        agent_answer = agent_state.get("answer") or ""
        evidence = "\n".join(piece.get("text") or "" for piece in agent_state.get("evidence") or [])
        rows.append(
            {
                "id": item.get("id") or question,
                "question": question,
                "plain_correct": _accurate(plain["answer"], facts),
                "agent_correct": _accurate(agent_answer, facts),
                "plain_groundedness": plain["groundedness"],
                "agent_groundedness": score_groundedness(agent_answer, evidence),
                "plain_tool_calls": plain["tool_calls"],
                "agent_tool_calls": len(agent_state.get("tool_calls") or []),
                "agent_reflections": agent_state.get("reflect_count") or 0,
                "plain_latency_ms": plain_ms,
                "agent_latency_ms": agent_ms,
                "agent_tools": [call.get("name") for call in agent_state.get("tool_calls") or []],
            }
        )

    def summarize(prefix: str) -> dict[str, Any]:
        correct = sum(1 for row in rows if row[f"{prefix}_correct"])
        return {
            "correct": correct,
            "questions": len(rows),
            "accuracy": (correct / len(rows)) if rows else 0.0,
            "mean_groundedness": _mean([row[f"{prefix}_groundedness"] for row in rows]),
            "mean_tool_calls": _mean([float(row[f"{prefix}_tool_calls"]) for row in rows]),
            "mean_latency_ms": _mean([row[f"{prefix}_latency_ms"] for row in rows]),
        }

    return {
        "backend": "mock",
        "retriever": "lexical",
        "questions": len(rows),
        "plain_rag": summarize("plain"),
        "agent": {
            **summarize("agent"),
            "reflection_retries": sum(row["agent_reflections"] for row in rows),
        },
        "rows": rows,
    }


def _format_report(report: dict[str, Any]) -> str:
    plain = report["plain_rag"]
    agent = report["agent"]
    lines = [
        "Plain RAG vs LangGraph agent",
        "----------------------------",
        f"Questions: {report['questions']}",
        f"Backend: {report['backend']}",
        f"Retriever: {report['retriever']}",
        "",
        f"{'system':<16} {'accuracy':>10} {'groundedness':>14} {'tool calls':>12} {'latency ms':>12}",
        (
            f"{'plain RAG':<16} {plain['accuracy']:>10.3f} {plain['mean_groundedness']:>14.3f} "
            f"{plain['mean_tool_calls']:>12.3f} {plain['mean_latency_ms']:>12.3f}"
        ),
        (
            f"{'LangGraph agent':<16} {agent['accuracy']:>10.3f} {agent['mean_groundedness']:>14.3f} "
            f"{agent['mean_tool_calls']:>12.3f} {agent['mean_latency_ms']:>12.3f}"
        ),
        f"Agent reflection retries: {agent['reflection_retries']}",
        "",
        "Per question",
    ]
    for row in report["rows"]:
        lines.append(
            f"- {row['id']}: plain_correct={row['plain_correct']} agent_correct={row['agent_correct']} "
            f"plain_tools={row['plain_tool_calls']} agent_tools={row['agent_tool_calls']} "
            f"agent_trace_tools={row['agent_tools']} "
            f"plain_groundedness={row['plain_groundedness']:.3f} "
            f"agent_groundedness={row['agent_groundedness']:.3f} "
            f"plain_ms={row['plain_latency_ms']:.3f} agent_ms={row['agent_latency_ms']:.3f} "
            f"reflections={row['agent_reflections']}"
        )
    return "\n".join(lines)


def main() -> None:
    print(_format_report(run_comparison()))


if __name__ == "__main__":
    main()
