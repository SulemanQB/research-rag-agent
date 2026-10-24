import json

import pytest

from src.agent.mcp_server import build_server
from src.agent.tools import (
    LexicalRetriever,
    calculate,
    evaluate_arithmetic,
    lookup_term,
    search_arxiv,
)


def test_lexical_retriever_returns_only_the_best_matching_note():
    from evaluation.agent_eval import load_fixture_documents

    retriever = LexicalRetriever(load_fixture_documents())
    hits = retriever.retrieve("What are the four core functions named in the local risk notes?")

    assert [hit["source"] for hit in hits] == ["risk_functions.md"]


def test_calculate_evaluates_a_percentage_expression():
    observation = calculate("7 / 40 * 100")

    assert observation["result"] == "17.5"
    assert observation["text"] == "7 / 40 * 100 = 17.5"


def test_calculate_rejects_calls_and_large_exponents():
    with pytest.raises(ValueError):
        evaluate_arithmetic("__import__('os').system('echo no')")

    rejected = calculate("2 ** 100")

    assert "error" in rejected
    assert "result" not in rejected


def test_calculate_rejects_division_by_zero():
    observation = calculate("1 / 0")

    assert observation["error"] == "division by zero"


def test_lookup_returns_the_map_definition():
    observation = lookup_term("the Map function")

    assert observation["found"] is True
    assert "identifies the context, intended use, and risks" in observation["text"]
    assert observation["source"] == "glossary"


def test_lookup_reports_unknown_terms():
    observation = lookup_term("teleportation")

    assert observation["found"] is False
    assert "Known terms" in observation["text"]


def test_arxiv_search_is_off_unless_enabled(monkeypatch):
    def fail_open(*args, **kwargs):
        raise AssertionError("arXiv was contacted while search was disabled")

    monkeypatch.setattr("src.agent.tools.urlopen", fail_open)

    observation = search_arxiv("retrieval augmented generation", enabled=False)

    assert observation["enabled"] is False
    assert observation["results"] == []
    assert "disabled" in observation["message"]


def test_arxiv_search_parses_atom_results(monkeypatch):
    atom = b"""<?xml version="1.0" encoding="UTF-8"?>
    <feed xmlns="http://www.w3.org/2005/Atom">
      <entry>
        <id>http://arxiv.org/abs/2401.00001v1</id>
        <title>Retrieval  augmented   generation</title>
        <summary>A short summary of the paper.</summary>
      </entry>
    </feed>
    """

    class _Response:
        def read(self):
            return atom

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, traceback):
            return False

    monkeypatch.setattr("src.agent.tools.urlopen", lambda *args, **kwargs: _Response())

    observation = search_arxiv("retrieval augmented generation", enabled=True, max_results=1)

    assert observation["enabled"] is True
    assert observation["results"][0]["title"] == "Retrieval augmented generation"
    assert observation["results"][0]["id"] == "http://arxiv.org/abs/2401.00001v1"


def test_mcp_server_registers_the_agent_tools():
    retriever = LexicalRetriever(
        [{"id": "note", "text": "Govern is a core function.", "metadata": {"source": "notes.md", "page": 1}}]
    )
    server = build_server(retriever=retriever)
    names = {tool.name for tool in server._tool_manager.list_tools()}

    assert names == {
        "retrieve_corpus",
        "calculate_expression",
        "lookup_glossary_term",
        "search_arxiv_papers",
    }

    retrieve = server._tool_manager.get_tool("retrieve_corpus").fn
    payload = json.loads(retrieve("Govern function", top_k=1))

    assert payload["passages"][0]["source"] == "notes.md"
    assert "Govern" in payload["passages"][0]["text"]
