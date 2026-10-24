import pytest

from src.agent.graph import build_agent_graph, run_agent
from src.agent.llm import MockChatModel, OpenAICompatibleChat, parse_action
from src.agent.tools import LexicalRetriever


NOTES = [
    {
        "id": "risk-functions",
        "text": "Govern Map Measure and Manage are the four core functions in the local notes.",
        "metadata": {"source": "risk_functions.md", "page": 2},
    }
]


class ScriptedChatModel:
    """Chat model with a fixed plan, action sequence, and draft sequence."""

    def __init__(self, decisions, drafts=None, plan="scripted plan"):
        self.decisions = list(decisions)
        self.drafts = list(drafts or [])
        self.plan_text = plan

    def plan(self, question: str) -> str:
        return self.plan_text

    def decide(self, question: str, tool_calls, *, web_search_enabled: bool):
        if self.decisions:
            decision = self.decisions.pop(0)
            if decision is not None:
                return decision
        return {"action": "retrieve", "arguments": {"query": question}, "thought": "retrieve again"}

    def draft(self, question: str, evidence, tool_calls) -> str:
        if self.drafts:
            return self.drafts.pop(0)
        return "fallback draft"


def _retriever():
    return LexicalRetriever(NOTES)


def test_graph_has_routing_tool_and_reflection_nodes():
    graph = build_agent_graph(MockChatModel(), _retriever())
    names = set(graph.get_graph().nodes)

    assert {
        "plan",
        "decide",
        "retrieve",
        "calculate",
        "lookup",
        "search",
        "draft",
        "reflect",
        "finalize",
    } <= names


def test_router_sends_corpus_math_and_glossary_questions_to_different_tools():
    retriever = _retriever()
    model = MockChatModel()

    corpus = run_agent(
        "What do the local notes say about the four core functions?",
        chat_model=model,
        retriever=retriever,
    )
    math = run_agent("Calculate 7 / 40 * 100.", chat_model=MockChatModel(), retriever=retriever)
    glossary = run_agent(
        "Look up the glossary entry for Map.",
        chat_model=MockChatModel(),
        retriever=retriever,
    )

    assert [call["name"] for call in corpus["tool_calls"]] == ["retrieve"]
    assert "Govern" in corpus["answer"]
    assert "risk_functions.md — page 2" in corpus["citations"]
    assert corpus["trace"][0] == "plan"
    assert "decide" in corpus["trace"]
    assert corpus["grounded"] is True

    assert [call["name"] for call in math["tool_calls"]] == ["calculate"]
    assert math["tool_calls"][0]["observation"]["result"] == "17.5"
    assert "17.5" in math["answer"]
    assert "calculator" in math["citations"]

    assert [call["name"] for call in glossary["tool_calls"]] == ["lookup"]
    assert "identifies the context, intended use, and risks" in glossary["answer"]
    assert "glossary" in glossary["citations"]


def test_step_limit_stops_repeated_retrieval():
    model = ScriptedChatModel(decisions=[None, None, None, None])
    result = run_agent(
        "What do the local notes say about the four core functions?",
        chat_model=model,
        retriever=_retriever(),
        max_steps=2,
    )

    retrieves = [call for call in result["tool_calls"] if call["name"] == "retrieve"]
    assert len(retrieves) == 2
    assert result["step_count"] == 2
    assert result["trace"].count("retrieve") == 2
    assert result["trace"][-1] == "finalize"
    assert result["trace"].count("decide") == 3


def test_step_limit_blocks_reflection_retry():
    model = ScriptedChatModel(
        decisions=[
            {"action": "retrieve", "arguments": {"query": "core functions"}, "thought": "retrieve"},
            {"action": "draft", "arguments": {}, "thought": "draft"},
        ],
        drafts=["unrelated pottery glaze schedule"],
    )
    result = run_agent(
        "What do the local notes say about the four core functions?",
        chat_model=model,
        retriever=_retriever(),
        max_steps=2,
        max_reflections=1,
    )

    assert result["trace"].count("retrieve") == 1
    assert result["reflect_count"] == 0
    assert result["grounded"] is False
    assert result["trace"][-1] == "finalize"


def test_reflection_retrieves_again_when_the_draft_is_not_grounded():
    grounded_draft = "Govern Map Measure and Manage are the four core functions in the local notes."
    model = ScriptedChatModel(
        decisions=[
            {"action": "retrieve", "arguments": {"query": "four core functions in the local notes"}, "thought": "retrieve"},
            {"action": "draft", "arguments": {}, "thought": "draft"},
        ],
        drafts=["unrelated pottery glaze schedule", grounded_draft],
    )
    result = run_agent(
        "What do the local notes say about the four core functions?",
        chat_model=model,
        retriever=_retriever(),
        max_steps=4,
        max_reflections=1,
    )

    retrieve_calls = [call for call in result["tool_calls"] if call["name"] == "retrieve"]
    first_reflect = result["trace"].index("reflect")
    second_retrieve = result["trace"].index("retrieve", result["trace"].index("retrieve") + 1)

    assert len(retrieve_calls) == 2
    assert retrieve_calls[0]["arguments"]["query"] != retrieve_calls[1]["arguments"]["query"]
    assert first_reflect < second_retrieve
    assert result["reflect_count"] == 1
    assert result["grounded"] is True
    assert grounded_draft in result["answer"]


def test_disabled_web_search_does_not_touch_the_network(monkeypatch):
    def fail_open(*args, **kwargs):
        raise AssertionError("network call")

    monkeypatch.setattr("src.agent.tools.urlopen", fail_open)
    result = run_agent(
        "Search arXiv for papers on retrieval augmented generation.",
        chat_model=MockChatModel(),
        retriever=_retriever(),
        web_search_enabled=False,
    )

    assert [call["name"] for call in result["tool_calls"]] == ["search"]
    assert result["tool_calls"][0]["observation"]["enabled"] is False
    assert "disabled" in result["answer"].lower()


def test_hosted_backend_parses_a_json_action_and_falls_back_when_json_is_invalid():
    class _Client:
        def __init__(self, responses):
            self.responses = list(responses)
            self.temperatures = []

        def generate(self, prompt: str, temperature=None):
            self.temperatures.append(temperature)
            return self.responses.pop(0)

    routed = _Client(['{"action":"retrieve","arguments":{"query":"core functions"},"thought":"look up the notes"}'])
    model = OpenAICompatibleChat(client=routed)
    decision = model.decide("What do the local notes say?", [], web_search_enabled=False)

    assert decision["action"] == "retrieve"
    assert decision["arguments"]["query"] == "core functions"
    assert routed.temperatures == [0]

    fallback = OpenAICompatibleChat(client=_Client(["I think you should search your files."]))
    decision = fallback.decide("Calculate 7 / 40 * 100.", [], web_search_enabled=False)

    assert decision["action"] == "calculate"
    assert "deterministic action policy" in decision["thought"]


def test_parse_action_rejects_unknown_actions():
    with pytest.raises(ValueError, match="unknown action"):
        parse_action('{"action":"delete","arguments":{},"thought":"no"}')


def test_run_agent_rejects_empty_questions():
    with pytest.raises(ValueError, match="question"):
        run_agent("  ", chat_model=MockChatModel(), retriever=_retriever())
