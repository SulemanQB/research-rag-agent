from src.router import route_query


def test_corpus_query_routes_to_corpus_search():
    result = route_query("What do the papers say about federated averaging?")

    assert result["route"] == "CORPUS_SEARCH"
    assert "corpus" in result["reason"].lower() or "papers" in result["reason"].lower()


def test_general_knowledge_routes_to_general_answer():
    result = route_query("What is the capital of France?")

    assert result["route"] == "GENERAL_ANSWER"


def test_nist_research_question_routes_to_corpus_search():
    result = route_query("What are the four core functions of the NIST AI Risk Management Framework?")

    assert result["route"] == "CORPUS_SEARCH"
