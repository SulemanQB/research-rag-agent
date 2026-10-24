from evaluation.agent_eval import run_comparison


def test_fixture_comparison_matches_the_measured_split():
    report = run_comparison()

    assert report["backend"] == "mock"
    assert report["questions"] == 6
    assert report["plain_rag"]["correct"] == 3
    assert report["agent"]["correct"] == 6
    assert report["plain_rag"]["mean_tool_calls"] == 1
    assert report["agent"]["mean_tool_calls"] == 1
    assert report["agent"]["reflection_retries"] == 0
    assert round(report["plain_rag"]["mean_groundedness"], 3) == 0.667
    assert round(report["agent"]["mean_groundedness"], 3) == 0.833
    assert all(row["plain_latency_ms"] > 0 and row["agent_latency_ms"] > 0 for row in report["rows"])
