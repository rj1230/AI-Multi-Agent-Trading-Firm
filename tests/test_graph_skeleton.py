"""
Graph wiring test — confirms state flows through every node in the right
order, for BOTH branches of the conditional edge.

IMPORTANT: run_news_agent and run_chart_agent are mocked here. Without
this, since Phase 4 wired the real NewsAgent/ChartAgent into build_graph(),
this test would hit real Groq/NewsAPI/Alpaca calls on every run — slow,
costs API quota, and flaky if those services are down or rate-limited.
This test's job is verifying graph SHAPE, not agent correctness (that's
covered by test_news_agent.py / test_chart_agent.py), so mocking the
agents' outputs is the right call, not a workaround.
"""

from unittest.mock import patch

from graph.build import build_graph
from graph.state import RiskDecision, Signal, TradingState

FAKE_NEWS_SIGNAL = Signal(direction="neutral", confidence=0.5, rationale="mocked news")
FAKE_CHART_SIGNAL = Signal(
    direction="neutral", confidence=0.5, rationale="mocked chart"
)


def _patched_agents():
    return patch.multiple(
        "graph.nodes",
        run_news_agent=lambda ticker: FAKE_NEWS_SIGNAL,
        run_chart_agent=lambda ticker: FAKE_CHART_SIGNAL,
    )


def test_graph_runs_approved_branch_in_order():
    with _patched_agents():
        graph = build_graph()
        result = graph.invoke(TradingState(ticker="AAPL"))

    node_order = [entry.node for entry in result["agent_logs"]]

    # NewsAgent/ChartAgent run in parallel — order between the two isn't
    # guaranteed, but both must precede SignalMerger, which must precede
    # RiskAgent, which must precede ExecutionAgent.
    assert set(node_order[:2]) == {"NewsAgent", "ChartAgent"}
    assert node_order[2] == "SignalMerger"
    assert node_order[3] == "RiskAgent"
    assert node_order[4] == "ExecutionAgent"
    assert "HoldNode" not in node_order

    assert result["risk_decision"] == RiskDecision.APPROVED


def test_graph_runs_rejected_branch_in_order():
    with _patched_agents():
        graph = build_graph()
        result = graph.invoke(TradingState(ticker="REJECT_TEST"))

    node_order = [entry.node for entry in result["agent_logs"]]

    assert set(node_order[:2]) == {"NewsAgent", "ChartAgent"}
    assert node_order[2] == "SignalMerger"
    assert node_order[3] == "RiskAgent"
    assert node_order[4] == "HoldNode"
    assert "ExecutionAgent" not in node_order

    assert result["risk_decision"] == RiskDecision.REJECTED
    assert len(result["risk_notes"]) > 0


def test_state_types_are_preserved():
    """Signals should come back as real Signal objects, not raw dicts —
    confirms the Pydantic schema is actually being enforced through the
    graph, not just passed through as untyped data."""
    with _patched_agents():
        graph = build_graph()
        result = graph.invoke(TradingState(ticker="AAPL"))

    assert isinstance(result["news_signal"], Signal)
    assert isinstance(result["chart_signal"], Signal)
    assert isinstance(result["merged_signal"], Signal)
