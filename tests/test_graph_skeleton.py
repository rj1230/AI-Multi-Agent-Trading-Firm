"""
Graph wiring test — confirms state flows through every node in the right
order, for BOTH branches of the conditional edge.

Phase 5/6 update: build_graph() now takes config/portfolio/broker, and the
real risk_agent_node calls fetch_ohlcv() internally for ATR/entry-price —
that's mocked here too, alongside run_news_agent/run_chart_agent, since
this test's job is graph SHAPE, not agent or risk-math correctness (those
are covered by test_news_agent.py / test_chart_agent.py / test_risk_agent.py).

FAKE_NEWS_SIGNAL/FAKE_CHART_SIGNAL are deliberately bullish + agreeing, not
neutral: RiskAgent rejects every neutral merged signal outright (see
test_risk_agent.py::test_neutral_signal_is_never_approved), so a neutral
fixture here would make the "approved branch" test fail for the wrong
reason -- signal direction, not graph wiring.
"""

from unittest.mock import patch

from broker.sim_broker import SimBroker
from config.risk_config import load_risk_config
from graph.build import build_graph
from graph.state import RiskDecision, Signal, TradingState
from portfolio.state import PortfolioSnapshot

FAKE_NEWS_SIGNAL = Signal(direction="bullish", confidence=0.8, rationale="mocked news")
FAKE_CHART_SIGNAL = Signal(
    direction="bullish", confidence=0.7, rationale="mocked chart"
)

# Fixed ATR/entry-price so RiskAgent's sizing math is deterministic here,
# independent of TRADING_MODE or real/cached OHLCV data.
FAKE_ATR = 10.0
FAKE_ENTRY_PRICE = 150.0


def _patched_agents():
    return patch.multiple(
        "graph.nodes",
        run_news_agent=lambda ticker: FAKE_NEWS_SIGNAL,
        run_chart_agent=lambda ticker: FAKE_CHART_SIGNAL,
        _compute_atr_and_entry_price=lambda ticker, period=14: (
            FAKE_ATR,
            FAKE_ENTRY_PRICE,
        ),
    )


def _fresh_graph():
    config = load_risk_config()
    portfolio = PortfolioSnapshot(
        equity=100_000.0,
        starting_equity=100_000.0,
        positions={},
        correlation_matrix=None,
    )
    broker = SimBroker()
    return build_graph(config, portfolio, broker)


def test_graph_runs_approved_branch_in_order():
    with _patched_agents():
        graph = _fresh_graph()
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
        graph = _fresh_graph()
        # REJECT_TEST has no sector_map entry -- real RiskAgent rejects it
        # for that reason now, not a hardcoded ticker-name hook.
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
        graph = _fresh_graph()
        result = graph.invoke(TradingState(ticker="AAPL"))

    assert isinstance(result["news_signal"], Signal)
    assert isinstance(result["chart_signal"], Signal)
    assert isinstance(result["merged_signal"], Signal)


def _fresh_graph():
    config = load_risk_config()
    portfolio = PortfolioSnapshot(
        equity=100_000.0,
        starting_equity=100_000.0,
        positions={},
        correlation_matrix=None,
    )
    broker = SimBroker(
        starting_cash=100_000.0,
        price_lookup=lambda ticker: FAKE_ENTRY_PRICE,
    )
    return build_graph(config, portfolio, broker)
