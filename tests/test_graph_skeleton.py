"""
Graph wiring test — confirms state flows through every node in the right
order, for BOTH branches of the conditional edge, using the real
singleton-based nodes (see tests/test_nodes.py for the established
patching patterns this file reuses: _ledger as a function override,
fetch_ohlcv patched directly on graph.nodes, _broker_singleton reset +
_price_lookup for a real SimBroker run).

Complements test_nodes.py rather than duplicating it: that file covers
the approved branch and unit-level RiskAgent rejection; this file adds
full-graph HoldNode routing and a dedicated type-preservation check.
"""

import graph.nodes as nodes_module
import portfolio.ledger as ledger_module
from data_sources.schemas import DataSourceMode, OHLCVBar, OHLCVSeries
from graph.build import build_graph
from graph.state import RiskDecision, Signal, TradingState

FAKE_ENTRY_PRICE = 150.0


def _fake_series(ticker: str, closes: list[float]) -> OHLCVSeries:
    from datetime import datetime, timedelta, timezone

    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    bars = [
        OHLCVBar(
            timestamp=start + timedelta(days=i),
            open=c,
            high=c + 5,
            low=c - 5,
            close=c,
            volume=1000,
        )
        for i, c in enumerate(closes)
    ]
    ts = bars[-1].timestamp if bars else start
    return OHLCVSeries(
        ticker=ticker, bars=bars, source="test", mode=DataSourceMode.LIVE, as_of=ts
    )


def _setup_common(monkeypatch, tmp_path):
    monkeypatch.setattr(
        nodes_module,
        "_ledger",
        lambda: ledger_module.PortfolioLedger(
            starting_equity=100_000.0, path=tmp_path / "ledger.json"
        ),
    )
    monkeypatch.setattr(nodes_module, "_broker_singleton", None)
    monkeypatch.setattr(nodes_module, "_price_lookup", lambda t: FAKE_ENTRY_PRICE)
    monkeypatch.setattr(
        nodes_module,
        "run_news_agent",
        lambda t: Signal(direction="bullish", confidence=0.7, rationale="mocked news"),
    )
    monkeypatch.setattr(
        nodes_module,
        "run_chart_agent",
        lambda t: Signal(direction="bullish", confidence=0.6, rationale="mocked chart"),
    )


def test_graph_runs_approved_branch_in_order(tmp_path, monkeypatch):
    _setup_common(monkeypatch, tmp_path)
    # Wide-spread, non-monotonic-ish closes: enough ATR to keep the sized
    # position comfortably under caps (same reasoning as
    # test_risk_agent_node_approves_clean_trade).
    closes = [100 + i for i in range(31)]
    monkeypatch.setattr(
        nodes_module, "fetch_ohlcv", lambda t, lookback_days=30: _fake_series(t, closes)
    )

    graph = build_graph()
    result = graph.invoke(TradingState(ticker="AAPL"))

    node_order = [entry.node for entry in result["agent_logs"]]
    assert set(node_order[:2]) == {"NewsAgent", "ChartAgent"}
    assert node_order[2] == "SignalMerger"
    assert node_order[3] == "RiskAgent"
    assert node_order[4] == "ExecutionAgent"
    assert "HoldNode" not in node_order
    assert result["risk_decision"] == RiskDecision.APPROVED


def test_graph_runs_rejected_branch_in_order(tmp_path, monkeypatch):
    _setup_common(monkeypatch, tmp_path)
    # Empty OHLCV history -> risk_agent_node's own early-exit rejects for
    # "insufficient price history for sizing" -- a real rejection path,
    # not a hardcoded ticker-name hook (see risk_agent_node's own guard).
    monkeypatch.setattr(
        nodes_module, "fetch_ohlcv", lambda t, lookback_days=30: _fake_series(t, [])
    )

    graph = build_graph()
    result = graph.invoke(TradingState(ticker="AAPL"))

    node_order = [entry.node for entry in result["agent_logs"]]
    assert set(node_order[:2]) == {"NewsAgent", "ChartAgent"}
    assert node_order[2] == "SignalMerger"
    assert node_order[3] == "RiskAgent"
    assert node_order[4] == "HoldNode"
    assert "ExecutionAgent" not in node_order
    assert result["risk_decision"] == RiskDecision.REJECTED
    assert len(result["risk_notes"]) > 0


def test_state_types_are_preserved(tmp_path, monkeypatch):
    """Signals should come back as real Signal objects, not raw dicts —
    confirms the Pydantic schema is enforced through the graph."""
    _setup_common(monkeypatch, tmp_path)
    closes = [100 + i for i in range(31)]
    monkeypatch.setattr(
        nodes_module, "fetch_ohlcv", lambda t, lookback_days=30: _fake_series(t, closes)
    )

    graph = build_graph()
    result = graph.invoke(TradingState(ticker="AAPL"))

    assert isinstance(result["news_signal"], Signal)
    assert isinstance(result["chart_signal"], Signal)
    assert isinstance(result["merged_signal"], Signal)
