from datetime import datetime, timedelta, timezone

import graph.nodes as nodes_module
import agents.chart_agent as chart_agent_module
import portfolio.ledger as ledger_module
from data_sources.schemas import OHLCVBar, OHLCVSeries, DataSourceMode
from graph.state import RiskDecision, Signal, TradingState


def _series(ticker: str, closes: list[float], highs=None, lows=None) -> OHLCVSeries:
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    highs = highs or [c + 1 for c in closes]
    lows = lows or [c - 1 for c in closes]
    bars = [
        OHLCVBar(timestamp=start + timedelta(days=i), open=c, high=h, low=l, close=c, volume=1000)
        for i, (c, h, l) in enumerate(zip(closes, highs, lows))
    ]
    ts = bars[-1].timestamp if bars else start
    return OHLCVSeries(ticker=ticker, bars=bars, source="test", mode=DataSourceMode.LIVE, as_of=ts)


def test_signal_merger_node_agreement():
    state = TradingState(
        ticker="AAPL",
        news_signal=Signal(direction="bullish", confidence=0.7, rationale="earnings"),
        chart_signal=Signal(direction="bullish", confidence=0.6, rationale="uptrend"),
    )
    result = nodes_module.signal_merger_node(state)
    assert result["merged_signal"].direction == "bullish"
    assert result["merged_signal"].confidence == 0.65


def test_signal_merger_node_disagreement_defaults_neutral():
    state = TradingState(
        ticker="AAPL",
        news_signal=Signal(direction="bullish", confidence=0.8, rationale="x"),
        chart_signal=Signal(direction="bearish", confidence=0.9, rationale="y"),
    )
    result = nodes_module.signal_merger_node(state)
    assert result["merged_signal"].direction == "neutral"


def test_risk_agent_node_rejects_on_missing_price_history(tmp_path, monkeypatch):
    monkeypatch.setattr(nodes_module, "_ledger_singleton", None)
    monkeypatch.setattr(nodes_module, "fetch_ohlcv", lambda t, lookback_days=30: _series(t, []))
    monkeypatch.setattr(
        nodes_module, "_ledger",
        lambda: ledger_module.PortfolioLedger(starting_equity=100_000.0, path=tmp_path / "ledger.json"),
    )

    state = TradingState(
        ticker="AAPL",
        merged_signal=Signal(direction="bullish", confidence=0.7, rationale="x"),
    )
    result = nodes_module.risk_agent_node(state)
    assert result["risk_decision"] == RiskDecision.REJECTED
    assert "insufficient" in result["risk_notes"][0].lower()


def test_risk_agent_node_approves_clean_trade(tmp_path, monkeypatch):
    closes = [100 + i for i in range(31)]
    # Wide high/low spread -> larger ATR -> smaller sized position, so the
    # resulting trade lands comfortably under the per-ticker/sector caps
    # instead of testing an (also legitimate) oversized-position rejection.
    highs = [c + 5 for c in closes]
    lows = [c - 5 for c in closes]
    monkeypatch.setattr(nodes_module, "fetch_ohlcv", lambda t, lookback_days=30: _series(t, closes, highs, lows))
    monkeypatch.setattr(
        nodes_module, "_ledger",
        lambda: ledger_module.PortfolioLedger(starting_equity=100_000.0, path=tmp_path / "ledger.json"),
    )

    state = TradingState(
        ticker="AAPL",
        merged_signal=Signal(direction="bullish", confidence=0.7, rationale="x"),
    )
    result = nodes_module.risk_agent_node(state)
    assert result["risk_decision"] == RiskDecision.APPROVED
    assert result["proposed_shares"] > 0
    assert result["sector"] == "Tech"  # AAPL is in the placeholder SECTOR_MAP


def test_full_graph_bullish_signal_executes_via_sim_broker(tmp_path, monkeypatch):
    """End-to-end: NewsAgent + ChartAgent both bullish -> SignalMerger
    agrees -> RiskAgent approves -> ExecutionAgent fills via SimBroker."""
    from graph.build import build_graph

    # Alternating up/down with net upward drift -- a monotonic uptrend
    # pins RSI at 100 (overbought), which ties against SMA's bullish vote
    # and produces neutral, not bullish. Real price series aren't
    # monotonic, so this fixture is the more realistic one to test against.
    closes = [100.0]
    for i in range(59):
        closes.append(closes[-1] - 1.3 if i % 2 == 1 else closes[-1] + 2.0)
    highs = [c + 5 for c in closes]
    lows = [c - 5 for c in closes]
    series = _series("AAPL", closes, highs, lows)

    monkeypatch.setattr(chart_agent_module, "fetch_ohlcv", lambda t, lookback_days=60: series)
    monkeypatch.setattr(nodes_module, "fetch_ohlcv", lambda t, lookback_days=30: series)
    monkeypatch.setattr(nodes_module, "run_news_agent", lambda t: Signal(direction="bullish", confidence=0.7, rationale="test news"))
    monkeypatch.setattr(
        nodes_module, "_ledger",
        lambda: ledger_module.PortfolioLedger(starting_equity=100_000.0, path=tmp_path / "ledger.json"),
    )
    monkeypatch.setattr(nodes_module, "_broker_singleton", None)
    monkeypatch.setattr(nodes_module, "_price_lookup", lambda t: closes[-1])

    graph = build_graph()
    result = graph.invoke(TradingState(ticker="AAPL"))

    assert result["risk_decision"] == RiskDecision.APPROVED
    node_names = [entry.node for entry in result["agent_logs"]]
    assert "ExecutionAgent" in node_names
    assert "HoldNode" not in node_names
