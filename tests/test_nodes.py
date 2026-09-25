from datetime import UTC, datetime, timedelta

import pytest

import agents.chart_agent as chart_agent_module
import graph.nodes as nodes_module
import portfolio.ledger as ledger_module
from agents.news_agent import NewsAgentResult
from data_sources.schemas import (
    DataSourceMode,
    NewsAvailability,
    OHLCVBar,
    OHLCVSeries,
)
from graph.state import RiskDecision, Signal, TradingState


def _series(
    ticker: str,
    closes: list[float],
    highs=None,
    lows=None,
) -> OHLCVSeries:
    start = datetime(2026, 1, 1, tzinfo=UTC)

    highs = highs or [c + 1 for c in closes]
    lows = lows or [c - 1 for c in closes]

    bars = [
        OHLCVBar(
            timestamp=start + timedelta(days=i),
            open=c,
            high=h,
            low=l,
            close=c,
            volume=1000,
        )
        for i, (c, h, l) in enumerate(zip(closes, highs, lows))
    ]

    ts = bars[-1].timestamp if bars else start

    return OHLCVSeries(
        ticker=ticker,
        bars=bars,
        source="test",
        mode=DataSourceMode.LIVE,
        as_of=ts,
    )


def _mock_news_result(
    direction: str = "bullish",
    confidence: float = 0.7,
    rationale: str = "test news",
) -> NewsAgentResult:
    """
    Build a deterministic NewsAgentResult for graph/node tests.

    The production graph consumes NewsAgentResult so that news
    provenance is preserved alongside the directional signal.
    """
    return NewsAgentResult(
        signal=Signal(
            direction=direction,
            confidence=confidence,
            rationale=rationale,
        ),
        availability=NewsAvailability.AVAILABLE,
        article_count=2,
        source="test",
        as_of=datetime(
            2026,
            1,
            1,
            tzinfo=UTC,
        ),
    )


def test_signal_merger_node_agreement():
    state = TradingState(
        ticker="AAPL",
        news_signal=Signal(
            direction="bullish",
            confidence=0.7,
            rationale="earnings",
        ),
        chart_signal=Signal(
            direction="bullish",
            confidence=0.6,
            rationale="uptrend",
        ),
    )

    result = nodes_module.signal_merger_node(state)

    assert result["merged_signal"].direction == "bullish"
    assert result["merged_signal"].confidence == 0.65


def test_signal_merger_node_disagreement_defaults_neutral():
    state = TradingState(
        ticker="AAPL",
        news_signal=Signal(
            direction="bullish",
            confidence=0.8,
            rationale="x",
        ),
        chart_signal=Signal(
            direction="bearish",
            confidence=0.9,
            rationale="y",
        ),
    )

    result = nodes_module.signal_merger_node(state)

    assert result["merged_signal"].direction == "neutral"


def test_risk_agent_node_rejects_on_missing_price_history(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr(
        nodes_module,
        "_ledger_singleton",
        None,
    )

    monkeypatch.setattr(
        nodes_module,
        "fetch_ohlcv",
        lambda t, lookback_days=30: _series(t, []),
    )

    monkeypatch.setattr(
        nodes_module,
        "_ledger",
        lambda: ledger_module.PortfolioLedger(
            starting_equity=100_000.0,
            path=tmp_path / "ledger.json",
        ),
    )

    state = TradingState(
        ticker="AAPL",
        merged_signal=Signal(
            direction="bullish",
            confidence=0.7,
            rationale="x",
        ),
    )

    result = nodes_module.risk_agent_node(state)

    assert result["risk_decision"] == RiskDecision.REJECTED
    assert "insufficient" in result["risk_notes"][0].lower()


def test_risk_agent_node_approves_clean_trade(
    tmp_path,
    monkeypatch,
):
    closes = [100 + i for i in range(31)]

    # Wide high/low spread -> larger ATR -> smaller position size.
    # This keeps the resulting trade comfortably below the
    # configured per-ticker/sector caps.
    highs = [c + 5 for c in closes]
    lows = [c - 5 for c in closes]

    monkeypatch.setattr(
        nodes_module,
        "fetch_ohlcv",
        lambda t, lookback_days=30: _series(
            t,
            closes,
            highs,
            lows,
        ),
    )

    monkeypatch.setattr(
        nodes_module,
        "_ledger",
        lambda: ledger_module.PortfolioLedger(
            starting_equity=100_000.0,
            path=tmp_path / "ledger.json",
        ),
    )

    state = TradingState(
        ticker="AAPL",
        merged_signal=Signal(
            direction="bullish",
            confidence=0.7,
            rationale="x",
        ),
    )

    result = nodes_module.risk_agent_node(state)

    assert result["risk_decision"] == RiskDecision.APPROVED
    assert result["proposed_shares"] > 0
    assert result["sector"] == "Tech"


def test_full_graph_bullish_signal_executes_via_sim_broker(
    tmp_path,
    monkeypatch,
):
    """
    End-to-end graph test:

        NewsAgent
            +
        ChartAgent
            ↓
        SignalMerger
            ↓
        RiskAgent
            ↓
        ExecutionAgent
            ↓
        SimBroker
            ↓
        PortfolioLedger

    Both NewsAgent and ChartAgent produce bullish signals, so the
    SignalMerger agrees and RiskAgent approves the trade.

    The test also verifies the critical execution-state invariant:

        broker shares == ledger shares
        broker cash   == ledger cash
    """

    from graph.build import build_graph

    # ------------------------------------------------------------
    # Deterministic market data
    # ------------------------------------------------------------

    closes = [100.0]

    for i in range(59):
        closes.append(closes[-1] - 1.3 if i % 2 == 1 else closes[-1] + 2.0)

    highs = [c + 5 for c in closes]
    lows = [c - 5 for c in closes]

    series = _series(
        "AAPL",
        closes,
        highs,
        lows,
    )

    # ------------------------------------------------------------
    # ChartAgent market-data dependency
    # ------------------------------------------------------------

    monkeypatch.setattr(
        chart_agent_module,
        "fetch_ohlcv",
        lambda t, lookback_days=60: series,
    )

    # ------------------------------------------------------------
    # RiskAgent market-data dependency
    # ------------------------------------------------------------

    monkeypatch.setattr(
        nodes_module,
        "fetch_ohlcv",
        lambda t, lookback_days=30: series,
    )

    # ------------------------------------------------------------
    # NewsAgent dependency
    # ------------------------------------------------------------

    # Prevent the test from reaching the real news/data-source layer.
    monkeypatch.setattr(
        nodes_module,
        "run_news_agent_with_metadata",
        lambda t: _mock_news_result(
            direction="bullish",
            confidence=0.7,
            rationale="test news",
        ),
    )

    # ------------------------------------------------------------
    # Isolated ledger
    # ------------------------------------------------------------

    ledger = ledger_module.PortfolioLedger(
        starting_equity=100_000.0,
        path=tmp_path / "ledger.json",
    )

    monkeypatch.setattr(
        nodes_module,
        "_ledger_singleton",
        ledger,
    )

    monkeypatch.setattr(
        nodes_module,
        "_ledger",
        lambda: ledger,
    )

    # ------------------------------------------------------------
    # Fresh broker
    # ------------------------------------------------------------

    monkeypatch.setattr(
        nodes_module,
        "_broker_singleton",
        None,
    )

    monkeypatch.setattr(
        nodes_module,
        "_price_lookup",
        lambda t: closes[-1],
    )

    # ------------------------------------------------------------
    # Execute graph
    # ------------------------------------------------------------

    graph = build_graph()

    result = graph.invoke(
        TradingState(
            ticker="AAPL",
        )
    )

    # ------------------------------------------------------------
    # Risk/execution assertions
    # ------------------------------------------------------------

    assert result["risk_decision"] == RiskDecision.APPROVED

    assert result["execution_success"] is True

    # ------------------------------------------------------------
    # Agent execution path
    # ------------------------------------------------------------

    node_names = [entry.node for entry in result["agent_logs"]]

    assert "NewsAgent" in node_names
    assert "ChartAgent" in node_names
    assert "SignalMerger" in node_names
    assert "RiskAgent" in node_names
    assert "ExecutionAgent" in node_names
    assert "HoldNode" not in node_names

    # ------------------------------------------------------------
    # News provenance
    # ------------------------------------------------------------

    assert result["news_availability"] == NewsAvailability.AVAILABLE.value

    assert result["news_article_count"] == 2
    assert result["news_source"] == "test"

    assert result["news_as_of"] == datetime(
        2026,
        1,
        1,
        tzinfo=UTC,
    )

    # ------------------------------------------------------------
    # Signal assertions
    # ------------------------------------------------------------

    assert result["news_signal"].direction == "bullish"
    assert result["chart_signal"].direction == "bullish"
    assert result["merged_signal"].direction == "bullish"
    assert result["merge_agreement"] is True

    # ------------------------------------------------------------
    # CRITICAL:
    # Verify actual broker ↔ ledger synchronization.
    # ------------------------------------------------------------

    broker = nodes_module._broker_singleton

    assert broker is not None

    broker_positions = broker.get_positions()

    # Successful execution must create a broker position.
    assert "AAPL" in broker_positions

    broker_position = broker_positions["AAPL"]

    assert broker_position.qty > 0

    # Successful execution must also create a ledger position.
    assert "AAPL" in ledger._data["positions"]

    ledger_position = ledger._data["positions"]["AAPL"]

    assert ledger_position["shares"] > 0

    # The same successful fill must be represented identically.
    assert broker_position.qty == pytest.approx(ledger_position["shares"])

    # Broker and ledger cash must also agree.
    assert broker.cash == pytest.approx(ledger._data["cash"])

    # The trade must have consumed some cash.
    assert broker.cash < 100_000.0
    assert ledger._data["cash"] < 100_000.0
