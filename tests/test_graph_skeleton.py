"""
Graph wiring test — confirms state flows through every node in the right
order, for BOTH branches of the conditional edge, using the real
singleton-based nodes.

The test mocks the production metadata-aware NewsAgent API so graph tests
never perform a real news fetch.

Complements test_nodes.py rather than duplicating it:
    - approved branch
    - full-graph HoldNode routing
    - state/type preservation
    - news provenance propagation
"""

from datetime import UTC, datetime, timedelta

import graph.nodes as nodes_module
import portfolio.ledger as ledger_module
from agents.news_agent import NewsAgentResult
from data_sources.schemas import (
    DataSourceMode,
    NewsAvailability,
    OHLCVBar,
    OHLCVSeries,
)
from graph.build import build_graph
from graph.state import RiskDecision, Signal, TradingState

FAKE_ENTRY_PRICE = 150.0


def _fake_series(
    ticker: str,
    closes: list[float],
) -> OHLCVSeries:
    start = datetime(
        2026,
        1,
        1,
        tzinfo=UTC,
    )

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
        ticker=ticker,
        bars=bars,
        source="test",
        mode=DataSourceMode.LIVE,
        as_of=ts,
    )


def _setup_common(
    monkeypatch,
    tmp_path,
):
    monkeypatch.setattr(
        nodes_module,
        "_ledger",
        lambda: ledger_module.PortfolioLedger(
            starting_equity=100_000.0,
            path=tmp_path / "ledger.json",
        ),
    )

    monkeypatch.setattr(
        nodes_module,
        "_broker_singleton",
        None,
    )

    monkeypatch.setattr(
        nodes_module,
        "_price_lookup",
        lambda t: FAKE_ENTRY_PRICE,
    )

    # --------------------------------------------------------
    # Mock the production metadata-aware NewsAgent API.
    #
    # news_agent_node() calls:
    #
    #     run_news_agent_with_metadata()
    #
    # Therefore the graph test must patch this function rather
    # than the backward-compatible run_news_agent().
    # --------------------------------------------------------

    monkeypatch.setattr(
        nodes_module,
        "run_news_agent_with_metadata",
        lambda t: NewsAgentResult(
            signal=Signal(
                direction="bullish",
                confidence=0.7,
                rationale="mocked news",
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
        ),
    )

    monkeypatch.setattr(
        nodes_module,
        "run_chart_agent",
        lambda t: Signal(
            direction="bullish",
            confidence=0.6,
            rationale="mocked chart",
        ),
    )


def test_graph_runs_approved_branch_in_order(
    tmp_path,
    monkeypatch,
):
    _setup_common(
        monkeypatch,
        tmp_path,
    )

    # Wide-spread, non-monotonic-ish closes: enough ATR to keep
    # the sized position comfortably under caps.
    closes = [100 + i for i in range(31)]

    monkeypatch.setattr(
        nodes_module,
        "fetch_ohlcv",
        lambda t, lookback_days=30: _fake_series(
            t,
            closes,
        ),
    )

    graph = build_graph()

    result = graph.invoke(
        TradingState(
            ticker="AAPL",
        )
    )

    node_order = [entry.node for entry in result["agent_logs"]]

    assert set(node_order[:2]) == {
        "NewsAgent",
        "ChartAgent",
    }

    assert node_order[2] == "SignalMerger"
    assert node_order[3] == "RiskAgent"
    assert node_order[4] == "ExecutionAgent"

    assert "HoldNode" not in node_order

    assert result["risk_decision"] == RiskDecision.APPROVED

    # --------------------------------------------------------
    # News provenance should survive the graph.
    # --------------------------------------------------------

    assert result["news_availability"] == NewsAvailability.AVAILABLE.value

    assert result["news_article_count"] == 2
    assert result["news_source"] == "test"

    assert result["news_as_of"] == datetime(
        2026,
        1,
        1,
        tzinfo=UTC,
    )


def test_graph_runs_rejected_branch_in_order(
    tmp_path,
    monkeypatch,
):
    _setup_common(
        monkeypatch,
        tmp_path,
    )

    # Empty OHLCV history -> risk_agent_node's own early-exit
    # rejects for insufficient price history.
    monkeypatch.setattr(
        nodes_module,
        "fetch_ohlcv",
        lambda t, lookback_days=30: _fake_series(
            t,
            [],
        ),
    )

    graph = build_graph()

    result = graph.invoke(
        TradingState(
            ticker="AAPL",
        )
    )

    node_order = [entry.node for entry in result["agent_logs"]]

    assert set(node_order[:2]) == {
        "NewsAgent",
        "ChartAgent",
    }

    assert node_order[2] == "SignalMerger"
    assert node_order[3] == "RiskAgent"
    assert node_order[4] == "HoldNode"

    assert "ExecutionAgent" not in node_order

    assert result["risk_decision"] == RiskDecision.REJECTED

    assert len(result["risk_notes"]) > 0

    # News provenance should still exist even when the
    # trade is rejected later by RiskAgent.

    assert result["news_availability"] == NewsAvailability.AVAILABLE.value

    assert result["news_article_count"] == 2
    assert result["news_source"] == "test"


def test_state_types_are_preserved(
    tmp_path,
    monkeypatch,
):
    """
    Signals should come back as real Signal objects, not raw dicts.

    Also verifies that the metadata-aware NewsAgent integration does
    not change the existing graph state contract.
    """

    _setup_common(
        monkeypatch,
        tmp_path,
    )

    closes = [100 + i for i in range(31)]

    monkeypatch.setattr(
        nodes_module,
        "fetch_ohlcv",
        lambda t, lookback_days=30: _fake_series(
            t,
            closes,
        ),
    )

    graph = build_graph()

    result = graph.invoke(
        TradingState(
            ticker="AAPL",
        )
    )

    assert isinstance(
        result["news_signal"],
        Signal,
    )

    assert isinstance(
        result["chart_signal"],
        Signal,
    )

    assert isinstance(
        result["merged_signal"],
        Signal,
    )

    # --------------------------------------------------------
    # Provenance state types.
    # --------------------------------------------------------

    assert isinstance(
        result["news_availability"],
        str,
    )

    assert isinstance(
        result["news_article_count"],
        int,
    )

    assert result["news_article_count"] == 2

    assert result["news_source"] == "test"

    assert isinstance(
        result["news_as_of"],
        datetime,
    )
