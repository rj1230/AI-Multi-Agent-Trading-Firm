"""
Integration tests for the multi-ticker orchestrator.

These tests exercise the complete local orchestration path while keeping
external data sources, the portfolio ledger, and broker state isolated.

Isolation notes
---------------
- graph.nodes.fetch_ohlcv is patched directly because nodes.py imports the
  facade into its own module namespace.
- graph.nodes._ledger is patched so every test receives an isolated,
  pre-initialized portfolio ledger.
- graph.nodes._broker_singleton is reset so broker state cannot leak between
  tests.
- graph.nodes._price_lookup is patched so execution uses deterministic prices.
- orchestrator.tick_runner.build_correlation_matrix is patched separately
  because tick_runner imports that symbol directly from portfolio.state.
  Patching nodes_module.fetch_ohlcv therefore does not affect that binding.

Concurrency note
----------------
run_tick() executes ticker pipelines concurrently using worker threads.
Therefore the test creates exactly one PortfolioLedger instance before
concurrent execution starts and returns that same instance from _ledger().
Creating separate PortfolioLedger instances against the same JSON file from
multiple worker threads can race during file initialization.

The tests intentionally do not depend on the ambient TRADING_MODE
environment variable. This prevents a previous manual backtest session from
changing the behavior of the integration tests.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone

import pytest

import graph.nodes as nodes_module
import portfolio.ledger as ledger_module
from agents.news_agent import NewsAgentResult
from data_sources.schemas import (
    DataSourceMode,
    NewsAvailability,
    OHLCVBar,
    OHLCVSeries,
)
from graph.state import Signal
from orchestrator.tick_runner import run_tick


# ---------------------------------------------------------------------------
# Deterministic test configuration
# ---------------------------------------------------------------------------

FAKE_ATR = 10.0

# Sized so the proposed position value is approximately 19% of
# a $100,000 account, leaving room for portfolio allocation checks.
FAKE_ENTRY_PRICE = 285.0

FAKE_CONFIDENCE = {
    "GOOGL": 0.9,
    "MSFT": 0.8,
    "AAPL": 0.7,
}


# ---------------------------------------------------------------------------
# Fake data / agent responses
# ---------------------------------------------------------------------------


def _fake_series(
    ticker: str,
    lookback_days: int,
) -> OHLCVSeries:
    """
    Return deterministic OHLCV data for integration tests.

    The ticker and lookback arguments are intentionally accepted so this
    function matches the production fetch_ohlcv interface.
    """
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)

    bar = OHLCVBar(
        timestamp=start,
        open=FAKE_ENTRY_PRICE,
        high=FAKE_ENTRY_PRICE + 1,
        low=FAKE_ENTRY_PRICE - 1,
        close=FAKE_ENTRY_PRICE,
        volume=1_000_000,
    )

    return OHLCVSeries(
        ticker=ticker,
        bars=[bar],
        source="test",
        mode=DataSourceMode.LIVE,
        as_of=start + timedelta(days=1),
    )


def _fake_news(ticker: str) -> NewsAgentResult:
    """
    Return a deterministic metadata-aware news result.

    The production graph consumes run_news_agent_with_metadata(),
    so tests must mock the complete NewsAgentResult rather than the
    backward-compatible run_news_agent() signal-only API.
    """
    return NewsAgentResult(
        signal=Signal(
            direction="bullish",
            confidence=FAKE_CONFIDENCE[ticker],
            rationale="mocked news",
        ),
        availability=NewsAvailability.AVAILABLE,
        article_count=2,
        source="test",
        as_of=datetime(
            2026,
            1,
            1,
            tzinfo=timezone.utc,
        ),
    )


def _fake_chart(ticker: str) -> Signal:
    """Return a deterministic bullish chart signal."""
    return Signal(
        direction="bullish",
        confidence=FAKE_CONFIDENCE[ticker],
        rationale="mocked chart",
    )


# ---------------------------------------------------------------------------
# Shared integration-test fixture
# ---------------------------------------------------------------------------


@pytest.fixture
def patched_singletons(tmp_path, monkeypatch):
    """
    Isolate ledger, broker, market data, and correlation state.

    A single PortfolioLedger instance is shared by all ticker worker threads.

    This is important because run_tick() executes ticker pipelines
    concurrently. Creating a separate PortfolioLedger for each worker would
    cause multiple threads to initialize/read the same JSON file concurrently,
    which can produce an empty-file JSONDecodeError.
    """

    ledger_path = tmp_path / "ledger.json"

    # Guarantee that PortfolioLedger takes its initialization path.
    if ledger_path.exists():
        ledger_path.unlink()

    # ---------------------------------------------------------------
    # Initialize the ledger exactly once before concurrent execution.
    # ---------------------------------------------------------------

    test_ledger = ledger_module.PortfolioLedger(
        starting_equity=100_000.0,
        path=ledger_path,
    )

    monkeypatch.setattr(
        nodes_module,
        "_ledger",
        lambda: test_ledger,
    )

    # ---------------------------------------------------------------
    # Fresh broker state for every test.
    # ---------------------------------------------------------------

    monkeypatch.setattr(
        nodes_module,
        "_broker_singleton",
        None,
    )

    # ---------------------------------------------------------------
    # Deterministic market price.
    # ---------------------------------------------------------------

    monkeypatch.setattr(
        nodes_module,
        "_price_lookup",
        lambda ticker: FAKE_ENTRY_PRICE,
    )

    # ---------------------------------------------------------------
    # Deterministic upstream agents.
    #
    # IMPORTANT:
    # news_agent_node() now calls run_news_agent_with_metadata().
    # Therefore we patch the metadata-aware function directly.
    # ---------------------------------------------------------------

    monkeypatch.setattr(
        nodes_module,
        "run_news_agent_with_metadata",
        _fake_news,
    )

    monkeypatch.setattr(
        nodes_module,
        "run_chart_agent",
        _fake_chart,
    )

    # ---------------------------------------------------------------
    # Deterministic OHLCV source.
    # ---------------------------------------------------------------

    monkeypatch.setattr(
        nodes_module,
        "fetch_ohlcv",
        _fake_series,
    )

    # ---------------------------------------------------------------
    # Deterministic ATR.
    # ---------------------------------------------------------------

    monkeypatch.setattr(
        nodes_module,
        "compute_atr",
        lambda bars: FAKE_ATR,
    )

    # ---------------------------------------------------------------
    # tick_runner imports build_correlation_matrix directly from
    # portfolio.state, so patch that binding separately.
    # ---------------------------------------------------------------

    monkeypatch.setattr(
        "orchestrator.tick_runner.build_correlation_matrix",
        lambda tickers: None,
    )


# ---------------------------------------------------------------------------
# Test 1: successful multi-ticker orchestration
# ---------------------------------------------------------------------------


def test_three_correlated_sector_proposals_highest_conviction_wins_end_to_end(
    patched_singletons,
):
    """
    Verify the complete successful multi-ticker orchestration path.

    GOOGL and MSFT have the highest conviction and fit within the configured
    sector cap.

    AAPL has the lowest conviction and is rejected by the portfolio
    coordinator.
    """

    results = asyncio.run(run_tick(["GOOGL", "MSFT", "AAPL"]))

    # ---------------------------------------------------------------
    # Final orchestration outcomes.
    # ---------------------------------------------------------------

    assert results["GOOGL"].outcome == "executed"
    assert results["MSFT"].outcome == "executed"
    assert results["AAPL"].outcome == "held_book_reject"

    # ---------------------------------------------------------------
    # Successful execution must create broker positions.
    # ---------------------------------------------------------------

    positions = nodes_module.get_broker().get_positions()

    assert "GOOGL" in positions
    assert "MSFT" in positions

    # AAPL was rejected before execution.
    assert "AAPL" not in positions


# ---------------------------------------------------------------------------
# Test 2: execution failure regression test
# ---------------------------------------------------------------------------


def test_execution_failure_is_not_reported_as_executed(
    patched_singletons,
    monkeypatch,
):
    """
    Regression test for execution-result semantics.

    A coordinator-approved trade must only be reported as "executed" when
    the execution agent confirms that the broker order actually succeeded.

    If execution fails, the orchestrator must report:

        held_execution_reject

    and must not create a broker position.
    """

    def fake_execution_agent(state):
        """
        Simulate a broker/execution failure after portfolio approval.
        """
        return {
            "execution_success": False,
            "execution_notes": (
                "Execution failed after 2 attempt(s): insufficient shares. "
                "Falling back to hold."
            ),
        }

    monkeypatch.setattr(
        "orchestrator.tick_runner.nodes.execution_agent_node",
        fake_execution_agent,
    )

    results = asyncio.run(run_tick(["GOOGL", "MSFT", "AAPL"]))

    # ---------------------------------------------------------------
    # Coordinator approved GOOGL/MSFT, but execution failed.
    # Therefore neither trade may be reported as executed.
    # ---------------------------------------------------------------

    assert results["GOOGL"].outcome == "held_execution_reject"
    assert results["MSFT"].outcome == "held_execution_reject"

    # ---------------------------------------------------------------
    # Failed execution must not report allocated shares as executed.
    # ---------------------------------------------------------------

    assert results["GOOGL"].shares == 0.0
    assert results["MSFT"].shares == 0.0

    # ---------------------------------------------------------------
    # Execution failure reason must propagate to the final result.
    # ---------------------------------------------------------------

    assert "Execution failed" in results["GOOGL"].notes
    assert "Falling back to hold" in results["GOOGL"].notes

    # ---------------------------------------------------------------
    # AAPL is still rejected by the portfolio coordinator.
    # ---------------------------------------------------------------

    assert results["AAPL"].outcome == "held_book_reject"
