"""
Integration tests for backtest/runner.py.

Regression coverage for:

1. The backtest lookahead guard in data_sources/__init__.py.
2. End-to-end RiskAgent position-cap clamping.
3. Historical session discovery from OHLCV data.
4. Restoration of backtest singletons and simulated-date state.

The historical-data fixture intentionally returns four daily bars. This is
important because backtest/runner.py builds its session calendar from the
bars returned by data_sources.fetch_ohlcv(). A fixture that returned only
the final simulated date would cause the runner to discover only that
single session and would not exercise the multi-session backtest.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

import data_sources.historical as historical_module
import graph.nodes as nodes_module
from data_sources import get_simulated_date, set_simulated_date
from data_sources.schemas import DataSourceMode, OHLCVBar, OHLCVSeries
from graph.state import Signal

FAKE_ENTRY_PRICE = 150.0
FAKE_ATR = 16.0


def _fake_historical_ohlcv(
    ticker: str,
    simulated_date: datetime,
    lookback_days: int = 30,
) -> OHLCVSeries:
    """
    Return deterministic historical bars for the backtest fixture.

    The real backtest runner discovers its session calendar from the OHLCV
    bars returned here. Therefore the fixture must provide all four sessions
    used by the integration test rather than returning only the current
    simulated date.

    For a simulated date of 2026-08-04 this returns:

        2026-08-01
        2026-08-02
        2026-08-03
        2026-08-04

    All bars use the same price so portfolio equity remains deterministic.
    """

    del lookback_days

    base_date = simulated_date.replace(
        hour=0,
        minute=0,
        second=0,
        microsecond=0,
    )

    bars: list[OHLCVBar] = []

    for offset in range(4):
        timestamp = base_date - timedelta(days=3 - offset)

        bars.append(
            OHLCVBar(
                timestamp=timestamp,
                open=FAKE_ENTRY_PRICE,
                high=FAKE_ENTRY_PRICE + 1,
                low=FAKE_ENTRY_PRICE - 1,
                close=FAKE_ENTRY_PRICE,
                volume=1_000_000,
            )
        )

    return OHLCVSeries(
        ticker=ticker,
        bars=bars,
        source="test_fixture",
        mode=DataSourceMode.BACKTEST,
        as_of=simulated_date,
    )


def _fake_bullish_signal(ticker: str) -> Signal:
    return Signal(
        direction="bullish",
        confidence=0.8,
        rationale="test fixture",
    )


def test_lookahead_guard_blocks_missing_simulated_date(monkeypatch):
    """
    Regression test for the Phase 9 lookahead bug.

    Backtest mode must not silently fall back to the real wall-clock
    datetime when no simulated_date is supplied.
    """

    monkeypatch.setenv("TRADING_MODE", "backtest")
    set_simulated_date(None)

    from data_sources import fetch_ohlcv

    try:
        raised = False

        try:
            fetch_ohlcv("AAPL", lookback_days=5)
        except RuntimeError:
            raised = True

        assert raised, "fetch_ohlcv must raise, not silently use real wall-clock time"

    finally:
        set_simulated_date(None)


def test_backtest_runner_executes_then_clamps_on_ticker_cap(
    tmp_path,
    monkeypatch,
):
    """
    Verify the complete multi-session backtest and RiskAgent cap behavior.

    With:

        starting equity = $100,000
        entry price     = $150
        ATR             = 16
        risk fraction   = 1%

    the raw proposed position is approximately:

        ($100,000 * 0.01) / (16 * 1.5)
        = 41.6667 shares
        = $6,250

    Three fills therefore reach approximately 18.75% of equity.

    The fourth fill would exceed the 20% ticker cap, so RiskAgent must
    clamp it to the remaining approximately $1,250 of room:

        $1,250 / $150
        = 8.3333 shares

    The test also verifies that the backtest restores the original graph
    singletons and clears the simulated date afterward.
    """

    monkeypatch.setenv("TRADING_MODE", "backtest")

    # Patch the underlying historical implementation. The data_sources
    # facade dynamically resolves historical.fetch_ohlcv(), so this patch
    # is shared by all callers in the real pipeline.
    monkeypatch.setattr(
        historical_module,
        "fetch_ohlcv",
        _fake_historical_ohlcv,
    )

    # Keep the integration test deterministic and independent of external
    # LLM/news/indicator behavior.
    monkeypatch.setattr(
        nodes_module,
        "run_news_agent",
        _fake_bullish_signal,
    )

    monkeypatch.setattr(
        nodes_module,
        "run_chart_agent",
        _fake_bullish_signal,
    )

    monkeypatch.setattr(
        nodes_module,
        "compute_atr",
        lambda bars: FAKE_ATR,
    )

    # Save the real singletons so run_backtest() can be verified to restore
    # them after the historical replay.
    saved_ledger = nodes_module._ledger_singleton
    saved_broker = nodes_module._broker_singleton
    saved_config = nodes_module._risk_config_singleton

    from backtest.runner import run_backtest

    result = run_backtest(
        tickers=["AAPL"],
        start_date="2026-08-01",
        end_date="2026-08-04",
        starting_equity=100_000.0,
        ledger_path=str(tmp_path / "test_backtest_ledger.json"),
        tick_delay_seconds=0,
    )

    # ---------------------------------------------------------------
    # Session discovery
    # ---------------------------------------------------------------

    assert len(result.equity_curve) == 4

    dates = [date for date, _equity in result.equity_curve]

    assert dates == [
        "2026-08-01",
        "2026-08-02",
        "2026-08-03",
        "2026-08-04",
    ]

    # ---------------------------------------------------------------
    # Execution outcomes
    # ---------------------------------------------------------------

    outcomes = {entry["date"]: entry["outcome"] for entry in result.tick_log}

    assert outcomes["2026-08-01"] == "executed"
    assert outcomes["2026-08-02"] == "executed"
    assert outcomes["2026-08-03"] == "executed"

    # Three fills at approximately 6.25% each produce approximately
    # 18.75% exposure. The fourth raw order would exceed the 20% cap,
    # so RiskAgent must CLAMP rather than reject the order.
    assert outcomes["2026-08-04"] == "executed"

    # ---------------------------------------------------------------
    # Fourth-order clamp
    # ---------------------------------------------------------------

    final_day_notes = result.tick_log[-1]["notes"]

    assert "8.333" in final_day_notes

    # All four days should have produced an actual broker fill.
    assert len(result.trades) == 4

    fourth_trade = result.trades[-1]

    assert fourth_trade["shares"] == pytest.approx(
        1250.0 / FAKE_ENTRY_PRICE,
        rel=1e-3,
    )

    # ---------------------------------------------------------------
    # Equity
    # ---------------------------------------------------------------

    final_date, final_equity = result.equity_curve[-1]

    assert final_date == "2026-08-04"

    # Fixed entry/mark price means there should be essentially no P&L drift.
    assert abs(final_equity - 100_000.0) < 1.0

    # ---------------------------------------------------------------
    # Backtest state cleanup
    # ---------------------------------------------------------------

    assert nodes_module._ledger_singleton is saved_ledger
    assert nodes_module._broker_singleton is saved_broker
    assert nodes_module._risk_config_singleton is saved_config

    # The simulated-date facade state must also be cleared so a subsequent
    # run cannot inherit stale historical state.
    assert get_simulated_date() is None
