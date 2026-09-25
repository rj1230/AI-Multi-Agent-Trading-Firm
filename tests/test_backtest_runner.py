"""
Integration tests for backtest/runner.py.

Regression coverage for:

1. The backtest lookahead guard in data_sources/__init__.py.
2. End-to-end RiskAgent position-cap clamping.
3. Historical session discovery from OHLCV data.
4. Full BUY -> SELL -> closed-position lifecycle.
5. Realized P&L calculation during historical replay.
6. Final backtest state capture after dependency cleanup.
7. Restoration of backtest singletons and simulated-date state.

The historical-data fixtures intentionally respect the simulated-date/as-of
boundary. This is critical for preventing lookahead bias and for ensuring
broker fills occur at the correct historical price.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

import data_sources.historical as historical_module
import graph.nodes as nodes_module
from agents.news_agent import NewsAgentResult, NewsAvailability
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
    Return deterministic historical bars for the ticker-cap test.

    The fixture provides four historical sessions ending at the supplied
    simulated date. All bars use the same close price so equity remains
    deterministic while position sizing and ticker-cap behavior are tested.
    """

    del ticker
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
        ticker="AAPL",
        bars=bars,
        source="test_fixture",
        mode=DataSourceMode.BACKTEST,
        as_of=simulated_date,
    )


def _fake_bullish_signal(ticker: str) -> Signal:
    """
    Return a deterministic bullish signal for the ticker-cap test.
    """

    del ticker

    return Signal(
        direction="bullish",
        confidence=0.8,
        rationale="test fixture",
    )


def _fake_bullish_news_result(ticker: str) -> NewsAgentResult:
    """
    Return deterministic metadata-aware bullish news output.

    This mirrors the production NewsAgent API used by graph.nodes.
    """

    current = get_simulated_date()

    assert current is not None

    return NewsAgentResult(
        signal=_fake_bullish_signal(ticker),
        availability=NewsAvailability.AVAILABLE,
        article_count=1,
        source="test_fixture",
        as_of=current,
    )


def test_lookahead_guard_blocks_missing_simulated_date(monkeypatch):
    """
    Backtest mode must reject OHLCV access when simulated_date is missing.

    The historical data layer must never silently substitute the real
    wall-clock time during a backtest because doing so could introduce
    future information into historical decisions.
    """

    monkeypatch.setenv(
        "TRADING_MODE",
        "backtest",
    )

    set_simulated_date(None)

    from data_sources import fetch_ohlcv

    try:
        with pytest.raises(RuntimeError):
            fetch_ohlcv(
                "AAPL",
                lookback_days=5,
            )
    finally:
        set_simulated_date(None)


def test_backtest_runner_executes_then_clamps_on_ticker_cap(
    tmp_path,
    monkeypatch,
):
    """
    Verify historical execution and RiskAgent ticker-cap clamping.

    Configuration:

        starting equity = $100,000
        entry price     = $150
        ATR             = 16
        risk fraction   = 1%
        stop multiple   = 1.5

    Raw risk-based sizing:

        ($100,000 * 0.01) / (16 * 1.5)
        = 41.6667 shares

    Position value:

        41.6667 * $150
        = $6,250

    Three executions therefore reach approximately 18.75% of equity.

    The fourth execution would exceed the 20% ticker cap. RiskAgent must
    therefore clamp the fourth order to the remaining approximately
    $1,250 of ticker-cap capacity:

        $1,250 / $150
        = 8.3333 shares

    The test also verifies that the backtest restores all process-level
    dependencies and clears simulated-date state after completion.
    """

    monkeypatch.setenv(
        "TRADING_MODE",
        "backtest",
    )

    monkeypatch.setattr(
        historical_module,
        "fetch_ohlcv",
        _fake_historical_ohlcv,
    )

    monkeypatch.setattr(
        nodes_module,
        "run_news_agent_with_metadata",
        _fake_bullish_news_result,
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
    # Historical session discovery
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
    assert outcomes["2026-08-04"] == "executed"

    # ---------------------------------------------------------------
    # Fourth-order ticker-cap clamp
    # ---------------------------------------------------------------

    final_day = result.tick_log[-1]

    assert "8.333" in final_day["notes"]

    assert len(result.trades) == 4

    fourth_trade = result.trades[-1]

    expected_fourth_shares = 1250.0 / FAKE_ENTRY_PRICE

    assert fourth_trade["shares"] == pytest.approx(
        expected_fourth_shares,
        rel=1e-3,
    )

    # ---------------------------------------------------------------
    # Equity remains approximately unchanged because all four
    # executions occur at the same historical price.
    # ---------------------------------------------------------------

    final_date, final_equity = result.equity_curve[-1]

    assert final_date == "2026-08-04"

    assert final_equity == pytest.approx(
        100_000.0,
        abs=1.0,
    )

    # ---------------------------------------------------------------
    # Final state and cleanup
    # ---------------------------------------------------------------

    assert nodes_module._ledger_singleton is saved_ledger
    assert nodes_module._broker_singleton is saved_broker
    assert nodes_module._risk_config_singleton is saved_config

    assert get_simulated_date() is None


def test_backtest_buy_then_sell_closes_position_and_realizes_pnl(
    tmp_path,
    monkeypatch,
):
    """
    Verify the complete long-position lifecycle.

    Session 1:
        bullish -> BUY -> position opens at $100

    Session 2:
        bearish -> SELL -> position closes at $110
        -> realized P&L is recorded

    Session 3:
        no AAPL position remains

    The historical fixture exposes only bars whose timestamps are less
    than or equal to simulated_date. This prevents the broker, agents, or
    data layer from seeing future prices during historical replay.
    """

    monkeypatch.setenv(
        "TRADING_MODE",
        "backtest",
    )

    prices = {
        "2026-08-01": 100.0,
        "2026-08-02": 110.0,
        "2026-08-03": 110.0,
    }

    def fake_historical_ohlcv(
        ticker: str,
        simulated_date: datetime,
        lookback_days: int = 30,
    ) -> OHLCVSeries:
        """
        Return only historical bars available at simulated_date.

        Production invariant:

            bar.timestamp <= simulated_date
        """

        del lookback_days

        bars: list[OHLCVBar] = []

        for date_key, price in prices.items():
            timestamp = datetime.fromisoformat(date_key).replace(
                tzinfo=UTC,
            )

            if timestamp > simulated_date:
                continue

            bars.append(
                OHLCVBar(
                    timestamp=timestamp,
                    open=price,
                    high=price + 1.0,
                    low=price - 1.0,
                    close=price,
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

    monkeypatch.setattr(
        historical_module,
        "fetch_ohlcv",
        fake_historical_ohlcv,
    )

    def fake_signal(ticker: str) -> Signal:
        """
        Return the deterministic signal required for lifecycle testing.

        Session 1 is bullish so the strategy opens a long position.

        Sessions 2 and 3 are bearish so the long-only execution path
        is exercised and the existing position must be closed on Session 2.
        """

        del ticker

        current = get_simulated_date()

        assert current is not None

        if current.date().isoformat() == "2026-08-01":
            return Signal(
                direction="bullish",
                confidence=0.9,
                rationale="test buy signal",
            )

        return Signal(
            direction="bearish",
            confidence=0.9,
            rationale="test exit signal",
        )

    def fake_news_result(ticker: str) -> NewsAgentResult:
        """
        Wrap the deterministic lifecycle signal in the production
        metadata-aware NewsAgentResult contract.
        """

        current = get_simulated_date()

        assert current is not None

        return NewsAgentResult(
            signal=fake_signal(ticker),
            availability=NewsAvailability.AVAILABLE,
            article_count=1,
            source="test_fixture",
            as_of=current,
        )

    monkeypatch.setattr(
        nodes_module,
        "run_news_agent_with_metadata",
        fake_news_result,
    )

    monkeypatch.setattr(
        nodes_module,
        "run_chart_agent",
        fake_signal,
    )

    monkeypatch.setattr(
        nodes_module,
        "compute_atr",
        lambda bars: 10.0,
    )

    saved_ledger = nodes_module._ledger_singleton
    saved_broker = nodes_module._broker_singleton
    saved_config = nodes_module._risk_config_singleton

    from backtest.runner import run_backtest

    result = run_backtest(
        tickers=["AAPL"],
        start_date="2026-08-01",
        end_date="2026-08-03",
        starting_equity=100_000.0,
        ledger_path=str(tmp_path / "lifecycle_ledger.json"),
        tick_delay_seconds=0,
    )

    # ---------------------------------------------------------------
    # Session discovery
    # ---------------------------------------------------------------

    assert len(result.tick_log) == 3

    dates = [entry["date"] for entry in result.tick_log]

    assert dates == [
        "2026-08-01",
        "2026-08-02",
        "2026-08-03",
    ]

    # ---------------------------------------------------------------
    # Session 1: BUY
    # ---------------------------------------------------------------

    day1 = result.tick_log[0]

    assert day1["signal_direction"] == "bullish"

    assert day1["execution_attempted"] is True
    assert day1["execution_success"] is True
    assert day1["execution_side"] == "buy"

    assert day1["price"] == pytest.approx(100.0)

    assert day1["remaining_shares"] > 0

    assert day1["position_closed"] is False

    bought_shares = day1["remaining_shares"]

    assert len(result.trades) >= 1

    first_trade = result.trades[0]

    assert first_trade["ticker"] == "AAPL"
    assert first_trade["execution_side"] == "buy"
    assert first_trade["price"] == pytest.approx(100.0)

    # ---------------------------------------------------------------
    # Session 2: SELL
    # ---------------------------------------------------------------

    day2 = result.tick_log[1]

    assert day2["signal_direction"] == "bearish"

    assert day2["execution_attempted"] is True
    assert day2["execution_success"] is True
    assert day2["execution_side"] == "sell"

    assert day2["price"] == pytest.approx(110.0)

    assert day2["remaining_shares"] == pytest.approx(
        0.0,
        abs=1e-9,
    )

    assert day2["position_closed"] is True

    # ---------------------------------------------------------------
    # Realized P&L
    # ---------------------------------------------------------------

    expected_pnl = bought_shares * (110.0 - 100.0)

    assert day2["realized_pnl"] == pytest.approx(
        expected_pnl,
        rel=1e-6,
    )

    assert len(result.closed_trades) == 1

    closed_trade = result.closed_trades[0]

    assert closed_trade["ticker"] == "AAPL"
    assert closed_trade["side"] == "sell"

    assert closed_trade["shares"] == pytest.approx(
        bought_shares,
        rel=1e-6,
    )

    assert closed_trade["exit_price"] == pytest.approx(110.0)

    assert closed_trade["average_cost"] == pytest.approx(
        100.0,
        rel=1e-6,
    )

    assert closed_trade["realized_pnl"] == pytest.approx(
        expected_pnl,
        rel=1e-6,
    )

    # ---------------------------------------------------------------
    # Session 3: position remains closed
    # ---------------------------------------------------------------

    day3 = result.tick_log[2]

    assert day3["remaining_shares"] == pytest.approx(
        0.0,
        abs=1e-9,
    )

    assert day3["position_closed"] is True

    assert len(result.trades) == 2

    # ---------------------------------------------------------------
    # Final backtest state
    # ---------------------------------------------------------------

    expected_final_equity = 100_000.0 + expected_pnl

    assert result.final_positions == {}

    assert result.final_equity == pytest.approx(
        expected_final_equity,
        rel=1e-6,
    )

    assert result.final_cash == pytest.approx(
        expected_final_equity,
        rel=1e-6,
    )

    assert result.final_realized_pnl == pytest.approx(
        expected_pnl,
        rel=1e-6,
    )

    assert result.equity_curve[-1][1] == pytest.approx(
        result.final_equity,
        rel=1e-6,
    )

    # ---------------------------------------------------------------
    # Backtest dependency cleanup
    # ---------------------------------------------------------------

    assert nodes_module._ledger_singleton is saved_ledger
    assert nodes_module._broker_singleton is saved_broker
    assert nodes_module._risk_config_singleton is saved_config

    assert get_simulated_date() is None
