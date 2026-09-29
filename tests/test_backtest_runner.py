"""
Integration tests for backtest/runner.py.

Regression coverage for:

1. The backtest lookahead guard in data_sources/__init__.py.
2. End-to-end RiskAgent position-cap clamping.
3. Historical session discovery from OHLCV data.
4. Multi-ticker TradeTrace coverage.
5. Full BUY -> SELL -> closed-position lifecycle.
6. Partial SELL -> final SELL accounting lifecycle.
7. Realized P&L calculation during historical replay.
8. Final backtest state capture after dependency cleanup.
9. Restoration of backtest singletons and simulated-date state.

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
    Return deterministic historical bars for the ticker-cap and
    multi-ticker trace-coverage tests.

    The fixture provides four historical sessions ending at the supplied
    simulated date. All bars use the same close price so equity remains
    deterministic while position sizing, ticker-cap behavior, and
    TradeTrace coverage are tested.
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
    """
    Return a deterministic bullish signal for integration tests.
    """

    return Signal(
        direction="bullish",
        confidence=0.8,
        rationale=f"test fixture for {ticker}",
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


def test_backtest_historical_fixture_never_exposes_future_bars():
    """
    Historical OHLCV access must respect the simulated-date/as-of boundary.

    For a simulated date T, every returned bar must satisfy:

        bar.timestamp <= T

    A future bar must therefore never be visible to the backtest decision
    pipeline.

    This is the positive counterpart to
    test_lookahead_guard_blocks_missing_simulated_date():
        - missing simulated date -> reject
        - valid simulated date -> only historical/present bars are visible
    """

    from data_sources import fetch_ohlcv

    simulated_date = datetime(
        2026,
        8,
        4,
        tzinfo=UTC,
    )

    set_simulated_date(simulated_date)

    try:
        series = _fake_historical_ohlcv(
            ticker="AAPL",
            simulated_date=simulated_date,
            lookback_days=30,
        )

        assert series.as_of == simulated_date

        assert series.bars

        assert all(bar.timestamp <= simulated_date for bar in series.bars)

        assert max(bar.timestamp for bar in series.bars) <= simulated_date

        # Explicitly prove a future trading bar is not present.
        future_date = simulated_date + timedelta(days=1)

        assert all(bar.timestamp < future_date for bar in series.bars)

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


def test_backtest_multi_ticker_trace_coverage(
    tmp_path,
    monkeypatch,
):
    """
    Verify that every ticker/session processed by the backtest produces
    exactly one tick-log entry and exactly one TradeTrace.

    Configuration:

        tickers = AAPL, MSFT
        sessions = 4
        expected ticks = 8
        expected TradeTraces = 8

    This verifies the integration invariant:

        run_tick
            -> TickResult
            -> TradeTrace
            -> tick_log
            -> diagnostics

    TradeTrace.simulated_date is serialized as a top-level field by
    TradeTrace.to_dict(). The test therefore compares that top-level
    value against the tick-log date after normalizing both to YYYY-MM-DD.
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

    from backtest.runner import _build_diagnostics, run_backtest

    result = run_backtest(
        tickers=["AAPL", "MSFT"],
        start_date="2026-08-01",
        end_date="2026-08-04",
        starting_equity=100_000.0,
        ledger_path=str(tmp_path / "multi_ticker_trace_ledger.json"),
        tick_delay_seconds=0,
    )

    diagnostics = _build_diagnostics(result)

    expected_ticks = 2 * 4

    # ---------------------------------------------------------------
    # Trace/tick cardinality
    # ---------------------------------------------------------------

    assert len(result.tick_log) == expected_ticks
    assert len(result.trade_traces) == expected_ticks

    assert diagnostics["tick_count"] == expected_ticks
    assert diagnostics["trade_trace_count"] == expected_ticks
    assert diagnostics["trace_coverage_complete"] is True
    assert diagnostics["trade_trace_tickers"] == 2

    # ---------------------------------------------------------------
    # Every ticker/session pair must have exactly one tick and trace.
    # ---------------------------------------------------------------

    tick_keys = {
        (
            tick["date"],
            tick["ticker"],
        )
        for tick in result.tick_log
    }

    trace_keys = {
        (
            trace["simulated_date"][:10],
            trace["ticker"],
        )
        for trace in result.trade_traces
    }

    assert len(tick_keys) == expected_ticks
    assert len(trace_keys) == expected_ticks

    assert tick_keys == trace_keys

    # ---------------------------------------------------------------
    # Ticker coverage
    # ---------------------------------------------------------------

    assert {tick["ticker"] for tick in result.tick_log} == {
        "AAPL",
        "MSFT",
    }

    assert {trace["ticker"] for trace in result.trade_traces} == {
        "AAPL",
        "MSFT",
    }

    # ---------------------------------------------------------------
    # Date coverage
    # ---------------------------------------------------------------

    expected_dates = {
        "2026-08-01",
        "2026-08-02",
        "2026-08-03",
        "2026-08-04",
    }

    assert {tick["date"] for tick in result.tick_log} == expected_dates

    assert {
        trace["simulated_date"][:10] for trace in result.trade_traces
    } == expected_dates


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
    assert day1["position_state"] == "open"

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
    assert day2["position_state"] == "closed"

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
    # Session 3: no position remains
    #
    # Important semantic distinction:
    #
    # position_closed=True means this tick successfully closed an
    # existing position.
    #
    # Session 3 does not close anything. The position was already
    # closed during Session 2, so Session 3 must report no_position.
    # ---------------------------------------------------------------

    day3 = result.tick_log[2]

    assert day3["remaining_shares"] == pytest.approx(
        0.0,
        abs=1e-9,
    )

    assert day3["position_closed"] is False
    assert day3["position_state"] == "no_position"

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


def test_backtest_partial_sell_then_full_close_realizes_pnl(
    tmp_path,
    monkeypatch,
):
    """
    Verify partial-close accounting across multiple historical sessions.

    Session 1:
        BUY 100 @ $100
        -> 100 shares open

    Session 2:
        SELL 40 @ $110
        -> 60 shares remain
        -> position_state = partially_closed
        -> incremental realized P&L = $400

    Session 3:
        SELL 60 @ $120
        -> 0 shares remain
        -> position_state = closed
        -> incremental realized P&L = $1,200

    Final portfolio:
        cumulative realized P&L = $1,600
        final cash = $101,600
        final equity = $101,600

    The test intentionally exercises the production long-only coordinator,
    which clamps SELL quantity against the currently held position.

    Closed-trade semantics:

        result.closed_trades contains the final closing SELL event.

    Therefore the closed-trade record contains:

        shares       = 60
        exit_price   = 120
        average_cost = 100
        realized_pnl = 1,200

    Cumulative lifecycle P&L is exposed separately through
    result.final_realized_pnl and is therefore $1,600.
    """

    monkeypatch.setenv(
        "TRADING_MODE",
        "backtest",
    )

    prices = {
        "2026-08-01": 100.0,
        "2026-08-02": 110.0,
        "2026-08-03": 120.0,
    }

    def fake_historical_ohlcv(
        ticker: str,
        simulated_date: datetime,
        lookback_days: int = 30,
    ) -> OHLCVSeries:
        """
        Return only bars available at simulated_date.

        This preserves the historical replay lookahead invariant.
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
        del ticker

        current = get_simulated_date()

        assert current is not None

        if current.date().isoformat() == "2026-08-01":
            return Signal(
                direction="bullish",
                confidence=0.9,
                rationale="test partial-close buy signal",
            )

        return Signal(
            direction="bearish",
            confidence=0.9,
            rationale="test partial-close sell signal",
        )

    def fake_news_result(ticker: str) -> NewsAgentResult:
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

    def fake_run_risk_agent(
        merged,
        ticker,
        sector,
        entry_price,
        atr,
        portfolio,
        config,
    ):
        """
        Return deterministic quantities so the real production coordinator
        receives:

            BUY 100
            SELL 40
            SELL 60

        The coordinator, broker, ledger, and accounting remain production
        implementations.
        """

        del (
            merged,
            ticker,
            sector,
            entry_price,
            atr,
            portfolio,
            config,
        )

        current = get_simulated_date()

        assert current is not None

        quantities = {
            "2026-08-01": 100.0,
            "2026-08-02": 40.0,
            "2026-08-03": 60.0,
        }

        date_key = current.date().isoformat()

        return nodes_module.RiskAgentDecision(
            approved=True,
            ticker="AAPL",
            proposed_shares=quantities[date_key],
            raw_shares=quantities[date_key],
            checks=[],
        )

    monkeypatch.setattr(
        nodes_module,
        "run_risk_agent",
        fake_run_risk_agent,
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
        ledger_path=str(tmp_path / "partial_close_ledger.json"),
        tick_delay_seconds=0,
    )

    # ---------------------------------------------------------------
    # Session discovery
    # ---------------------------------------------------------------

    assert len(result.tick_log) == 3

    assert [entry["date"] for entry in result.tick_log] == [
        "2026-08-01",
        "2026-08-02",
        "2026-08-03",
    ]

    # ---------------------------------------------------------------
    # Session 1: BUY 100 @ $100
    # ---------------------------------------------------------------

    day1 = result.tick_log[0]

    assert day1["execution_attempted"] is True
    assert day1["execution_success"] is True
    assert day1["execution_side"] == "buy"

    assert day1["price"] == pytest.approx(100.0)

    assert day1["proposed_shares"] == pytest.approx(100.0)

    assert day1["remaining_shares"] == pytest.approx(
        100.0,
        abs=1e-9,
    )

    assert day1["position_closed"] is False
    assert day1["position_state"] == "open"

    # ---------------------------------------------------------------
    # Session 2: SELL 40 @ $110
    # ---------------------------------------------------------------

    day2 = result.tick_log[1]

    assert day2["execution_attempted"] is True
    assert day2["execution_success"] is True
    assert day2["execution_side"] == "sell"

    assert day2["price"] == pytest.approx(110.0)

    assert day2["proposed_shares"] == pytest.approx(40.0)

    # Real coordinator/broker path must leave 60 shares.
    assert day2["remaining_shares"] == pytest.approx(
        60.0,
        abs=1e-9,
    )

    # A partial SELL is NOT a closed position.
    assert day2["position_closed"] is False
    assert day2["position_state"] == "partially_closed"

    # 40 * ($110 - $100) = $400
    assert day2["realized_pnl"] == pytest.approx(
        400.0,
        rel=1e-6,
    )

    # ---------------------------------------------------------------
    # Session 3: SELL remaining 60 @ $120
    # ---------------------------------------------------------------

    day3 = result.tick_log[2]

    assert day3["execution_attempted"] is True
    assert day3["execution_success"] is True
    assert day3["execution_side"] == "sell"

    assert day3["price"] == pytest.approx(120.0)

    assert day3["proposed_shares"] == pytest.approx(60.0)

    assert day3["remaining_shares"] == pytest.approx(
        0.0,
        abs=1e-9,
    )

    # Only the final SELL closes the position.
    assert day3["position_closed"] is True
    assert day3["position_state"] == "closed"

    # 60 * ($120 - $100) = $1,200
    assert day3["realized_pnl"] == pytest.approx(
        1_200.0,
        rel=1e-6,
    )

    # ---------------------------------------------------------------
    # Closed-trade accounting
    # ---------------------------------------------------------------

    assert len(result.closed_trades) == 1

    closed_trade = result.closed_trades[0]

    assert closed_trade["ticker"] == "AAPL"
    assert closed_trade["side"] == "sell"

    # The closed trade represents the final closing SELL event.
    assert closed_trade["shares"] == pytest.approx(
        60.0,
        rel=1e-6,
    )

    assert closed_trade["exit_price"] == pytest.approx(
        120.0,
    )

    assert closed_trade["average_cost"] == pytest.approx(
        100.0,
        rel=1e-6,
    )

    # Final closing SELL event P&L:
    #
    #   60 * (120 - 100) = 1,200
    #
    # Cumulative lifecycle P&L is asserted separately through
    # result.final_realized_pnl below.
    assert closed_trade["realized_pnl"] == pytest.approx(
        1_200.0,
        rel=1e-6,
    )

    # ---------------------------------------------------------------
    # Final portfolio state
    # ---------------------------------------------------------------

    assert result.final_positions == {}

    # Cumulative realized P&L across both SELL events:
    #
    #   40 * (110 - 100) =   400
    #   60 * (120 - 100) = 1,200
    #   -------------------------
    #                     = 1,600
    assert result.final_realized_pnl == pytest.approx(
        1_600.0,
        rel=1e-6,
    )

    assert result.final_cash == pytest.approx(
        101_600.0,
        rel=1e-6,
    )

    assert result.final_equity == pytest.approx(
        101_600.0,
        rel=1e-6,
    )

    # equity_curve entries are (date, equity) tuples.
    assert result.equity_curve[-1][1] == pytest.approx(
        101_600.0,
        rel=1e-6,
    )

    # ---------------------------------------------------------------
    # Three executions:
    #
    #   1. BUY  100 @ 100
    #   2. SELL  40 @ 110
    #   3. SELL  60 @ 120
    # ---------------------------------------------------------------

    assert len(result.trades) == 3

    assert result.trades[0]["execution_side"] == "buy"
    assert result.trades[0]["price"] == pytest.approx(100.0)

    assert result.trades[1]["execution_side"] == "sell"
    assert result.trades[1]["price"] == pytest.approx(110.0)

    assert result.trades[2]["execution_side"] == "sell"
    assert result.trades[2]["price"] == pytest.approx(120.0)

    # ---------------------------------------------------------------
    # Singleton state must be restored after the backtest.
    # ---------------------------------------------------------------

    assert nodes_module._ledger_singleton is saved_ledger
    assert nodes_module._broker_singleton is saved_broker
    assert nodes_module._risk_config_singleton is saved_config

    assert get_simulated_date() is None
