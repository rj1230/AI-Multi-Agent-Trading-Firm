"""
Integration tests for backtest/runner.py.

Two things worth locking in as regression tests, both tied to real bugs
found and fixed while building this: (1) the lookahead-guard fix in
data_sources/__init__.py (silently falling back to real wall-clock
datetime.now() in backtest mode was a genuine leak), and (2) the actual
end-to-end behavior confirmed manually -- RiskAgent's per-ticker cap
correctly allowing several same-direction adds, then CLAMPING (not
rejecting) once the cumulative position would otherwise cross 20% of
equity.

CHANGE: this used to assert the 4th same-direction add was rejected
outright ("held_local_reject") once the raw risk-based size would have
pushed the position past the 20% cap. That was the old reject-on-breach
RiskAgent behavior. RiskAgent now clamps the trade size down to whatever
room remains under the cap instead of rejecting the whole trade -- see
agents/risk_agent.py's compute_capped_size(). With $1,250 of room left
after three $6,250 fills, the 4th day now executes at the clamped size
(~8.33 sh, ~$1,250) instead of being held.

Patch strategy: nodes.run_news_agent/run_chart_agent are mocked directly
(bypassing their real Groq/indicator-math internals entirely -- those are
already covered by test_new_agent.py/test_chart_agent.py). The one real
data dependency left is OHLCV price/ATR lookups, which flow through many
different call sites (risk_agent_node directly, PortfolioLedger's
mark-to-market and correlation build, tick_runner's own correlation
rebuild) -- rather than patching each site separately, this patches
data_sources.historical.fetch_ohlcv itself, since the facade always
resolves that attribute at call time regardless of which module's bound
`fetch_ohlcv` name triggered it.

Scenario: fixed entry_price=150.0, fixed ATR=16.0 (via a compute_atr
patch, since bar-crafting isn't needed once historical.fetch_ohlcv is
already mocked) => raw proposed_shares = (100_000 * 0.01) / (16 * 1.5) =
41.6667 shares/trade = ~$6,250/trade = 6.25% of equity per fill.
Three same-direction fills stack to 18.75% (under the 20% per-ticker
cap), leaving $1,250 of room. A fourth trade's raw size would total 25%,
which now gets CLAMPED down to the $1,250 of remaining room (~8.33 sh)
rather than rejected -- landing the position at exactly the 20% cap.
Fixed price also means equity never drifts from P&L, keeping every day's
cap percentage exactly predictable.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

import graph.nodes as nodes_module
import data_sources.historical as historical_module
from data_sources import get_simulated_date, set_simulated_date
from data_sources.schemas import DataSourceMode, OHLCVBar, OHLCVSeries
from graph.state import Signal

FAKE_ENTRY_PRICE = 150.0
FAKE_ATR = 16.0


def _fake_historical_ohlcv(
    ticker: str, simulated_date, lookback_days: int = 30
) -> OHLCVSeries:
    """Ignores lookback_days/ticker specifics -- always returns one bar
    at the fixed price, timestamped exactly at simulated_date. Real
    lookahead protection (facade raising if simulated_date is None) is
    tested separately below; this fixture exists to make the *sizing*
    math deterministic, not to re-test the date-slicing logic itself."""
    bar = OHLCVBar(
        timestamp=simulated_date,
        open=FAKE_ENTRY_PRICE,
        high=FAKE_ENTRY_PRICE + 1,
        low=FAKE_ENTRY_PRICE - 1,
        close=FAKE_ENTRY_PRICE,
        volume=1_000_000,
    )
    return OHLCVSeries(
        ticker=ticker,
        bars=[bar],
        source="test_fixture",
        mode=DataSourceMode.BACKTEST,
        as_of=simulated_date,
    )


def _fake_bullish_signal(ticker: str) -> Signal:
    return Signal(direction="bullish", confidence=0.8, rationale="test fixture")


def test_lookahead_guard_blocks_missing_simulated_date(monkeypatch):
    """Regression test for the real bug found while building Phase 9:
    backtest mode used to silently fall back to real wall-clock
    datetime.now() when no simulated_date was passed -- which is what
    every real caller (chart_agent, news_agent, risk_agent_node) does.
    Must now raise instead of leak future data."""
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


def test_backtest_runner_executes_then_clamps_on_ticker_cap(tmp_path, monkeypatch):
    monkeypatch.setenv("TRADING_MODE", "backtest")
    monkeypatch.setattr(historical_module, "fetch_ohlcv", _fake_historical_ohlcv)
    monkeypatch.setattr(nodes_module, "run_news_agent", _fake_bullish_signal)
    monkeypatch.setattr(nodes_module, "run_chart_agent", _fake_bullish_signal)
    monkeypatch.setattr(nodes_module, "compute_atr", lambda bars: FAKE_ATR)

    # Save real singletons so we can confirm run_backtest() restores them
    # afterward -- a backtest run must never leave graph.nodes pointed at
    # backtest state.
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
    )

    outcomes = {entry["date"]: entry["outcome"] for entry in result.tick_log}
    assert outcomes["2026-08-01"] == "executed"
    assert outcomes["2026-08-02"] == "executed"
    assert outcomes["2026-08-03"] == "executed"
    # Three fills at ~6.25% each = ~18.75%, leaving $1,250 of room under
    # the 20% per-ticker cap. The 4th trade's raw size would breach the
    # cap, but RiskAgent now CLAMPS it down to the remaining room instead
    # of rejecting the trade outright -- so it still executes, just small.
    assert outcomes["2026-08-04"] == "executed"

    # The 4th fill's broker confirmation should show the clamped size
    # (~$1,250 of remaining cap room / $150 = ~8.333 sh), not the raw
    # ~41.67 sh a clean 6.25%-of-equity trade would otherwise produce.
    final_day_notes = result.tick_log[-1]["notes"]
    assert "8.333" in final_day_notes

    # All four days produced a fill now, not just three.
    assert len(result.trades) == 4

    # The 4th fill should be clamped to ~$1,250 of remaining room
    # (~8.33 sh at the fixed $150 price), not the raw ~41.67 sh a clean
    # 6.25%-of-equity trade would otherwise be.
    #
    # NOTE: adjust the "shares" key below if result.trades entries use a
    # different field name in your actual trade-log shape.
    fourth_trade = result.trades[-1]
    assert fourth_trade["shares"] == pytest.approx(1250.0 / FAKE_ENTRY_PRICE, rel=1e-3)

    # Equity never drifts (fixed price -> no P&L), so cash-plus-position
    # value should still equal starting equity almost exactly.
    final_date, final_equity = result.equity_curve[-1]
    assert final_date == "2026-08-04"
    assert abs(final_equity - 100_000.0) < 1.0

    # Real singletons restored -- a backtest run must not leak into the
    # process's live/paper-trading state.
    assert nodes_module._ledger_singleton is saved_ledger
    assert nodes_module._broker_singleton is saved_broker
    assert nodes_module._risk_config_singleton is saved_config

    # The facade's simulated-date global is cleared after the run too --
    # otherwise a subsequent live-mode call could pick up a stale date.
    assert get_simulated_date() is None
