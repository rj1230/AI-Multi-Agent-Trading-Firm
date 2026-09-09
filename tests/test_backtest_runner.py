"""
Integration tests for backtest/runner.py.

Two things worth locking in as regression tests, both tied to real bugs
found and fixed while building this: (1) the lookahead-guard fix in
data_sources/__init__.py (silently falling back to real wall-clock
datetime.now() in backtest mode was a genuine leak), and (2) the actual
end-to-end behavior confirmed manually -- RiskAgent's per-ticker cap
correctly allowing several same-direction adds before rejecting once the
cumulative position crosses 20% of equity.

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
already mocked) => proposed_shares = (100_000 * 0.01) / (16 * 1.5) =
41.6667 shares/trade = ~$6,250/trade = 6.25% of equity per fill.
Three same-direction fills stack to 18.75% (under the 20% per-ticker
cap); a fourth would total 25% -- BREACH, rejected. Fixed price also
means equity never drifts from P&L, keeping every day's cap percentage
exactly predictable.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

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


def test_backtest_runner_executes_then_blocks_on_ticker_cap(tmp_path, monkeypatch):
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
    # Three fills at ~6.25% each = ~18.75%; a fourth would total ~25%,
    # breaching the 20% per-ticker cap -- rejected locally, before ever
    # reaching PortfolioRiskCoordinator.
    assert outcomes["2026-08-04"] == "held_local_reject"
    assert "BREACH" in result.tick_log[-1]["notes"]

    assert len(result.trades) == 3

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
