"""
Dual-mode data source facade (Phase 4.1 pulled forward from v2).

Agent code should import THIS module, not data_sources.live or
data_sources.historical directly:

    from data_sources import fetch_news, fetch_ohlcv

A single env var, TRADING_MODE ("live" | "backtest"), decides which
underlying module actually runs.

Phase 9 addition: set_simulated_date()/get_simulated_date(). No agent
code (chart_agent.py, news_agent.py, graph/nodes.py) passes an explicit
simulated_date -- they all call fetch_ohlcv(ticker, lookback_days=X) the
same way in both modes, per Phase 4.1's whole point. Without this, a
None simulated_date in backtest mode silently fell back to
datetime.now(timezone.utc) -- REAL wall-clock now, a lookahead violation
invisible to any test that calls historical.py directly with an explicit
date instead of going through this facade the way the real graph does.
backtest/runner.py calls set_simulated_date() once per simulated tick,
before invoking run_tick() -- same singleton-override pattern
graph/nodes.py already uses for _ledger/_broker/_risk_config.
"""

from __future__ import annotations

import os
from datetime import datetime, timezone

from data_sources import historical, live
from data_sources.schemas import NewsResult, OHLCVSeries

_simulated_date: datetime | None = None


def set_simulated_date(dt: datetime | None) -> None:
    global _simulated_date
    _simulated_date = dt


def get_simulated_date() -> datetime | None:
    return _simulated_date


def get_mode() -> str:
    return os.getenv("TRADING_MODE", "live").lower()


def fetch_ohlcv(
    ticker: str, simulated_date: datetime | None = None, lookback_days: int = 30
) -> OHLCVSeries:
    if get_mode() == "backtest":
        sim_date = simulated_date or _simulated_date
        if sim_date is None:
            raise RuntimeError(
                "backtest mode: no simulated_date provided and none set via "
                "set_simulated_date() -- refusing to silently fall back to "
                "real wall-clock time (lookahead risk)."
            )
        return historical.fetch_ohlcv(ticker, sim_date, lookback_days)
    return live.fetch_ohlcv(ticker, simulated_date, lookback_days)


def fetch_news(
    ticker: str, simulated_date: datetime | None = None, page_size: int = 10
) -> NewsResult:
    if get_mode() == "backtest":
        sim_date = simulated_date or _simulated_date
        if sim_date is None:
            raise RuntimeError(
                "backtest mode: no simulated_date provided and none set via "
                "set_simulated_date() -- refusing to silently fall back to "
                "real wall-clock time (lookahead risk)."
            )
        return historical.fetch_news(ticker, sim_date)
    return live.fetch_news(ticker, simulated_date, page_size)
