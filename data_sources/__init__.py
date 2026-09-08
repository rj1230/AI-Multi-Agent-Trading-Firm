"""
Dual-mode data source facade (Phase 4.1 pulled forward from v2).

Agent code should import THIS module, not data_sources.live or
data_sources.historical directly:

    from data_sources import fetch_news, fetch_ohlcv

A single env var, TRADING_MODE ("live" | "backtest"), decides which
underlying module actually runs. This is what makes the doc's Phase 4.1
self-check trivially true: swap the env var, and zero lines of agent code
change, because agent code never branches on mode itself.
"""

from __future__ import annotations

import os
from datetime import datetime, timezone

from data_sources import historical, live
from data_sources.schemas import NewsResult, OHLCVSeries


def get_mode() -> str:
    return os.getenv("TRADING_MODE", "live").lower()


def fetch_ohlcv(
    ticker: str, simulated_date: datetime | None = None, lookback_days: int = 30
) -> OHLCVSeries:
    if get_mode() == "backtest":
        sim_date = simulated_date or datetime.now(timezone.utc)
        return historical.fetch_ohlcv(ticker, sim_date, lookback_days)
    return live.fetch_ohlcv(ticker, simulated_date, lookback_days)


def fetch_news(
    ticker: str, simulated_date: datetime | None = None, page_size: int = 10
) -> NewsResult:
    if get_mode() == "backtest":
        sim_date = simulated_date or datetime.now(timezone.utc)
        return historical.fetch_news(ticker, sim_date)
    return live.fetch_news(ticker, simulated_date, page_size)
