"""
Portfolio state shapes used by RiskAgent, plus the function that closes
the gap between RiskAgent's tests (fixture correlation data) and RiskAgent
running for real inside the graph: build_correlation_matrix() fetches
actual OHLCV through the same dual-mode data_sources facade every other
agent uses, so LIVE and BACKTEST feed RiskAgent identically-shaped data,
same as everywhere else in the system (see data_sources/schemas.py docstring).
"""
from __future__ import annotations

from typing import Optional

import pandas as pd
from pydantic import BaseModel, ConfigDict

from data_sources import fetch_ohlcv
from portfolio.correlation import returns_from_ohlcv, update_correlation_matrix


class Position(BaseModel):
    ticker: str
    shares: float
    sector: str
    market_value: float


class PortfolioSnapshot(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)  # for the DataFrame field

    equity: float
    starting_equity: float  # equity as of session/day open -- circuit breaker baseline
    positions: dict[str, Position] = {}
    correlation_matrix: Optional[pd.DataFrame] = None


def build_correlation_matrix(tickers: list[str], lookback_days: int = 30) -> pd.DataFrame:
    """
    Fetches OHLCV for every ticker (held + any new candidate) and returns
    the rolling correlation matrix RiskAgent's correlation check reads
    from. Call this once per tick -- before running RiskAgent for any
    individual ticker -- not once per ticker, since it needs the full set
    to build a meaningful matrix.
    """
    returns_by_ticker: dict[str, pd.Series] = {}
    for ticker in tickers:
        series = fetch_ohlcv(ticker, lookback_days=lookback_days)
        if series.is_empty:
            continue
        returns_by_ticker[ticker] = returns_from_ohlcv(series)
    return update_correlation_matrix(returns_by_ticker)
