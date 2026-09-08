"""
Maintains/updates the rolling 30-day return correlation matrix across held +
candidate positions. Flags or downsizes new positions correlated > 0.7 with
an existing holding.
"""
from __future__ import annotations

import pandas as pd

from data_sources.schemas import OHLCVSeries


def returns_from_ohlcv(series: OHLCVSeries) -> pd.Series:
    """Daily pct-change returns for one ticker, indexed by bar timestamp.
    Works identically on live or historical-replay data since both produce
    the same OHLCVSeries shape (see data_sources/schemas.py)."""
    if series.is_empty:
        return pd.Series(name=series.ticker, dtype=float)
    closes = pd.Series(
        [bar.close for bar in series.bars],
        index=[bar.timestamp for bar in series.bars],
        name=series.ticker,
    )
    return closes.pct_change().dropna()


def update_correlation_matrix(returns_by_ticker: dict[str, pd.Series]) -> pd.DataFrame:
    """Builds a ticker x ticker correlation matrix from each ticker's
    return series. min_periods=5 means a pair with fewer than 5 overlapping
    trading days correlates to NaN rather than a misleadingly confident
    number from 2-3 data points."""
    if not returns_by_ticker:
        return pd.DataFrame()
    df = pd.DataFrame(returns_by_ticker)
    return df.corr(min_periods=5)


def check_correlation(
    candidate_ticker: str, held_tickers: list[str], matrix: pd.DataFrame,
    max_correlation: float = 0.7,
) -> dict:
    """Returns {'flagged': bool, 'max_correlation': float, 'against': str | None}."""
    if matrix is None or matrix.empty or candidate_ticker not in matrix.columns:
        # No correlation data yet (e.g. brand-new ticker, insufficient
        # history) -- fail open with flagged=False rather than blocking
        # every trade just because the matrix hasn't caught up.
        return {"flagged": False, "max_correlation": 0.0, "against": None}

    best_ticker: str | None = None
    best_corr = 0.0
    for ticker in held_tickers:
        if ticker == candidate_ticker or ticker not in matrix.columns:
            continue
        corr = matrix.loc[candidate_ticker, ticker]
        if pd.isna(corr):
            continue
        if abs(corr) > abs(best_corr):
            best_corr = float(corr)
            best_ticker = ticker

    flagged = abs(best_corr) > max_correlation
    return {
        "flagged": flagged,
        "max_correlation": round(best_corr, 4),
        "against": best_ticker if flagged else None,
    }
