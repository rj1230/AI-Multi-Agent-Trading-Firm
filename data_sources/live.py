"""
Live data source layer.

Design rule from the architecture doc: this module and historical.py must
be swappable with zero changes to agent code. Every function here returns
the shared Pydantic shapes from schemas.py.

Fallback chain for OHLCV: Alpaca Market Data (free IEX tier) -> yFinance.
News has no fallback provider (NewsAPI only) but zero-article results are
returned as a valid, typed empty NewsResult rather than raised as errors.
"""

from __future__ import annotations

import logging
import os
from datetime import datetime, timedelta, timezone

import requests

from data_sources.schemas import (
    DataSourceMode,
    NewsArticle,
    NewsResult,
    OHLCVBar,
    OHLCVSeries,
)

logger = logging.getLogger(__name__)

NEWSAPI_URL = "https://newsapi.org/v2/top-headlines"


def fetch_ohlcv(
    ticker: str, simulated_date: datetime | None = None, lookback_days: int = 30
) -> OHLCVSeries:
    """Fetch daily OHLCV bars for `ticker`. Tries Alpaca first, falls back
    to yFinance on any failure or empty result. Never raises for a data
    problem — callers should check `.is_empty` instead of catching
    exceptions for the "no data" case.

    `simulated_date` is accepted (and ignored) so this function has the
    same call signature as historical.fetch_ohlcv — see
    data_sources/__init__.py, which lets agent code call
    data_sources.fetch_ohlcv(...) without knowing which mode it's in."""
    now = datetime.now(timezone.utc)

    series = _fetch_ohlcv_alpaca(ticker, lookback_days, now)
    if series is not None and not series.is_empty:
        return series

    logger.warning(
        "Alpaca OHLCV fetch failed or empty for %s, falling back to yfinance",
        ticker,
    )
    series = _fetch_ohlcv_yfinance(ticker, lookback_days, now)
    if series is not None:
        return series

    # Both sources failed — return a typed empty result rather than raising,
    # so downstream ChartAgent can decide how to handle "no data" itself.
    return OHLCVSeries(
        ticker=ticker,
        bars=[],
        source="none",
        mode=DataSourceMode.LIVE,
        as_of=now,
    )


def _fetch_ohlcv_alpaca(
    ticker: str, lookback_days: int, now: datetime
) -> OHLCVSeries | None:
    try:
        from alpaca.data.historical import StockHistoricalDataClient
        from alpaca.data.requests import StockBarsRequest
        from alpaca.data.timeframe import TimeFrame
    except ImportError:
        logger.warning("alpaca-py not installed; skipping Alpaca OHLCV source")
        return None

    api_key = os.getenv("ALPACA_API_KEY")
    secret_key = os.getenv("ALPACA_SECRET_KEY")
    if not api_key or not secret_key:
        return None

    try:
        client = StockHistoricalDataClient(api_key, secret_key)
        request = StockBarsRequest(
            symbol_or_symbols=ticker,
            timeframe=TimeFrame.Day,
            start=now - timedelta(days=lookback_days),
            end=now,
        )
        bar_set = client.get_stock_bars(request)
        raw_bars = bar_set.data.get(ticker, [])

        bars = [
            OHLCVBar(
                timestamp=b.timestamp,
                open=b.open,
                high=b.high,
                low=b.low,
                close=b.close,
                volume=b.volume,
            )
            for b in raw_bars
        ]
        return OHLCVSeries(
            ticker=ticker,
            bars=bars,
            source="alpaca",
            mode=DataSourceMode.LIVE,
            as_of=now,
        )
    except Exception as e:
        logger.warning("Alpaca OHLCV fetch error for %s: %s", ticker, e)
        return None


def _fetch_ohlcv_yfinance(
    ticker: str, lookback_days: int, now: datetime
) -> OHLCVSeries | None:
    try:
        import yfinance as yf
    except ImportError:
        logger.warning("yfinance not installed; no fallback OHLCV source available")
        return None

    try:
        df = yf.Ticker(ticker).history(period=f"{lookback_days}d", interval="1d")
        if df.empty:
            return OHLCVSeries(
                ticker=ticker,
                bars=[],
                source="yfinance",
                mode=DataSourceMode.LIVE,
                as_of=now,
            )

        bars = [
            OHLCVBar(
                timestamp=idx.to_pydatetime(),
                open=float(row["Open"]),
                high=float(row["High"]),
                low=float(row["Low"]),
                close=float(row["Close"]),
                volume=int(row["Volume"]),
            )
            for idx, row in df.iterrows()
        ]
        return OHLCVSeries(
            ticker=ticker,
            bars=bars,
            source="yfinance",
            mode=DataSourceMode.LIVE,
            as_of=now,
        )
    except Exception as e:
        logger.warning("yfinance OHLCV fetch error for %s: %s", ticker, e)
        return None


def fetch_news(
    ticker: str, simulated_date: datetime | None = None, page_size: int = 10
) -> NewsResult:
    """Fetch recent headlines for `ticker` via NewsAPI. Zero results is a
    valid, expected outcome — returned as an empty NewsResult, not an
    error. NewsAgent must handle `.is_empty` explicitly (see Phase 4).

    `simulated_date` is accepted (and ignored) for signature parity with
    historical.fetch_news — see data_sources/__init__.py."""
    now = datetime.now(timezone.utc)
    api_key = os.getenv("NEWSAPI_KEY")

    if not api_key:
        logger.warning("NEWSAPI_KEY not set; returning empty NewsResult for %s", ticker)
        return NewsResult(
            ticker=ticker,
            articles=[],
            source="newsapi",
            mode=DataSourceMode.LIVE,
            as_of=now,
        )

    try:
        resp = requests.get(
            NEWSAPI_URL,
            params={
                "q": ticker,
                "language": "en",
                "pageSize": page_size,
                "apiKey": api_key,
            },
            timeout=10,
        )
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        logger.warning("NewsAPI fetch error for %s: %s", ticker, e)
        return NewsResult(
            ticker=ticker,
            articles=[],
            source="newsapi",
            mode=DataSourceMode.LIVE,
            as_of=now,
        )

    articles = []
    for a in data.get("articles", []):
        try:
            articles.append(
                NewsArticle(
                    title=a["title"],
                    description=a.get("description"),
                    source=a.get("source", {}).get("name", "unknown"),
                    published_at=a["publishedAt"],
                    url=a["url"],
                )
            )
        except Exception as e:
            # One malformed article shouldn't sink the whole fetch.
            logger.warning("Skipping malformed article for %s: %s", ticker, e)
            continue

    return NewsResult(
        ticker=ticker,
        articles=articles,
        source="newsapi",
        mode=DataSourceMode.LIVE,
        as_of=now,
    )
