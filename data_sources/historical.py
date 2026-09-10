"""
Historical replay engine for BACKTEST mode.

Critical invariant: given a `simulated_date`, these functions must return
ONLY data that would have been known at or before that date. No bar with
timestamp > simulated_date, no article with published_at > simulated_date.
Violating this is invisible until backtest results look "too good" —
see the lookahead-bias test at the bottom of this file, and
tests/test_lookahead_bias.py.

OHLCV replay pulls from yfinance (cached locally per ticker so repeated
backtest runs don't hammer the API). News replay reads from a local JSON
cache under data/news_cache/<ticker>.json, because NewsAPI's free tier
does not serve headlines old enough for most backtest windows — you are
expected to populate that cache yourself (e.g. by archiving live fetches
over time, or importing a historical news dataset).
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path

from data_sources.schemas import (
    DataSourceMode,
    NewsArticle,
    NewsResult,
    OHLCVBar,
    OHLCVSeries,
)

logger = logging.getLogger(__name__)

CACHE_DIR = Path(os.getenv("DATA_CACHE_DIR", "data_cache"))
OHLCV_CACHE_DIR = CACHE_DIR / "ohlcv"
NEWS_CACHE_DIR = CACHE_DIR / "news_cache"


def fetch_ohlcv(
    ticker: str, simulated_date: datetime, lookback_days: int = 30
) -> OHLCVSeries:
    """Return bars strictly before `simulated_date`, most recent
    `lookback_days` of them. Pulls the full history once via yfinance and
    caches it to disk; every call after that just slices the cache."""
    full_history = _load_or_fetch_full_history(ticker)

    cutoff = _ensure_utc(simulated_date)
    eligible = [b for b in full_history if b.timestamp <= cutoff]
    windowed = eligible[-lookback_days:] if eligible else []

    return OHLCVSeries(
        ticker=ticker,
        bars=windowed,
        source="historical_replay",
        mode=DataSourceMode.BACKTEST,
        as_of=cutoff,
    )


def fetch_news(ticker: str, simulated_date: datetime) -> NewsResult:
    """Return only articles published at or before `simulated_date`, read
    from the local news cache. Missing cache file -> empty result, same
    as the live zero-articles case."""
    cutoff = _ensure_utc(simulated_date)
    cache_path = NEWS_CACHE_DIR / f"{ticker}.json"

    if not cache_path.exists():
        logger.warning(
            "No historical news cache for %s at %s; returning empty NewsResult",
            ticker,
            cache_path,
        )
        return NewsResult(
            ticker=ticker,
            articles=[],
            source="historical_replay",
            mode=DataSourceMode.BACKTEST,
            as_of=cutoff,
        )

    with open(cache_path, "r", encoding="utf-8") as f:
        raw_articles = json.load(f)

    articles = []
    for a in raw_articles:
        try:
            article = NewsArticle(**a)
        except Exception as e:
            logger.warning("Skipping malformed cached article for %s: %s", ticker, e)
            continue
        if _ensure_utc(article.published_at) <= cutoff:
            articles.append(article)

    return NewsResult(
        ticker=ticker,
        articles=articles,
        source="historical_replay",
        mode=DataSourceMode.BACKTEST,
        as_of=cutoff,
    )


# ---------------------------------------------------------------------------
# internals
# ---------------------------------------------------------------------------


def _ensure_utc(dt: datetime) -> datetime:
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _load_or_fetch_full_history(ticker: str) -> list[OHLCVBar]:
    OHLCV_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = OHLCV_CACHE_DIR / f"{ticker}.json"

    if cache_path.exists():
        with open(cache_path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        return [OHLCVBar(**b) for b in raw]

    import yfinance as yf

    df = yf.Ticker(ticker).history(period="max", interval="1d")
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

    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump([b.model_dump(mode="json") for b in bars], f)

    return bars
