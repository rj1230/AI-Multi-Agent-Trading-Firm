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
cache under data_cache/news_cache/<ticker>.json, because NewsAPI's free
tier does not serve headlines old enough for most backtest windows — you
are expected to populate that cache yourself (e.g. by archiving live
fetches over time, or importing a historical news dataset).

Important backtest data-quality invariant:

    Missing historical news cache
        !=
    Historical period genuinely had zero news.

The NewsResult.availability field explicitly preserves that distinction.
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
    NewsAvailability,
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
    """
    Return only articles published at or before `simulated_date`.

    The returned availability explicitly distinguishes:

        UNAVAILABLE
            No historical cache exists for this ticker.

        AVAILABLE + articles
            Historical cache exists and matching articles were found.

        AVAILABLE + empty
            Historical cache exists, but no cached article was published
            on or before the simulated date.

    No future article is ever exposed to the backtest.
    """
    cutoff = _ensure_utc(simulated_date)
    cache_path = NEWS_CACHE_DIR / f"{ticker}.json"

    # ---------------------------------------------------------------
    # No cache = historical news data is unavailable.
    # This is NOT equivalent to "there was no news."
    # ---------------------------------------------------------------
    if not cache_path.exists():
        logger.warning(
            "Historical news unavailable for %s: cache does not exist at %s",
            ticker,
            cache_path,
        )

        return NewsResult(
            ticker=ticker,
            articles=[],
            source="historical_replay",
            mode=DataSourceMode.BACKTEST,
            as_of=cutoff,
            availability=NewsAvailability.UNAVAILABLE,
        )

    # ---------------------------------------------------------------
    # Cache exists, so the source itself is available.
    # ---------------------------------------------------------------
    try:
        with open(cache_path, "r", encoding="utf-8") as f:
            raw_articles = json.load(f)
    except Exception as e:
        logger.warning(
            "Historical news cache failed to load for %s at %s: %s",
            ticker,
            cache_path,
            e,
        )

        return NewsResult(
            ticker=ticker,
            articles=[],
            source="historical_replay",
            mode=DataSourceMode.BACKTEST,
            as_of=cutoff,
            availability=NewsAvailability.ERROR,
        )

    if not isinstance(raw_articles, list):
        logger.warning(
            "Historical news cache for %s is not a JSON list: %s",
            ticker,
            cache_path,
        )

        return NewsResult(
            ticker=ticker,
            articles=[],
            source="historical_replay",
            mode=DataSourceMode.BACKTEST,
            as_of=cutoff,
            availability=NewsAvailability.ERROR,
        )

    articles: list[NewsArticle] = []

    for raw_article in raw_articles:
        try:
            article = NewsArticle(**raw_article)
        except Exception as e:
            logger.warning(
                "Skipping malformed cached article for %s: %s",
                ticker,
                e,
            )
            continue

        # -----------------------------------------------------------
        # Lookahead protection:
        # future articles are never visible during replay.
        # -----------------------------------------------------------
        if _ensure_utc(article.published_at) <= cutoff:
            articles.append(article)

    logger.info(
        "Historical news replay: ticker=%s cutoff=%s "
        "cached=%s eligible=%s availability=%s",
        ticker,
        cutoff.isoformat(),
        len(raw_articles),
        len(articles),
        NewsAvailability.AVAILABLE.value,
    )

    return NewsResult(
        ticker=ticker,
        articles=articles,
        source="historical_replay",
        mode=DataSourceMode.BACKTEST,
        as_of=cutoff,
        availability=NewsAvailability.AVAILABLE,
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
