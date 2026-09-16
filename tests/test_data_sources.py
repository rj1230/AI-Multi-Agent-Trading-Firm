"""
Phase 2 data-source contract and historical replay tests.

Coverage:

1. OHLCV replay never exposes future bars.
2. The lookahead test itself catches a deliberately injected leak.
3. News replay never exposes future articles.
4. Missing historical news cache is explicitly UNAVAILABLE.
5. Existing historical news cache with no eligible articles is still
   AVAILABLE, not silently treated as missing data.
6. Historical news cache containing eligible and future articles returns
   only eligible articles.
7. Live and historical data-source modules expose the expected interface.
"""

from datetime import datetime, timedelta, timezone
import json

import pytest

from data_sources.schemas import (
    DataSourceMode,
    NewsAvailability,
    NewsArticle,
    NewsResult,
    OHLCVBar,
    OHLCVSeries,
)
from data_sources import historical


def test_no_lookahead_bias_ohlcv(monkeypatch, tmp_path):
    """No bar in the replayed series may be timestamped after simulated_date."""
    now = datetime(2026, 6, 15, tzinfo=timezone.utc)

    bars = [
        OHLCVBar(
            timestamp=now - timedelta(days=i),
            open=100,
            high=101,
            low=99,
            close=100,
            volume=1000,
        )
        for i in range(0, 10)
    ]

    # Feed in ascending order, as _load_or_fetch_full_history would produce.
    bars = list(reversed(bars))

    monkeypatch.setattr(
        historical,
        "_load_or_fetch_full_history",
        lambda ticker: bars,
    )

    simulated_date = now - timedelta(days=4)

    series = historical.fetch_ohlcv(
        "AAPL",
        simulated_date,
        lookback_days=30,
    )

    assert all(b.timestamp <= simulated_date for b in series.bars), (
        "Lookahead bias detected: replay returned bars after simulated_date"
    )


def test_lookahead_bias_test_catches_a_real_leak(monkeypatch):
    """Sanity check on the test itself: deliberately inject a future bar
    and confirm the assertion actually fails."""
    now = datetime(2026, 6, 15, tzinfo=timezone.utc)
    simulated_date = now - timedelta(days=4)

    leaked_future_bar = OHLCVBar(
        timestamp=simulated_date + timedelta(days=1),
        open=100,
        high=101,
        low=99,
        close=100,
        volume=1000,
    )

    with pytest.raises(AssertionError):
        assert leaked_future_bar.timestamp <= simulated_date


def test_no_lookahead_bias_news(tmp_path, monkeypatch):
    """Historical news replay must never expose a future article."""
    ticker = "AAPL"
    now = datetime(2026, 6, 15, tzinfo=timezone.utc)

    cache_dir = tmp_path / "news_cache"
    cache_dir.mkdir()

    articles = [
        {
            "title": "Old news",
            "description": None,
            "source": "TestWire",
            "published_at": (now - timedelta(days=5)).isoformat(),
            "url": "https://example.com/old",
        },
        {
            "title": "Future leak",
            "description": None,
            "source": "TestWire",
            "published_at": (now + timedelta(days=1)).isoformat(),
            "url": "https://example.com/future",
        },
    ]

    (cache_dir / f"{ticker}.json").write_text(
        json.dumps(articles),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        historical,
        "NEWS_CACHE_DIR",
        cache_dir,
    )

    result = historical.fetch_news(ticker, now)

    titles = [article.title for article in result.articles]

    assert result.availability == NewsAvailability.AVAILABLE
    assert result.has_articles

    assert "Future leak" not in titles, (
        "Lookahead bias: future article leaked into replay"
    )

    assert "Old news" in titles


def test_missing_historical_news_cache_is_unavailable(
    tmp_path,
    monkeypatch,
):
    """
    A missing historical cache must be explicitly marked UNAVAILABLE.

    This is different from:
        AVAILABLE + zero matching articles
    """

    cache_dir = tmp_path / "news_cache"
    cache_dir.mkdir()

    monkeypatch.setattr(
        historical,
        "NEWS_CACHE_DIR",
        cache_dir,
    )

    simulated_date = datetime(
        2024,
        10,
        15,
        tzinfo=timezone.utc,
    )

    result = historical.fetch_news(
        "GOOGL",
        simulated_date,
    )

    assert result.articles == []
    assert result.is_empty
    assert result.is_unavailable
    assert not result.is_available
    assert not result.has_articles
    assert result.availability == NewsAvailability.UNAVAILABLE
    assert result.mode == DataSourceMode.BACKTEST


def test_existing_cache_with_no_eligible_articles_is_available(
    tmp_path,
    monkeypatch,
):
    """
    If a cache exists but all cached articles are after the simulated date,
    the source itself is AVAILABLE.

    The replay correctly returns zero articles because future articles must
    not leak into the simulation.
    """

    cache_dir = tmp_path / "news_cache"
    cache_dir.mkdir()

    ticker = "AAPL"

    simulated_date = datetime(
        2024,
        10,
        15,
        tzinfo=timezone.utc,
    )

    future_article = {
        "title": "Future headline",
        "description": "Published after the simulated date.",
        "source": "TestWire",
        "published_at": "2026-01-01T12:00:00+00:00",
        "url": "https://example.com/future",
    }

    (cache_dir / f"{ticker}.json").write_text(
        json.dumps([future_article]),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        historical,
        "NEWS_CACHE_DIR",
        cache_dir,
    )

    result = historical.fetch_news(
        ticker,
        simulated_date,
    )

    assert result.articles == []
    assert result.is_empty

    # Cache exists and loaded successfully.
    assert result.is_available
    assert not result.is_unavailable
    assert not result.has_articles
    assert result.availability == NewsAvailability.AVAILABLE


def test_historical_news_returns_only_eligible_articles(
    tmp_path,
    monkeypatch,
):
    """
    A cache containing past and future articles must return only the
    articles known at the simulated date.
    """

    cache_dir = tmp_path / "news_cache"
    cache_dir.mkdir()

    ticker = "MSFT"

    simulated_date = datetime(
        2024,
        10,
        15,
        12,
        0,
        tzinfo=timezone.utc,
    )

    articles = [
        {
            "title": "Eligible historical headline",
            "description": "Known before the simulation date.",
            "source": "TestWire",
            "published_at": "2024-10-10T09:00:00+00:00",
            "url": "https://example.com/eligible",
        },
        {
            "title": "Same-day headline",
            "description": "Known on the simulation date.",
            "source": "TestWire",
            "published_at": "2024-10-15T09:00:00+00:00",
            "url": "https://example.com/same-day",
        },
        {
            "title": "Future headline",
            "description": "Must never enter the replay.",
            "source": "TestWire",
            "published_at": "2024-10-16T09:00:00+00:00",
            "url": "https://example.com/future",
        },
    ]

    (cache_dir / f"{ticker}.json").write_text(
        json.dumps(articles),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        historical,
        "NEWS_CACHE_DIR",
        cache_dir,
    )

    result = historical.fetch_news(
        ticker,
        simulated_date,
    )

    titles = [article.title for article in result.articles]

    assert result.availability == NewsAvailability.AVAILABLE
    assert result.has_articles
    assert len(result.articles) == 2

    assert "Eligible historical headline" in titles
    assert "Same-day headline" in titles
    assert "Future headline" not in titles

    assert all(article.published_at <= simulated_date for article in result.articles)


def test_live_and_historical_same_shape():
    """
    Both modules must expose the expected data-source interface.

    NewsResult explicitly includes availability metadata so downstream
    consumers can distinguish unavailable historical data from a valid
    zero-article result.
    """
    from data_sources import live

    assert set(OHLCVSeries.model_fields.keys()) == {
        "ticker",
        "bars",
        "source",
        "mode",
        "as_of",
    }

    assert set(NewsResult.model_fields.keys()) == {
        "ticker",
        "articles",
        "source",
        "mode",
        "as_of",
        "availability",
    }

    assert callable(live.fetch_ohlcv)
    assert callable(historical.fetch_ohlcv)

    assert callable(live.fetch_news)
    assert callable(historical.fetch_news)
