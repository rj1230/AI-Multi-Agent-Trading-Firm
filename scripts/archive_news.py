r"""
Rolling news archiver for BACKTEST mode's historical news cache.

Run this periodically (e.g. once a day) for each ticker you care about.
Each run fetches today's live headlines via the existing NewsAPI wrapper
in data_sources/live.py and merges them into
data_cache/news_cache/<ticker>.json, so historical.fetch_news() has real
data to serve for any *future* backtest date >= when you started running
this script.

This does NOT and cannot backfill dates before you started archiving --
NewsAPI's free tier doesn't serve headlines that old. Historical news
signal for dates before your archive began will stay neutral; that's a
known, accepted gap (see historical.py's docstring).

Usage:
    python scripts/archive_news.py AAPL MSFT TSLA
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# scripts/ is a subfolder, so add the project root to sys.path --
# otherwise `data_sources` (which lives at the project root) can't be
# found when this file is run directly as `python scripts/archive_news.py`.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data_sources import live
from data_sources.schemas import NewsArticle

NEWS_CACHE_DIR = PROJECT_ROOT / "data_cache" / "news_cache"


def _load_existing(ticker: str) -> list[dict]:
    path = NEWS_CACHE_DIR / f"{ticker}.json"
    if not path.exists():
        return []
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _save(ticker: str, articles: list[dict]) -> None:
    NEWS_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = NEWS_CACHE_DIR / f"{ticker}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(articles, f, indent=2, default=str)


def archive_ticker(ticker: str) -> None:
    existing_raw = _load_existing(ticker)
    existing_urls = {a["url"] for a in existing_raw}

    # live.fetch_news should hit NewsAPI directly for "now" -- no
    # simulated_date needed here since this always runs in real time,
    # regardless of TRADING_MODE.
    result = live.fetch_news(ticker)

    new_count = 0
    for article in result.articles:
        if article.url in existing_urls:
            continue
        existing_raw.append(article.model_dump(mode="json"))
        existing_urls.add(article.url)
        new_count += 1

    _save(ticker, existing_raw)
    print(
        f"{ticker}: fetched {len(result.articles)}, added {new_count} new, "
        f"cache now has {len(existing_raw)} total"
    )


def main() -> None:
    tickers = sys.argv[1:]
    if not tickers:
        print("Usage: python scripts/archive_news.py TICKER [TICKER ...]")
