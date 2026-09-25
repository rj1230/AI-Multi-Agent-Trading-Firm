"""
Seed data_cache/news_cache/<TICKER>.json for backtesting.

Fetches current NewsAPI headlines via data_sources.live.fetch_news() and
writes them to the exact cache path/shape data_sources.historical.fetch_news()
reads from (a flat JSON list of NewsArticle-shaped dicts).

Run from the project root (same place you run `python -m backtest.runner`),
with NEWSAPI_KEY set in your environment/.env:

    python seed_news_cache.py AAPL MSFT GOOGL JPM

Or with no args, defaults to AAPL,MSFT,GOOGL,JPM.

Important limitation:
NewsAPI's free tier only serves articles from roughly the last month, and
live.py calls the /v2/top-headlines endpoint (breaking-news style, filtered
by a `q` search term) rather than the /v2/everything search endpoint — so
coverage for a specific ticker can be thin even within that window. This
script will not usefully backfill a backtest window older than ~30 days
from today. Re-run it periodically to accumulate more history over time —
it merges with (not overwrites) whatever's already cached.
"""

import json
import sys
from pathlib import Path

from data_sources.live import fetch_news

NEWS_CACHE_DIR = Path("data_cache/news_cache")


def seed_ticker(ticker: str, page_size: int = 100) -> int:
    result = fetch_news(ticker, page_size=page_size)

    if result.is_empty:
        print(f"[{ticker}] WARNING: NewsAPI returned 0 articles — nothing to cache")
        return 0

    NEWS_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = NEWS_CACHE_DIR / f"{ticker}.json"

    # Merge with any existing cached articles instead of overwriting, so
    # repeated seeding runs accumulate coverage rather than losing it.
    existing: list[dict] = []
    if cache_path.exists():
        with open(cache_path, "r", encoding="utf-8") as f:
            existing = json.load(f)

    existing_urls = {a["url"] for a in existing}
    new_articles = [
        a.model_dump(mode="json") for a in result.articles if a.url not in existing_urls
    ]

    merged = existing + new_articles

    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(merged, f, indent=2)

    print(
        f"[{ticker}] cached {len(new_articles)} new articles "
        f"({len(merged)} total) -> {cache_path}"
    )
    return len(new_articles)


def main():
    tickers = [t.upper() for t in sys.argv[1:]] or ["AAPL", "MSFT", "GOOGL", "JPM"]
    for ticker in tickers:
        seed_ticker(ticker)


if __name__ == "__main__":
    main()
