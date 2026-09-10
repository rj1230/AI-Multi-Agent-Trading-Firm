"""
Populates data_cache/news_cache/<ticker>.json with sample headlines so
backtest-mode NewsAgent has something to read.

IMPORTANT — read before running:
This generates PLACEHOLDER headlines, not real historical news. It exists
to unblock your Phase 9 vectorbt crosscheck, whose actual job is
validating Sharpe/drawdown MATH against a real equity curve -- it needs
*some* non-neutral signals to produce trades, not genuinely predictive
news (see run_vectorbt_crosscheck.py's own docstring: it explicitly does
not validate trading strategy quality).

DO NOT use this cache to draw real conclusions about AAPL's actual
performance or your strategy's real-world edge -- the sentiment labels
below are arbitrary, not researched. For a real backtest, replace this
file's contents with genuinely archived headlines (e.g. from a paid news
archive API, or your own accumulated live-fetch history).

Usage:
    python populate_news_cache_sample.py AAPL 2026-07-01 2026-07-31
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

CACHE_DIR = Path("data_cache") / "news_cache"

# Rotates through a few headline "flavors" so the sample cache produces a
# mix of bullish / bearish / neutral days rather than all-identical text
# (which the LLM might otherwise score identically every time).
SAMPLE_HEADLINES = [
    ("bullish", "{ticker} shares climb on strong quarterly demand signals"),
    ("bullish", "Analysts raise price targets for {ticker} citing product momentum"),
    ("bearish", "{ticker} faces supply chain headwinds, analysts note caution"),
    ("bearish", "{ticker} stock slips amid broader tech sector pullback"),
    ("neutral", "{ticker} trades flat as investors await earnings guidance"),
    ("neutral", "{ticker} holds steady in low-volume summer trading session"),
]


def build_articles(ticker: str, start: datetime, end: datetime) -> list[dict]:
    articles = []
    current = start
    i = 0
    while current <= end:
        flavor, template = SAMPLE_HEADLINES[i % len(SAMPLE_HEADLINES)]
        articles.append(
            {
                "title": template.format(ticker=ticker),
                "description": f"Placeholder {flavor} headline generated for backtest coverage.",
                "source": "sample_cache",
                "published_at": current.replace(
                    hour=13, minute=0, second=0, tzinfo=timezone.utc
                ).isoformat(),
                "url": f"https://example.com/{ticker.lower()}-news-{current.date()}",
            }
        )
        current += timedelta(days=1)
        i += 1
    return articles


def main() -> None:
    if len(sys.argv) != 4:
        print("Usage: python populate_news_cache_sample.py TICKER START_DATE END_DATE")
        print(
            "Example: python populate_news_cache_sample.py AAPL 2026-07-01 2026-07-31"
        )
        sys.exit(1)

    ticker = sys.argv[1].upper()
    start = datetime.strptime(sys.argv[2], "%Y-%m-%d").replace(tzinfo=timezone.utc)
    end = datetime.strptime(sys.argv[3], "%Y-%m-%d").replace(tzinfo=timezone.utc)

    articles = build_articles(ticker, start, end)

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = CACHE_DIR / f"{ticker}.json"

    if cache_path.exists():
        print(f"{cache_path} already exists -- overwrite? (y/n): ", end="")
        if input().strip().lower() != "y":
            print("Aborted.")
            return

    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(articles, f, indent=2)

    print(f"Wrote {len(articles)} placeholder articles to {cache_path}")
    print("Remember: these are PLACEHOLDER headlines for unblocking the")
    print("metrics crosscheck, not real historical news. See this file's")
    print("module docstring before using the results for real analysis.")


if __name__ == "__main__":
    main()
