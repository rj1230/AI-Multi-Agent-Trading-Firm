"""
Ticker -> sector lookup for RiskAgent's per-sector-cap check.

THIS IS A PLACEHOLDER. There's no real sector data provider wired in yet
-- a genuine gap, not an oversight. Real options, in rough order of
effort: (1) hand-maintain this dict for whatever universe of tickers the
bot actually trades, (2) pull sector from Alpaca's asset metadata
(client.get_asset(ticker)) if it's populated for your account tier, or
(3) a free reference like the SEC's company facts / a static S&P sector
CSV. Until one of those is wired in, unmapped tickers fall back to
"Unknown" -- which the per-sector cap still enforces (an "Unknown" bucket
is still a bucket), it just won't group unmapped tickers with their real
sector peers.
"""
from __future__ import annotations

SECTOR_MAP: dict[str, str] = {
    "AAPL": "Tech",
    "MSFT": "Tech",
    "GOOGL": "Tech",
    "NVDA": "Tech",
    "META": "Tech",
    "AMZN": "Consumer Discretionary",
    "TSLA": "Consumer Discretionary",
    "XOM": "Energy",
    "CVX": "Energy",
    "JPM": "Financials",
    "BAC": "Financials",
}


def get_sector(ticker: str) -> str:
    return SECTOR_MAP.get(ticker.upper(), "Unknown")
