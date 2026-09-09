"""
Local JSON-backed portfolio ledger -- the concrete implementation behind
"fixed paper balance, positions tracked in a local file" (chosen over
polling broker.get_account() every run, or a real DB).

Two different "starting" concepts, kept deliberately separate:
  - PAPER_STARTING_EQUITY (config/settings.py): the account's lifetime
    opening balance. Set once, never touched again after the ledger file
    is first created.
  - session starting equity: equity as of the start of TODAY's session --
    this is what CircuitBreaker.check() compares against, per the doc's
    "-3% account equity IN A DAY" rule (section 3). It rolls forward every
    calendar day, not every run, so a bad day doesn't get diluted by
    yesterday's gains and a good day doesn't hide today's losses.

File layout (portfolio/ledger.json by default):
{
  "cash": 100000.0,
  "positions": {"AAPL": {"shares": 10, "sector": "Tech", "avg_entry_price": 190.0}},
  "session_date": "2026-09-09",
  "session_starting_equity": 100000.0
}
"""
from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from typing import Optional

from data_sources import fetch_ohlcv
from portfolio.correlation import returns_from_ohlcv, update_correlation_matrix
from portfolio.state import PortfolioSnapshot, Position

DEFAULT_LEDGER_PATH = Path("portfolio/ledger.json")


class PortfolioLedger:
    def __init__(self, starting_equity: float, path: Path = DEFAULT_LEDGER_PATH):
        self.path = Path(path)
        self.starting_equity = starting_equity
        self._data = self._load_or_init()

    def _load_or_init(self) -> dict:
        if self.path.exists():
            return json.loads(self.path.read_text())
        data = {
            "cash": self.starting_equity,
            "positions": {},
            "session_date": None,
            "session_starting_equity": self.starting_equity,
        }
        self._save(data)
        return data

    def _save(self, data: Optional[dict] = None) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(data if data is not None else self._data, indent=2))

    def _mark_to_market(self) -> dict[str, float]:
        """market_value per held ticker, using the latest close. Falls
        back to avg_entry_price if a live price can't be fetched, rather
        than crashing a risk check over a data hiccup."""
        values = {}
        for ticker, pos in self._data["positions"].items():
            try:
                series = fetch_ohlcv(ticker, lookback_days=1)
                price = series.latest.close if not series.is_empty else pos["avg_entry_price"]
            except Exception:
                price = pos["avg_entry_price"]
            values[ticker] = price * pos["shares"]
        return values

    def _roll_session_if_new_day(self, current_equity: float) -> None:
        today = date.today().isoformat()
        if self._data.get("session_date") != today:
            self._data["session_date"] = today
            self._data["session_starting_equity"] = current_equity
            self._save()

    def snapshot(self) -> PortfolioSnapshot:
        """Builds a PortfolioSnapshot for RiskAgent/PortfolioRiskCoordinator
        from current ledger state. Rolls the session starting-equity
        forward if this is the first call on a new calendar day."""
        market_values = self._mark_to_market()
        equity = self._data["cash"] + sum(market_values.values())
        self._roll_session_if_new_day(equity)

        positions = {
            ticker: Position(
                ticker=ticker, shares=pos["shares"], sector=pos["sector"],
                market_value=market_values[ticker],
            )
            for ticker, pos in self._data["positions"].items()
        }

        correlation_matrix = None
        tickers = list(self._data["positions"].keys())
        if tickers:
            returns_by_ticker = {}
            for ticker in tickers:
                series = fetch_ohlcv(ticker, lookback_days=30)
                if not series.is_empty:
                    returns_by_ticker[ticker] = returns_from_ohlcv(series)
            correlation_matrix = update_correlation_matrix(returns_by_ticker)

        return PortfolioSnapshot(
            equity=equity,
            starting_equity=self._data["session_starting_equity"],
            positions=positions,
            correlation_matrix=correlation_matrix,
        )

    def record_fill(self, ticker: str, side: str, qty: float, price: float, sector: str) -> None:
        """Called by ExecutionAgent after a broker fill to keep the ledger
        in sync with what actually happened."""
        cost = price * qty
        if side == "buy":
            self._data["cash"] -= cost
            existing = self._data["positions"].get(ticker)
            if existing:
                new_qty = existing["shares"] + qty
                new_avg = (existing["avg_entry_price"] * existing["shares"] + cost) / new_qty
                self._data["positions"][ticker] = {
                    "shares": new_qty, "sector": sector, "avg_entry_price": new_avg,
                }
            else:
                self._data["positions"][ticker] = {
                    "shares": qty, "sector": sector, "avg_entry_price": price,
                }
        elif side == "sell":
            existing = self._data["positions"].get(ticker)
            if not existing:
                raise ValueError(f"Cannot sell {ticker}: no position on record in the ledger")
            self._data["cash"] += cost
            remaining = existing["shares"] - qty
            if remaining <= 0:
                del self._data["positions"][ticker]
            else:
                self._data["positions"][ticker] = {**existing, "shares": remaining}
        else:
            raise ValueError(f"Unknown side: {side}")
        self._save()
