"""
Local JSON-backed portfolio ledger.

The ledger is the portfolio source of truth for both paper/live operation
and historical backtesting.

Important:
- Live/paper sessions use today's real calendar date.
- Backtests can set ``ledger.simulated_date`` so session accounting follows
  the historical simulation date rather than the machine's current date.
- BUY and SELL fills are recorded only after actual broker execution.
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

        # Historical backtests set this explicitly for every simulated day.
        # Live/paper mode leaves it as None and therefore uses today's date.
        self.simulated_date: Optional[date] = None

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
        self.path.write_text(
            json.dumps(
                data if data is not None else self._data,
                indent=2,
            )
        )

    def _current_session_date(self) -> str:
        """
        Return the date that should govern session accounting.

        Backtests explicitly set ``simulated_date``.
        Live/paper trading falls back to the real calendar date.
        """
        if self.simulated_date is not None:
            return self.simulated_date.isoformat()

        return date.today().isoformat()

    def _mark_to_market(self) -> dict[str, float]:
        """
        Return market value per held ticker using the latest price visible
        to the current data-source clock.

        During backtests, data_sources.set_simulated_date() controls the
        historical cutoff, preventing future prices from leaking into the
        current simulated session.
        """
        values: dict[str, float] = {}

        for ticker, pos in self._data["positions"].items():
            try:
                series = fetch_ohlcv(
                    ticker,
                    lookback_days=1,
                )

                if series.is_empty or series.latest is None:
                    price = pos["avg_entry_price"]
                else:
                    price = series.latest.close

            except Exception:
                price = pos["avg_entry_price"]

            values[ticker] = price * pos["shares"]

        return values

    def _roll_session_if_new_day(self, current_equity: float) -> None:
        """
        Roll session starting equity when the effective trading date changes.

        The effective date is simulated during backtests and the real date
        during live/paper operation.
        """
        session_date = self._current_session_date()

        if self._data.get("session_date") != session_date:
            self._data["session_date"] = session_date
            self._data["session_starting_equity"] = current_equity
            self._save()

    def snapshot(self) -> PortfolioSnapshot:
        """
        Build a PortfolioSnapshot for RiskAgent and PortfolioRiskCoordinator.

        The snapshot is marked to market using the current data-source clock,
        then session starting equity is rolled using the effective session
        date.
        """
        market_values = self._mark_to_market()

        equity = self._data["cash"] + sum(market_values.values())

        self._roll_session_if_new_day(equity)

        positions = {
            ticker: Position(
                ticker=ticker,
                shares=pos["shares"],
                sector=pos["sector"],
                market_value=market_values[ticker],
            )
            for ticker, pos in self._data["positions"].items()
        }

        correlation_matrix = None

        tickers = list(self._data["positions"].keys())

        if tickers:
            returns_by_ticker = {}

            for ticker in tickers:
                series = fetch_ohlcv(
                    ticker,
                    lookback_days=30,
                )

                if not series.is_empty:
                    returns_by_ticker[ticker] = returns_from_ohlcv(series)

            correlation_matrix = update_correlation_matrix(returns_by_ticker)

        return PortfolioSnapshot(
            equity=equity,
            starting_equity=self._data["session_starting_equity"],
            positions=positions,
            correlation_matrix=correlation_matrix,
        )

    def record_fill(
        self,
        ticker: str,
        side: str,
        qty: float,
        price: float,
        sector: str,
    ) -> None:
        """
        Record an actual broker fill.

        This method is intentionally called only after the broker reports
        a successful fill.
        """
        if qty <= 0:
            raise ValueError("Fill quantity must be positive.")

        if price <= 0:
            raise ValueError("Fill price must be positive.")

        cost = price * qty

        if side == "buy":
            if cost > self._data["cash"] + 1e-9:
                raise ValueError(
                    f"Cannot record BUY for {ticker}: "
                    f"cost={cost:.2f} exceeds cash={self._data['cash']:.2f}"
                )

            self._data["cash"] -= cost

            existing = self._data["positions"].get(ticker)

            if existing:
                old_qty = existing["shares"]
                new_qty = old_qty + qty

                new_avg = (existing["avg_entry_price"] * old_qty + cost) / new_qty

                self._data["positions"][ticker] = {
                    "shares": new_qty,
                    "sector": sector,
                    "avg_entry_price": new_avg,
                }

            else:
                self._data["positions"][ticker] = {
                    "shares": qty,
                    "sector": sector,
                    "avg_entry_price": price,
                }

        elif side == "sell":
            existing = self._data["positions"].get(ticker)

            if not existing:
                raise ValueError(
                    f"Cannot sell {ticker}: no position on record in the ledger"
                )

            held_qty = existing["shares"]

            if qty > held_qty + 1e-9:
                raise ValueError(
                    f"Cannot sell {ticker}: requested {qty} shares, "
                    f"but only {held_qty} are held."
                )

            self._data["cash"] += cost

            remaining = held_qty - qty

            if remaining <= 1e-9:
                del self._data["positions"][ticker]
            else:
                self._data["positions"][ticker] = {
                    **existing,
                    "shares": remaining,
                }

        else:
            raise ValueError(f"Unknown side: {side}")

        self._save()
