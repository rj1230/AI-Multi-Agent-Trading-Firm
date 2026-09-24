"""
Portfolio Ledger
================

Persistent local portfolio state used by the trading system.

Design principles:

1. The ledger is the local portfolio source of truth.
2. Successful broker fills are the only events that mutate positions.
3. A fresh backtest must be explicitly isolated from previous state.
4. Live/paper operation may continue loading an existing ledger.
5. Backtests should pass fresh=True so an old ledger file can never leak
   positions or cash into a new simulation.
6. Historical acquisition cost is stored separately from current market value.
7. Realized P&L is recorded only when an actual sell fill is confirmed.
8. Each confirmed fill returns a FillAccounting event so downstream
   backtesting can distinguish execution events from realized trades.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from portfolio.state import (
    PortfolioSnapshot,
    Position,
    build_correlation_matrix,
)


DEFAULT_LEDGER_PATH = Path("portfolio") / "portfolio_ledger.json"


@dataclass(frozen=True)
class FillAccounting:
    """
    Accounting result produced by one confirmed broker fill.

    This represents the accounting impact of a single execution,
    rather than the cumulative portfolio realized P&L.
    """

    ticker: str
    side: str
    qty: float
    price: float
    average_cost: float
    realized_pnl: float
    remaining_shares: float
    position_closed: bool


class PortfolioLedger:
    def __init__(
        self,
        starting_equity: float,
        path: Path = DEFAULT_LEDGER_PATH,
        *,
        fresh: bool = False,
    ):
        if starting_equity <= 0:
            raise ValueError("starting_equity must be greater than zero.")

        self.path = Path(path)
        self.starting_equity = float(starting_equity)
        self.simulated_date: datetime | None = None

        if fresh:
            self._data = self._fresh_state()
            self._save(self._data)
        else:
            self._data = self._load_or_init()

    def _fresh_state(self) -> dict[str, Any]:
        return {
            "cash": self.starting_equity,
            "positions": {},
            "session_date": None,
            "session_starting_equity": self.starting_equity,
            "realized_pnl": 0.0,
        }

    def _load_or_init(self) -> dict[str, Any]:
        if self.path.exists():
            try:
                data = json.loads(self.path.read_text(encoding="utf-8"))

                if not isinstance(data, dict):
                    raise ValueError("Ledger JSON root must be an object.")

                data.setdefault("cash", self.starting_equity)
                data.setdefault("positions", {})
                data.setdefault("session_date", None)
                data.setdefault(
                    "session_starting_equity",
                    self.starting_equity,
                )
                data.setdefault("realized_pnl", 0.0)

                return data

            except (
                OSError,
                json.JSONDecodeError,
                ValueError,
            ) as exc:
                raise RuntimeError(
                    f"Unable to load portfolio ledger from {self.path}: {exc}"
                ) from exc

        data = self._fresh_state()
        self._save(data)
        return data

    def _save(
        self,
        data: dict[str, Any] | None = None,
    ) -> None:
        if data is None:
            data = self._data

        self.path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.path.write_text(
            json.dumps(
                data,
                indent=2,
                default=str,
            ),
            encoding="utf-8",
        )

    def set_simulated_date(
        self,
        simulated_date: datetime | None,
    ) -> None:
        if simulated_date is None:
            self.simulated_date = None
            return

        if simulated_date.tzinfo is None:
            simulated_date = simulated_date.replace(tzinfo=timezone.utc)

        self.simulated_date = simulated_date.astimezone(timezone.utc)

    def _current_timestamp(self) -> datetime:
        if self.simulated_date is not None:
            return self.simulated_date

        return datetime.now(timezone.utc)

    def _mark_to_market(self) -> None:
        if not self._data.get("positions"):
            return

        positions = self._data["positions"]
        tickers = list(positions.keys())

        try:
            from data_sources import fetch_ohlcv
        except ImportError:
            return

        for ticker in tickers:
            position = positions.get(ticker)

            if not isinstance(position, dict):
                continue

            shares = float(position.get("shares", 0.0))

            if shares <= 0:
                continue

            try:
                series = fetch_ohlcv(
                    ticker,
                    lookback_days=1,
                )

                if series.is_empty or series.latest is None:
                    continue

                price = float(series.latest.close)

                # IMPORTANT:
                # Mark-to-market updates only current market value.
                # It must never overwrite historical average cost.
                position["market_value"] = shares * price

            except Exception:
                continue

        self._save()

    def record_fill(
        self,
        ticker: str,
        side: str,
        qty: float,
        price: float,
        sector: str,
    ) -> FillAccounting:
        """
        Record a confirmed broker fill.

        BUY:
            - increases shares
            - updates weighted average cost
            - records zero realized P&L

        SELL:
            - decreases shares
            - preserves historical average cost
            - records realized P&L for this specific fill
            - removes the position when fully closed

        Returns:
            FillAccounting describing the accounting impact of this
            individual confirmed fill.
        """

        ticker = str(ticker).upper().strip()
        side = str(side).lower().strip()
        qty = float(qty)
        price = float(price)

        if not ticker:
            raise ValueError("ticker must not be empty.")

        if side not in {"buy", "sell"}:
            raise ValueError(f"Unsupported fill side: {side}")

        if qty <= 0:
            raise ValueError("qty must be greater than zero.")

        if price <= 0:
            raise ValueError("price must be greater than zero.")

        positions = self._data.setdefault(
            "positions",
            {},
        )

        # --------------------------------------------------------------
        # BUY
        # --------------------------------------------------------------
        if side == "buy":
            existing = positions.get(ticker)

            if existing is None:
                average_cost = price
                new_shares = qty

                positions[ticker] = {
                    "ticker": ticker,
                    "shares": new_shares,
                    "sector": sector,
                    "average_cost": average_cost,
                    "market_value": new_shares * price,
                }

            else:
                old_shares = float(existing.get("shares", 0.0))

                if "average_cost" in existing:
                    old_average_cost = float(existing["average_cost"])

                elif old_shares > 0:
                    # Backward compatibility with older ledger files
                    # that did not persist average_cost.
                    old_market_value = float(
                        existing.get(
                            "market_value",
                            0.0,
                        )
                    )

                    old_average_cost = old_market_value / old_shares

                else:
                    old_average_cost = price

                new_shares = old_shares + qty

                average_cost = (
                    (old_shares * old_average_cost) + (qty * price)
                ) / new_shares

                existing["ticker"] = ticker
                existing["shares"] = new_shares
                existing["sector"] = sector
                existing["average_cost"] = average_cost
                existing["market_value"] = new_shares * price

            self._data["cash"] = float(
                self._data.get(
                    "cash",
                    self.starting_equity,
                )
            ) - (qty * price)

            self._data["session_date"] = self._current_timestamp().isoformat()

            self._save()

            return FillAccounting(
                ticker=ticker,
                side=side,
                qty=qty,
                price=price,
                average_cost=float(
                    positions[ticker].get(
                        "average_cost",
                        price,
                    )
                ),
                realized_pnl=0.0,
                remaining_shares=float(
                    positions[ticker].get(
                        "shares",
                        0.0,
                    )
                ),
                position_closed=False,
            )

        # --------------------------------------------------------------
        # SELL
        # --------------------------------------------------------------
        existing = positions.get(ticker)

        if existing is None:
            raise ValueError(f"Cannot sell {ticker}: no position exists.")

        existing_shares = float(existing.get("shares", 0.0))

        if qty > existing_shares + 1e-9:
            raise ValueError(
                f"Cannot sell {qty} shares of {ticker}: "
                f"only {existing_shares} shares held."
            )

        # Historical acquisition cost.
        # This MUST NOT be replaced by current market value.
        average_cost = float(
            existing.get(
                "average_cost",
                price,
            )
        )

        # Realized P&L for THIS sell fill only.
        realized_pnl = qty * (price - average_cost)

        # Persist cumulative realized P&L.
        self._data["realized_pnl"] = (
            float(
                self._data.get(
                    "realized_pnl",
                    0.0,
                )
            )
            + realized_pnl
        )

        remaining = existing_shares - qty

        self._data["cash"] = float(
            self._data.get(
                "cash",
                self.starting_equity,
            )
        ) + (qty * price)

        position_closed = remaining <= 1e-9

        if position_closed:
            positions.pop(ticker, None)
            remaining = 0.0

        else:
            existing["shares"] = remaining

            # Current market value at the confirmed fill price.
            # Historical average_cost remains unchanged.
            existing["market_value"] = remaining * price

        self._data["session_date"] = self._current_timestamp().isoformat()

        self._save()

        return FillAccounting(
            ticker=ticker,
            side=side,
            qty=qty,
            price=price,
            average_cost=average_cost,
            realized_pnl=realized_pnl,
            remaining_shares=remaining,
            position_closed=position_closed,
        )

    def snapshot(self) -> PortfolioSnapshot:
        self._mark_to_market()

        raw_positions = self._data.get(
            "positions",
            {},
        )

        positions: dict[str, Position] = {}

        total_market_value = 0.0

        for ticker, raw in raw_positions.items():
            shares = float(raw.get("shares", 0.0))

            if shares <= 0:
                continue

            market_value = float(
                raw.get(
                    "market_value",
                    0.0,
                )
            )

            sector = str(
                raw.get(
                    "sector",
                    "Unknown",
                )
            )

            positions[ticker] = Position(
                ticker=ticker,
                shares=shares,
                sector=sector,
                market_value=market_value,
            )

            total_market_value += market_value

        cash = float(
            self._data.get(
                "cash",
                self.starting_equity,
            )
        )

        equity = cash + total_market_value

        correlation_matrix = build_correlation_matrix(
            list(positions.keys()),
            lookback_days=30,
        )

        return PortfolioSnapshot(
            equity=equity,
            starting_equity=self.starting_equity,
            positions=positions,
            correlation_matrix=correlation_matrix,
        )

    def get_cash(self) -> float:
        return float(
            self._data.get(
                "cash",
                self.starting_equity,
            )
        )

    def get_realized_pnl(self) -> float:
        return float(
            self._data.get(
                "realized_pnl",
                0.0,
            )
        )

    def get_positions(
        self,
    ) -> dict[str, dict[str, Any]]:
        return json.loads(
            json.dumps(
                self._data.get(
                    "positions",
                    {},
                )
            )
        )

    def reset(self) -> None:
        self._data = self._fresh_state()
        self._save(self._data)
