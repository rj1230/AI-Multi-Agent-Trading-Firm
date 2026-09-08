"""
SimBroker: in-memory broker for BACKTEST mode -- no network calls. Fills
market orders instantly at whatever price_lookup(ticker) returns; the
backtest runner (Phase 9) wires that to the current simulated bar's close,
so a strategy's backtest and live runs go through the identical
ExecutionAgent code path (Phase 4.1's dual-mode principle, applied here).

Idempotency: if the same client_order_id is submitted twice (e.g.
ExecutionAgent's retry-after-timeout path), SimBroker returns the
original fill instead of double-executing -- this exists specifically to
answer the doc's Phase 6 self-check about double-submission on retry.
"""
from __future__ import annotations

from typing import Callable, Optional

from broker.protocol import AccountInfo, BrokerPosition, OrderResult, OrderSide


class SimBroker:
    def __init__(self, starting_cash: float, price_lookup: Callable[[str], float]):
        self.cash = starting_cash
        self.price_lookup = price_lookup
        self._positions: dict[str, BrokerPosition] = {}
        self._orders_by_client_id: dict[str, OrderResult] = {}
        self._next_order_id = 1

    def submit_order(
        self, ticker: str, side: OrderSide, qty: float,
        client_order_id: Optional[str] = None,
    ) -> OrderResult:
        if client_order_id and client_order_id in self._orders_by_client_id:
            return self._orders_by_client_id[client_order_id]

        order_id = f"SIM-{self._next_order_id}"
        self._next_order_id += 1
        price = self.price_lookup(ticker)
        cost = price * qty

        if side == "buy":
            result = self._fill_buy(order_id, ticker, qty, price, cost)
        else:
            result = self._fill_sell(order_id, ticker, qty, price, cost)

        if client_order_id:
            self._orders_by_client_id[client_order_id] = result
        return result

    def _fill_buy(self, order_id, ticker, qty, price, cost) -> OrderResult:
        if cost > self.cash:
            return OrderResult(
                order_id=order_id, ticker=ticker, side="buy", qty=qty,
                status="rejected", raw={"reason": "insufficient cash"},
            )
        self.cash -= cost
        existing = self._positions.get(ticker)
        if existing:
            new_qty = existing.qty + qty
            new_avg = (existing.avg_entry_price * existing.qty + cost) / new_qty
            self._positions[ticker] = BrokerPosition(
                ticker=ticker, qty=new_qty, market_value=new_qty * price, avg_entry_price=new_avg,
            )
        else:
            self._positions[ticker] = BrokerPosition(
                ticker=ticker, qty=qty, market_value=cost, avg_entry_price=price,
            )
        return OrderResult(
            order_id=order_id, ticker=ticker, side="buy", qty=qty,
            status="filled", filled_avg_price=price,
        )

    def _fill_sell(self, order_id, ticker, qty, price, cost) -> OrderResult:
        existing = self._positions.get(ticker)
        held_qty = existing.qty if existing else 0.0
        if qty > held_qty:
            return OrderResult(
                order_id=order_id, ticker=ticker, side="sell", qty=qty,
                status="rejected", raw={"reason": "insufficient shares"},
            )
        self.cash += cost
        remaining = held_qty - qty
        if remaining <= 0:
            del self._positions[ticker]
        else:
            self._positions[ticker] = BrokerPosition(
                ticker=ticker, qty=remaining, market_value=remaining * price,
                avg_entry_price=existing.avg_entry_price,
            )
        return OrderResult(
            order_id=order_id, ticker=ticker, side="sell", qty=qty,
            status="filled", filled_avg_price=price,
        )

    def get_positions(self) -> dict[str, BrokerPosition]:
        return dict(self._positions)

    def get_account(self) -> AccountInfo:
        equity = self.cash + sum(p.market_value for p in self._positions.values())
        return AccountInfo(equity=equity, cash=self.cash, buying_power=self.cash)
