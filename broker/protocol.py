"""
Broker protocol: the interface ExecutionAgent depends on. It never imports
Alpaca or SimBroker directly -- only this file. Swapping AlpacaBroker for
SimBroker (or a future second broker) is a config-line change, zero
changes to agents/execution_agent.py. See architecture doc, Phase 6.
"""
from __future__ import annotations

from typing import Literal, Optional, Protocol

from pydantic import BaseModel

OrderSide = Literal["buy", "sell"]
OrderStatus = Literal["filled", "pending", "rejected"]


class OrderResult(BaseModel):
    order_id: str
    ticker: str
    side: OrderSide
    qty: float
    status: OrderStatus
    filled_avg_price: Optional[float] = None
    raw: dict = {}


class BrokerPosition(BaseModel):
    ticker: str
    qty: float
    market_value: float
    avg_entry_price: float


class AccountInfo(BaseModel):
    equity: float
    cash: float
    buying_power: float


class Broker(Protocol):
    def submit_order(
        self, ticker: str, side: OrderSide, qty: float,
        client_order_id: Optional[str] = None,
    ) -> OrderResult: ...

    def get_positions(self) -> dict[str, BrokerPosition]: ...

    def get_account(self) -> AccountInfo: ...
