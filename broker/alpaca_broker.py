"""
AlpacaBroker: wraps alpaca-py's TradingClient for LIVE mode. Same
submit_order/get_positions/get_account signature as SimBroker (see
broker/protocol.py) -- ExecutionAgent never knows or cares which one it's
talking to.

client_order_id is forwarded to Alpaca on every order. This is the fix for
the doc's Phase 6 self-check ("if Alpaca times out mid-order, do you know
if it went through, or could it double-submit on retry?"): Alpaca dedups
orders by client_order_id server-side, so a retried submission with the
same id either returns the original order or is rejected as a duplicate --
either way, no double fill.
"""

from __future__ import annotations

import os

from broker.protocol import AccountInfo, BrokerPosition, OrderResult, OrderSide

try:
    from alpaca.trading.client import TradingClient
    from alpaca.trading.enums import OrderSide as AlpacaOrderSide
    from alpaca.trading.enums import TimeInForce
    from alpaca.trading.requests import MarketOrderRequest
except ImportError:
    TradingClient = None  # alpaca-py not installed -- see __init__ below


def _map_status(alpaca_status) -> str:
    """
    Map Alpaca's order lifecycle status into the broker protocol.

    Known lifecycle states are preserved rather than collapsed into
    ``rejected`` so downstream execution, accounting, and audit layers can
    distinguish a full fill from a partial fill, pending order, cancellation,
    or expiration.

    Unknown non-terminal states intentionally map to ``pending``.
    """

    value = (
        alpaca_status.value if hasattr(alpaca_status, "value") else str(alpaca_status)
    )

    if value == "filled":
        return "filled"

    if value == "partially_filled":
        return "partially_filled"

    if value == "canceled":
        return "canceled"

    if value == "expired":
        return "expired"

    if value == "rejected":
        return "rejected"

    return "pending"


class AlpacaBroker:
    def __init__(
        self,
        api_key: str | None = None,
        secret_key: str | None = None,
        paper: bool = True,
    ):
        if TradingClient is None:
            raise ImportError(
                "alpaca-py is not installed -- run: uv pip install alpaca-py"
            )

        self.client = TradingClient(
            api_key or os.environ["ALPACA_API_KEY"],
            secret_key or os.environ["ALPACA_SECRET_KEY"],
            paper=paper,
        )

    def submit_order(
        self,
        ticker: str,
        side: OrderSide,
        qty: float,
        client_order_id: str | None = None,
    ) -> OrderResult:
        alpaca_side = AlpacaOrderSide.BUY if side == "buy" else AlpacaOrderSide.SELL

        request = MarketOrderRequest(
            symbol=ticker,
            qty=qty,
            side=alpaca_side,
            time_in_force=TimeInForce.DAY,
            client_order_id=client_order_id,
        )

        try:
            order = self.client.submit_order(request)
        except Exception as exc:  # noqa: BLE001
            return OrderResult(
                order_id="",
                ticker=ticker,
                side=side,
                qty=qty,
                status="rejected",
                filled_qty=0.0,
                raw={"error": str(exc)},
            )

        return OrderResult(
            order_id=str(order.id),
            ticker=ticker,
            side=side,
            qty=qty,
            status=_map_status(order.status),
            filled_qty=float(order.filled_qty or 0.0),
            filled_avg_price=float(order.filled_avg_price)
            if order.filled_avg_price
            else None,
            raw={
                "alpaca_status": str(order.status),
                "client_order_id": order.client_order_id,
            },
        )

    def get_positions(self) -> dict[str, BrokerPosition]:
        positions = self.client.get_all_positions()

        return {
            p.symbol: BrokerPosition(
                ticker=p.symbol,
                qty=float(p.qty),
                market_value=float(p.market_value),
                avg_entry_price=float(p.avg_entry_price),
            )
            for p in positions
        }

    def get_account(self) -> AccountInfo:
        account = self.client.get_account()

        return AccountInfo(
            equity=float(account.equity),
            cash=float(account.cash),
            buying_power=float(account.buying_power),
        )
