"""
SimBroker: deterministic in-memory broker for BACKTEST mode.

Backtest market orders are resolved synchronously:

    submit_order()
        -> filled
        -> rejected

There is intentionally no artificial pending state in the simulator.
A live broker may return accepted/pending orders, but the backtest broker
must provide deterministic execution semantics.

Idempotency:
    If the same client_order_id is submitted more than once, the broker
    returns the original OrderResult without applying the order again.

This makes ExecutionAgent retries safe and prevents duplicate portfolio
mutations.

Slippage:
    The simulator applies deterministic percentage-point slippage expressed
    in basis points (bps).

    BUY:
        execution_price = market_price * (1 + slippage)

    SELL:
        execution_price = market_price * (1 - slippage)

The price returned by price_lookup() remains the reference market price.
filled_avg_price contains the actual execution price after slippage.
"""

from __future__ import annotations

from collections.abc import Callable

from broker.protocol import AccountInfo, BrokerPosition, OrderResult, OrderSide


class SimBroker:
    """Deterministic in-memory broker used by the backtest engine."""

    def __init__(
        self,
        starting_cash: float,
        price_lookup: Callable[[str], float],
        *,
        slippage_bps: float = 0.0,
    ):
        if starting_cash < 0:
            raise ValueError("starting_cash must be non-negative")

        if slippage_bps < 0:
            raise ValueError("slippage_bps must be non-negative")

        self.cash = starting_cash
        self.price_lookup = price_lookup
        self.slippage_bps = float(slippage_bps)

        self._positions: dict[str, BrokerPosition] = {}

        # Logical client-order identity -> immutable execution result.
        #
        # A retry with the same client_order_id must return the original
        # result and must not mutate cash or positions again.
        self._orders_by_client_id: dict[str, OrderResult] = {}

        # Broker order identity -> execution result.
        #
        # This gives the simulator a minimal in-process order lifecycle
        # representation without introducing asynchronous behavior.
        self._orders_by_order_id: dict[str, OrderResult] = {}

        self._next_order_id = 1

    # ------------------------------------------------------------------
    # Order submission
    # ------------------------------------------------------------------

    def submit_order(
        self,
        ticker: str,
        side: OrderSide,
        qty: float,
        client_order_id: str | None = None,
    ) -> OrderResult:
        """
        Submit one market order.

        Backtest semantics are synchronous:

            valid order + sufficient resources -> filled
            invalid/unsupported order -> rejected

        Repeated submission with the same client_order_id is idempotent.
        """

        ticker = ticker.strip().upper()

        if not ticker:
            raise ValueError("ticker must not be empty")

        if side not in {"buy", "sell"}:
            raise ValueError(f"Unsupported order side: {side!r}")

        if qty <= 0:
            raise ValueError("qty must be greater than zero")

        # --------------------------------------------------------------
        # Idempotency boundary.
        #
        # This check MUST happen before price lookup or portfolio
        # mutation. A retry therefore cannot double-fill.
        # --------------------------------------------------------------
        if client_order_id:
            existing = self._orders_by_client_id.get(client_order_id)

            if existing is not None:
                return existing

        # --------------------------------------------------------------
        # Allocate broker order identity only for a genuinely new order.
        # --------------------------------------------------------------
        order_id = f"SIM-{self._next_order_id}"
        self._next_order_id += 1

        price = self.price_lookup(ticker)

        if price <= 0:
            result = OrderResult(
                order_id=order_id,
                ticker=ticker,
                side=side,
                qty=qty,
                status="rejected",
                raw={"reason": "invalid market price"},
            )
            return self._record_order(result, client_order_id)

        # --------------------------------------------------------------
        # Apply deterministic slippage to the reference market price.
        #
        # price_lookup() remains the reference price.
        # execution_price is the actual simulated fill price.
        # --------------------------------------------------------------
        slippage = self.slippage_bps / 10_000

        if side == "buy":
            execution_price = price * (1 + slippage)
        else:
            execution_price = price * (1 - slippage)

        cost = execution_price * qty

        # --------------------------------------------------------------
        # Apply the actual execution price to the fill.
        # --------------------------------------------------------------
        if side == "buy":
            result = self._fill_buy(
                order_id=order_id,
                ticker=ticker,
                qty=qty,
                price=execution_price,
                cost=cost,
            )
        else:
            result = self._fill_sell(
                order_id=order_id,
                ticker=ticker,
                qty=qty,
                price=execution_price,
                cost=cost,
            )

        # Record BOTH successful and rejected results so retries remain
        # idempotent.
        return self._record_order(result, client_order_id)

    # ------------------------------------------------------------------
    # Order persistence / idempotency
    # ------------------------------------------------------------------

    def _record_order(
        self,
        result: OrderResult,
        client_order_id: str | None,
    ) -> OrderResult:
        """
        Record the broker result.

        Both successful and rejected results are stored against the
        client_order_id so a retry of the same logical order receives
        the same outcome.
        """

        self._orders_by_order_id[result.order_id] = result

        if client_order_id:
            self._orders_by_client_id[client_order_id] = result

        return result

    # ------------------------------------------------------------------
    # Fill handling
    # ------------------------------------------------------------------

    def _fill_buy(
        self,
        order_id: str,
        ticker: str,
        qty: float,
        price: float,
        cost: float,
    ) -> OrderResult:
        if cost > self.cash:
            return OrderResult(
                order_id=order_id,
                ticker=ticker,
                side="buy",
                qty=qty,
                status="rejected",
                raw={"reason": "insufficient cash"},
            )

        self.cash -= cost

        existing = self._positions.get(ticker)

        if existing:
            new_qty = existing.qty + qty

            new_avg = (existing.avg_entry_price * existing.qty + cost) / new_qty

            self._positions[ticker] = BrokerPosition(
                ticker=ticker,
                qty=new_qty,
                market_value=new_qty * price,
                avg_entry_price=new_avg,
            )
        else:
            self._positions[ticker] = BrokerPosition(
                ticker=ticker,
                qty=qty,
                market_value=cost,
                avg_entry_price=price,
            )

        return OrderResult(
            order_id=order_id,
            ticker=ticker,
            side="buy",
            qty=qty,
            status="filled",
            filled_qty=qty,
            filled_avg_price=price,
        )

    def _fill_sell(
        self,
        order_id: str,
        ticker: str,
        qty: float,
        price: float,
        cost: float,
    ) -> OrderResult:
        existing = self._positions.get(ticker)

        held_qty = existing.qty if existing else 0.0

        if qty > held_qty:
            return OrderResult(
                order_id=order_id,
                ticker=ticker,
                side="sell",
                qty=qty,
                status="rejected",
                raw={"reason": "insufficient shares"},
            )

        self.cash += cost

        remaining = held_qty - qty

        if remaining <= 0:
            del self._positions[ticker]
        else:
            self._positions[ticker] = BrokerPosition(
                ticker=ticker,
                qty=remaining,
                market_value=remaining * price,
                avg_entry_price=existing.avg_entry_price,
            )

        return OrderResult(
            order_id=order_id,
            ticker=ticker,
            side="sell",
            qty=qty,
            status="filled",
            filled_qty=qty,
            filled_avg_price=price,
        )

    # ------------------------------------------------------------------
    # Broker state
    # ------------------------------------------------------------------

    def get_positions(self) -> dict[str, BrokerPosition]:
        """Return a snapshot of current broker positions."""
        return dict(self._positions)

    def get_account(self) -> AccountInfo:
        """Return current cash, buying power, and marked-to-market equity."""

        equity = self.cash + sum(
            position.market_value for position in self._positions.values()
        )

        return AccountInfo(
            equity=equity,
            cash=self.cash,
            buying_power=self.cash,
        )
