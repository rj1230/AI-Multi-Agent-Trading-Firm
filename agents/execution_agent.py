"""
ExecutionAgent: takes a risk-approved trade and places it through whichever
Broker is configured. This node is deliberately "boring" -- all the
interesting decisions (what to trade, how much, whether it's safe)
happened upstream in SignalMerger and RiskAgent. Its only job is faithfully
carrying out an already-approved instruction.

Retries once on submission failure (network blip, broker timeout), then
falls back to a logged hold rather than silently losing the trade or
retrying forever.

Broker lifecycle states are preserved:

    filled
    partially_filled
    pending
    rejected
    canceled
    expired

Only ``filled`` is considered a fully successful execution.

The client_order_id is scoped to the logical execution context
(run_id + tick_id + ticker + side + quantity), making retries of the same
logical execution idempotent while allowing legitimate executions from
different ticks or runs.
"""

from __future__ import annotations

from dataclasses import dataclass

from agents.risk_agent import RiskDecision
from agents.signal_merger import MergedSignal
from broker.protocol import Broker, OrderResult


@dataclass
class ExecutionResult:
    ticker: str
    executed: bool
    order: OrderResult | None
    notes: str


def _direction_to_side(direction: str) -> str:
    # v1 core loop is long-only: bullish -> buy, bearish -> sell (i.e.
    # close/reduce an existing long). Short-selling isn't in scope here --
    # extending this mapping is a v2 decision, not an oversight.
    return "buy" if direction == "bullish" else "sell"


def run_execution_agent(
    risk_decision: RiskDecision,
    merged_signal: MergedSignal,
    broker: Broker,
    max_retries: int = 1,
    *,
    run_id: str | None = None,
    tick_id: int | None = None,
) -> ExecutionResult:
    if not risk_decision.approved:
        return ExecutionResult(
            ticker=risk_decision.ticker,
            executed=False,
            order=None,
            notes="RiskAgent did not approve this trade -- routed to HoldNode.",
        )

    side = _direction_to_side(merged_signal.direction)

    # Scope the client order ID to the logical execution context.
    #
    # Same run + same tick + same order:
    #     -> same client_order_id -> idempotent retry
    #
    # Different tick or different run:
    #     -> different client_order_id -> legitimate new execution
    client_order_id = (
        f"{run_id or 'standalone'}:"
        f"{tick_id if tick_id is not None else 'na'}:"
        f"{risk_decision.ticker}:"
        f"{side}:"
        f"{risk_decision.proposed_shares}"
    )

    last_error = "unknown error"

    for attempt in range(max_retries + 1):
        try:
            order = broker.submit_order(
                risk_decision.ticker,
                side,
                risk_decision.proposed_shares,
                client_order_id=client_order_id,
            )
        except Exception as exc:  # noqa: BLE001
            # A submission exception is the uncertain failure case.
            #
            # Retry using the exact same client_order_id so the broker can
            # deduplicate the logical order if the first submission actually
            # reached the broker before the exception occurred.
            last_error = str(exc)
            continue

        # --------------------------------------------------------------
        # Broker lifecycle result received successfully.
        #
        # Do NOT retry based on lifecycle status. The broker has accepted
        # and identified the order, so its lifecycle must be preserved.
        #
        # In particular, a partial fill must not be converted into a full
        # execution and must not be submitted again automatically.
        # --------------------------------------------------------------

        if order.status == "filled":
            return ExecutionResult(
                ticker=risk_decision.ticker,
                executed=True,
                order=order,
                notes=(
                    f"Order {order.order_id} filled "
                    f"{order.filled_qty} of {order.qty} shares "
                    f"of {order.ticker}."
                ),
            )

        if order.status == "partially_filled":
            return ExecutionResult(
                ticker=risk_decision.ticker,
                executed=False,
                order=order,
                notes=(
                    f"Order {order.order_id} partially filled "
                    f"{order.filled_qty} of {order.qty} shares "
                    f"of {order.ticker}."
                ),
            )

        if order.status == "pending":
            return ExecutionResult(
                ticker=risk_decision.ticker,
                executed=False,
                order=order,
                notes=(
                    f"Order {order.order_id} is pending for "
                    f"{order.qty} shares of {order.ticker}."
                ),
            )

        if order.status == "canceled":
            return ExecutionResult(
                ticker=risk_decision.ticker,
                executed=False,
                order=order,
                notes=(
                    f"Order {order.order_id} was canceled after requesting "
                    f"{order.qty} shares of {order.ticker}."
                ),
            )

        if order.status == "expired":
            return ExecutionResult(
                ticker=risk_decision.ticker,
                executed=False,
                order=order,
                notes=(
                    f"Order {order.order_id} expired after requesting "
                    f"{order.qty} shares of {order.ticker}."
                ),
            )

        if order.status == "rejected":
            reason = order.raw.get("reason") or order.raw.get("error") or "rejected"

            return ExecutionResult(
                ticker=risk_decision.ticker,
                executed=False,
                order=order,
                notes=(f"Order {order.order_id or 'unknown'} rejected: {reason}."),
            )

        # Defensive fallback for an unexpected broker status.
        #
        # The protocol currently constrains status values, but preserving
        # the order is safer than treating an unknown status as a fill.
        return ExecutionResult(
            ticker=risk_decision.ticker,
            executed=False,
            order=order,
            notes=(
                f"Order {order.order_id} returned unexpected status "
                f"{order.status!r}; no accounting mutation performed."
            ),
        )

    return ExecutionResult(
        ticker=risk_decision.ticker,
        executed=False,
        order=None,
        notes=(
            f"Execution submission failed after "
            f"{max_retries + 1} attempt(s): "
            f"{last_error}. Falling back to hold."
        ),
    )
