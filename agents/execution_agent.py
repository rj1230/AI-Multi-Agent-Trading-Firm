"""
ExecutionAgent: takes a risk-approved trade and places it through whichever
Broker is configured. This node is deliberately "boring" -- all the
interesting decisions (what to trade, how much, whether it's safe)
happened upstream in SignalMerger and RiskAgent. Its only job is faithfully
carrying out an already-approved instruction.

Retries once on failure (network blip, broker timeout), then falls back to
a logged hold rather than silently losing the trade or retrying forever.
The client_order_id passed to the broker makes that retry safe: see
broker/sim_broker.py and broker/alpaca_broker.py for how each broker
prevents a retried submission from becoming a double fill.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from agents.risk_agent import RiskDecision
from agents.signal_merger import MergedSignal
from broker.protocol import Broker, OrderResult


@dataclass
class ExecutionResult:
    ticker: str
    executed: bool
    order: Optional[OrderResult]
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
) -> ExecutionResult:
    if not risk_decision.approved:
        return ExecutionResult(
            ticker=risk_decision.ticker,
            executed=False,
            order=None,
            notes="RiskAgent did not approve this trade -- routed to HoldNode.",
        )

    side = _direction_to_side(merged_signal.direction)
    # Deterministic per-decision id: the same (approved) decision retried
    # produces the same client_order_id, which is what makes the retry
    # loop below safe against double-submission.
    client_order_id = (
        f"{risk_decision.ticker}-{merged_signal.direction}-{risk_decision.proposed_shares}"
    )

    last_error = "unknown error"
    for attempt in range(max_retries + 1):
        try:
            order = broker.submit_order(
                risk_decision.ticker, side, risk_decision.proposed_shares,
                client_order_id=client_order_id,
            )
        except Exception as exc:
            last_error = str(exc)
            continue

        if order.status == "rejected":
            last_error = order.raw.get("reason") or order.raw.get("error") or "rejected"
            continue

        return ExecutionResult(
            ticker=risk_decision.ticker,
            executed=True,
            order=order,
            notes=f"Order {order.order_id} {order.status} for {order.qty} shares of {order.ticker}.",
        )

    return ExecutionResult(
        ticker=risk_decision.ticker,
        executed=False,
        order=None,
        notes=(
            f"Execution failed after {max_retries + 1} attempt(s): {last_error}. "
            "Falling back to hold."
        ),
    )
