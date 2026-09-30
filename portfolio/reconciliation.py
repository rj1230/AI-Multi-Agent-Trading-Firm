from __future__ import annotations

from broker.protocol import Broker
from portfolio.ledger import PortfolioLedger


def reconcile(
    broker: Broker,
    ledger: PortfolioLedger,
    *,
    tolerance: float = 1e-6,
) -> bool:
    """Return True when broker and local ledger agree on cash and positions."""

    broker_account = broker.get_account()
    broker_positions = broker.get_positions()

    ledger_positions = ledger.get_positions()

    if abs(broker_account.cash - ledger.get_cash()) > tolerance:
        return False

    if set(broker_positions) != set(ledger_positions):
        return False

    for ticker, broker_position in broker_positions.items():
        ledger_position = ledger_positions[ticker]

        if (
            abs(broker_position.qty - float(ledger_position.get("shares", 0.0)))
            > tolerance
        ):
            return False

    return True
