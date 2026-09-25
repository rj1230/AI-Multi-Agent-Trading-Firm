"""
Phase 1 — Backtest State Isolation
==================================

These tests protect the most important backtest invariant:

    Every new backtest starts from:
        Ledger  = starting cash + zero positions
        Broker  = starting cash + zero positions

A previous backtest must never leak portfolio positions or cash into
a new simulation.

The tests also configure a simulated date because the project's
historical data layer intentionally refuses to use wall-clock time
when TRADING_MODE=backtest.
"""

from datetime import UTC, datetime
from pathlib import Path

from broker.sim_broker import SimBroker
from data_sources import set_simulated_date
from portfolio.ledger import PortfolioLedger

STARTING_EQUITY = 100_000.0

TEST_SIMULATED_DATE = datetime(
    2024,
    9,
    20,
    23,
    59,
    tzinfo=UTC,
)


def configure_test_clock() -> None:
    """
    Configure a deterministic historical timestamp for tests.

    This prevents the historical data layer from falling back to
    real wall-clock time and protects against lookahead.
    """
    set_simulated_date(TEST_SIMULATED_DATE)


def test_fresh_ledger_starts_empty(
    tmp_path: Path,
):
    configure_test_clock()

    ledger_path = tmp_path / "backtest_ledger.json"

    ledger = PortfolioLedger(
        starting_equity=STARTING_EQUITY,
        path=ledger_path,
        fresh=True,
    )

    snapshot = ledger.snapshot()

    assert snapshot.equity == STARTING_EQUITY
    assert snapshot.starting_equity == STARTING_EQUITY
    assert snapshot.positions == {}

    assert ledger.get_cash() == STARTING_EQUITY
    assert ledger.get_positions() == {}


def test_fresh_ledger_does_not_inherit_previous_state(
    tmp_path: Path,
):
    configure_test_clock()

    ledger_path = tmp_path / "backtest_ledger.json"

    # --------------------------------------------------------------
    # Simulate an earlier backtest.
    # --------------------------------------------------------------

    old_ledger = PortfolioLedger(
        starting_equity=STARTING_EQUITY,
        path=ledger_path,
        fresh=True,
    )

    old_ledger.record_fill(
        ticker="AAPL",
        side="buy",
        qty=10,
        price=100.0,
        sector="Technology",
    )

    old_snapshot = old_ledger.snapshot()

    assert "AAPL" in old_snapshot.positions
    assert old_snapshot.positions["AAPL"].shares == 10

    # --------------------------------------------------------------
    # Start a completely new backtest using the same path.
    # --------------------------------------------------------------

    new_ledger = PortfolioLedger(
        starting_equity=STARTING_EQUITY,
        path=ledger_path,
        fresh=True,
    )

    new_snapshot = new_ledger.snapshot()

    assert new_snapshot.equity == STARTING_EQUITY
    assert new_snapshot.starting_equity == STARTING_EQUITY
    assert new_snapshot.positions == {}

    assert new_ledger.get_cash() == STARTING_EQUITY

    assert new_ledger.get_positions() == {}


def test_non_fresh_ledger_preserves_existing_state(
    tmp_path: Path,
):
    configure_test_clock()

    ledger_path = tmp_path / "portfolio_ledger.json"

    first = PortfolioLedger(
        starting_equity=STARTING_EQUITY,
        path=ledger_path,
        fresh=True,
    )

    first.record_fill(
        ticker="MSFT",
        side="buy",
        qty=5,
        price=200.0,
        sector="Technology",
    )

    second = PortfolioLedger(
        starting_equity=STARTING_EQUITY,
        path=ledger_path,
        fresh=False,
    )

    snapshot = second.snapshot()

    assert "MSFT" in snapshot.positions
    assert snapshot.positions["MSFT"].shares == 5


def test_fresh_sim_broker_starts_without_positions():
    broker = SimBroker(
        starting_cash=STARTING_EQUITY,
        price_lookup=lambda ticker: 100.0,
    )

    account = broker.get_account()
    positions = broker.get_positions()

    assert account.cash == STARTING_EQUITY
    assert positions == {}


def test_fresh_ledger_and_broker_start_consistently(
    tmp_path: Path,
):
    configure_test_clock()

    ledger = PortfolioLedger(
        starting_equity=STARTING_EQUITY,
        path=tmp_path / "ledger.json",
        fresh=True,
    )

    broker = SimBroker(
        starting_cash=STARTING_EQUITY,
        price_lookup=lambda ticker: 100.0,
    )

    ledger_snapshot = ledger.snapshot()
    broker_account = broker.get_account()

    assert ledger_snapshot.equity == STARTING_EQUITY
    assert ledger_snapshot.positions == {}

    assert broker_account.cash == STARTING_EQUITY
    assert broker.get_positions() == {}


def test_fresh_ledger_can_sell_only_after_buy(
    tmp_path: Path,
):
    configure_test_clock()

    ledger = PortfolioLedger(
        starting_equity=STARTING_EQUITY,
        path=tmp_path / "ledger.json",
        fresh=True,
    )

    ledger.record_fill(
        ticker="AAPL",
        side="buy",
        qty=10,
        price=100.0,
        sector="Technology",
    )

    snapshot_after_buy = ledger.snapshot()

    assert snapshot_after_buy.positions["AAPL"].shares == 10

    ledger.record_fill(
        ticker="AAPL",
        side="sell",
        qty=10,
        price=110.0,
        sector="Technology",
    )

    snapshot_after_sell = ledger.snapshot()

    assert snapshot_after_sell.positions == {}
    assert snapshot_after_sell.equity == 100_100.0
