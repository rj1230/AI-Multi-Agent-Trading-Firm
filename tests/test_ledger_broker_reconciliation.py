import pytest

from broker.sim_broker import SimBroker
from portfolio.ledger import PortfolioLedger
from portfolio.reconciliation import reconcile


def test_successful_buy_keeps_ledger_and_broker_consistent(tmp_path):
    price = 100.0
    qty = 10

    broker = SimBroker(
        starting_cash=10_000.0,
        price_lookup=lambda ticker: price,
    )

    ledger = PortfolioLedger(
        starting_equity=10_000.0,
        path=tmp_path / "ledger.json",
    )

    order = broker.submit_order("AAPL", "buy", qty)

    assert order.status == "filled"
    assert order.filled_avg_price == pytest.approx(price)

    ledger.record_fill(
        ticker="AAPL",
        side=order.side,
        qty=order.qty,
        price=order.filled_avg_price,
        sector="Tech",
    )

    broker_position = broker.get_positions()["AAPL"]

    assert broker.cash == pytest.approx(9_000.0)
    assert ledger._data["cash"] == pytest.approx(9_000.0)

    assert broker_position.qty == pytest.approx(10)
    assert ledger._data["positions"]["AAPL"]["shares"] == pytest.approx(10)


def test_reconciliation_passes_after_buy(tmp_path):
    broker = SimBroker(
        starting_cash=10_000.0,
        price_lookup=lambda ticker: 100.0,
    )

    ledger = PortfolioLedger(
        starting_equity=10_000.0,
        path=tmp_path / "ledger.json",
    )

    order = broker.submit_order("AAPL", "buy", 10)

    ledger.record_fill(
        ticker="AAPL",
        side=order.side,
        qty=order.filled_qty,
        price=order.filled_avg_price,
        sector="Tech",
    )

    assert reconcile(broker, ledger)


def test_ledger_state_survives_restart(tmp_path):
    path = tmp_path / "ledger.json"

    ledger = PortfolioLedger(
        starting_equity=10_000.0,
        path=path,
    )

    ledger.record_fill(
        ticker="AAPL",
        side="buy",
        qty=10,
        price=100.0,
        sector="Tech",
    )

    restarted = PortfolioLedger(
        starting_equity=10_000.0,
        path=path,
    )

    assert restarted.get_cash() == pytest.approx(9_000.0)
    assert restarted.get_positions()["AAPL"]["shares"] == pytest.approx(10)


def test_successful_sell_keeps_ledger_and_broker_consistent(tmp_path):
    price = 100.0

    broker = SimBroker(
        starting_cash=10_000.0,
        price_lookup=lambda ticker: price,
    )

    ledger = PortfolioLedger(
        starting_equity=10_000.0,
        path=tmp_path / "ledger.json",
    )

    buy = broker.submit_order("AAPL", "buy", 10)

    assert buy.status == "filled"

    ledger.record_fill(
        ticker="AAPL",
        side=buy.side,
        qty=buy.qty,
        price=buy.filled_avg_price,
        sector="Tech",
    )

    sell = broker.submit_order("AAPL", "sell", 4)

    assert sell.status == "filled"

    ledger.record_fill(
        ticker="AAPL",
        side=sell.side,
        qty=sell.qty,
        price=sell.filled_avg_price,
        sector="Tech",
    )

    broker_position = broker.get_positions()["AAPL"]

    assert broker.cash == pytest.approx(9_400.0)
    assert ledger._data["cash"] == pytest.approx(9_400.0)

    assert broker_position.qty == pytest.approx(6)
    assert ledger._data["positions"]["AAPL"]["shares"] == pytest.approx(6)


def test_full_sell_removes_position_from_both(tmp_path):
    price = 100.0

    broker = SimBroker(
        starting_cash=10_000.0,
        price_lookup=lambda ticker: price,
    )

    ledger = PortfolioLedger(
        starting_equity=10_000.0,
        path=tmp_path / "ledger.json",
    )

    buy = broker.submit_order("AAPL", "buy", 10)

    assert buy.status == "filled"

    ledger.record_fill(
        ticker="AAPL",
        side=buy.side,
        qty=buy.qty,
        price=buy.filled_avg_price,
        sector="Tech",
    )

    sell = broker.submit_order("AAPL", "sell", 10)

    assert sell.status == "filled"

    ledger.record_fill(
        ticker="AAPL",
        side=sell.side,
        qty=sell.qty,
        price=sell.filled_avg_price,
        sector="Tech",
    )

    assert "AAPL" not in broker.get_positions()
    assert "AAPL" not in ledger._data["positions"]

    assert broker.cash == pytest.approx(10_000.0)
    assert ledger._data["cash"] == pytest.approx(10_000.0)
