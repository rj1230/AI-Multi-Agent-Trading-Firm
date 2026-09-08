import pytest
from broker.sim_broker import SimBroker


@pytest.fixture
def broker():
    return SimBroker(starting_cash=10_000.0, price_lookup=lambda ticker: 100.0)


def test_buy_fills_and_deducts_cash(broker):
    order = broker.submit_order("AAPL", "buy", 10)
    assert order.status == "filled"
    assert order.filled_avg_price == 100.0
    assert broker.cash == 9_000.0
    positions = broker.get_positions()
    assert positions["AAPL"].qty == 10
    assert positions["AAPL"].market_value == 1_000.0


def test_buy_rejected_on_insufficient_cash(broker):
    order = broker.submit_order("AAPL", "buy", 1000)  # costs 100,000 > 10,000 cash
    assert order.status == "rejected"
    assert order.raw["reason"] == "insufficient cash"
    assert broker.cash == 10_000.0  # unchanged


def test_sell_more_than_held_is_rejected(broker):
    broker.submit_order("AAPL", "buy", 10)
    order = broker.submit_order("AAPL", "sell", 20)
    assert order.status == "rejected"
    assert order.raw["reason"] == "insufficient shares"


def test_sell_reduces_position_and_returns_cash(broker):
    broker.submit_order("AAPL", "buy", 10)
    order = broker.submit_order("AAPL", "sell", 4)
    assert order.status == "filled"
    positions = broker.get_positions()
    assert positions["AAPL"].qty == 6
    assert broker.cash == pytest.approx(9_000.0 + 400.0)


def test_selling_entire_position_removes_it(broker):
    broker.submit_order("AAPL", "buy", 10)
    broker.submit_order("AAPL", "sell", 10)
    assert "AAPL" not in broker.get_positions()


def test_averaging_into_existing_position_updates_avg_entry_price():
    prices = iter([100.0, 200.0])
    broker = SimBroker(starting_cash=100_000.0, price_lookup=lambda t: next(prices))
    broker.submit_order("AAPL", "buy", 10)  # 10 @ 100
    broker.submit_order("AAPL", "buy", 10)  # 10 @ 200
    position = broker.get_positions()["AAPL"]
    assert position.qty == 20
    assert position.avg_entry_price == pytest.approx(150.0)


def test_duplicate_client_order_id_does_not_double_fill(broker):
    first = broker.submit_order("AAPL", "buy", 10, client_order_id="order-1")
    cash_after_first = broker.cash
    second = broker.submit_order("AAPL", "buy", 10, client_order_id="order-1")
    assert second.order_id == first.order_id
    assert broker.cash == cash_after_first  # not deducted twice
    assert broker.get_positions()["AAPL"].qty == 10  # not doubled


def test_get_account_reflects_cash_plus_positions(broker):
    broker.submit_order("AAPL", "buy", 10)  # spends 1000, holds 1000 in AAPL
    account = broker.get_account()
    assert account.equity == pytest.approx(10_000.0)  # cash + position value nets out
    assert account.cash == pytest.approx(9_000.0)
