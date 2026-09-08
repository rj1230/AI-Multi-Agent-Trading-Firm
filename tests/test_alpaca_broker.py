from unittest.mock import MagicMock

import pytest

import broker.alpaca_broker as alpaca_broker_module
from broker.alpaca_broker import AlpacaBroker
from alpaca.trading.enums import OrderStatus


@pytest.fixture
def broker(monkeypatch):
    """AlpacaBroker with a mocked TradingClient -- validates our
    request/response conversion logic without hitting the network or
    needing real API keys."""
    fake_client_instance = MagicMock()
    fake_client_class = MagicMock(return_value=fake_client_instance)
    monkeypatch.setattr(alpaca_broker_module, "TradingClient", fake_client_class)

    b = AlpacaBroker(api_key="test", secret_key="test", paper=True)
    b._fake_client = fake_client_instance  # stash for assertions
    return b


def test_submit_order_maps_filled_status(broker):
    fake_order = MagicMock(
        id="abc-123", status=OrderStatus.FILLED, filled_avg_price="150.25",
        client_order_id="AAPL-bullish-10",
    )
    broker._fake_client.submit_order.return_value = fake_order

    result = broker.submit_order("AAPL", "buy", 10, client_order_id="AAPL-bullish-10")

    assert result.status == "filled"
    assert result.order_id == "abc-123"
    assert result.filled_avg_price == pytest.approx(150.25)


def test_submit_order_maps_rejected_status(broker):
    fake_order = MagicMock(id="abc-124", status=OrderStatus.REJECTED, filled_avg_price=None, client_order_id="x")
    broker._fake_client.submit_order.return_value = fake_order

    result = broker.submit_order("AAPL", "buy", 10)
    assert result.status == "rejected"


def test_submit_order_maps_pending_style_status_to_pending(broker):
    fake_order = MagicMock(id="abc-125", status=OrderStatus.ACCEPTED, filled_avg_price=None, client_order_id="x")
    broker._fake_client.submit_order.return_value = fake_order

    result = broker.submit_order("AAPL", "buy", 10)
    assert result.status == "pending"


def test_submit_order_network_exception_becomes_rejected_result(broker):
    broker._fake_client.submit_order.side_effect = ConnectionError("timed out")
    result = broker.submit_order("AAPL", "buy", 10)
    assert result.status == "rejected"
    assert "timed out" in result.raw["error"]


def test_get_positions_converts_alpaca_positions(broker):
    fake_position = MagicMock(symbol="AAPL", qty="10", market_value="1500.0", avg_entry_price="150.0")
    broker._fake_client.get_all_positions.return_value = [fake_position]

    positions = broker.get_positions()
    assert positions["AAPL"].qty == 10.0
    assert positions["AAPL"].market_value == 1500.0


def test_get_account_converts_alpaca_account(broker):
    fake_account = MagicMock(equity="105000.0", cash="20000.0", buying_power="40000.0")
    broker._fake_client.get_account.return_value = fake_account

    account = broker.get_account()
    assert account.equity == 105000.0
    assert account.cash == 20000.0
    assert account.buying_power == 40000.0


def test_import_error_message_when_alpaca_py_missing(monkeypatch):
    monkeypatch.setattr(alpaca_broker_module, "TradingClient", None)
    with pytest.raises(ImportError, match="alpaca-py is not installed"):
        AlpacaBroker(api_key="x", secret_key="y")
