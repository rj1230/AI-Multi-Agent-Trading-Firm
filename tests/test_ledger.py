import pandas as pd
import pytest

from portfolio.ledger import PortfolioLedger

# ======================================================================
# Initialization / Fresh State
# ======================================================================


def test_fresh_ledger_starts_empty(tmp_path):
    ledger = PortfolioLedger(
        starting_equity=100_000.0,
        path=tmp_path / "ledger.json",
        fresh=True,
    )

    assert ledger.get_cash() == 100_000.0
    assert ledger.get_positions() == {}
    assert ledger._data["realized_pnl"] == 0.0


def test_fresh_ledger_ignores_existing_state(tmp_path):
    path = tmp_path / "ledger.json"

    existing = PortfolioLedger(
        starting_equity=100_000.0,
        path=path,
        fresh=True,
    )

    existing.record_fill(
        "AAPL",
        "buy",
        qty=10,
        price=100.0,
        sector="Tech",
    )

    isolated = PortfolioLedger(
        starting_equity=50_000.0,
        path=path,
        fresh=True,
    )

    assert isolated.get_cash() == 50_000.0
    assert isolated.get_positions() == {}
    assert isolated._data["realized_pnl"] == 0.0


# ======================================================================
# Buy / Sell Accounting
# ======================================================================


def test_buy_updates_cash_and_position(tmp_path):
    ledger = PortfolioLedger(
        starting_equity=100_000.0,
        path=tmp_path / "ledger.json",
        fresh=True,
    )

    ledger.record_fill(
        "AAPL",
        "buy",
        qty=10,
        price=100.0,
        sector="Tech",
    )

    assert ledger.get_cash() == 99_000.0

    positions = ledger.get_positions()

    assert positions["AAPL"]["shares"] == 10
    assert positions["AAPL"]["sector"] == "Tech"
    assert positions["AAPL"]["average_cost"] == 100.0
    assert positions["AAPL"]["market_value"] == 1_000.0


def test_sell_reduces_position(tmp_path):
    ledger = PortfolioLedger(
        starting_equity=100_000.0,
        path=tmp_path / "ledger.json",
        fresh=True,
    )

    ledger.record_fill(
        "AAPL",
        "buy",
        qty=10,
        price=100.0,
        sector="Tech",
    )

    ledger.record_fill(
        "AAPL",
        "sell",
        qty=4,
        price=120.0,
        sector="Tech",
    )

    assert ledger.get_cash() == 99_480.0

    positions = ledger.get_positions()

    assert positions["AAPL"]["shares"] == 6
    assert positions["AAPL"]["average_cost"] == 100.0
    assert ledger._data["realized_pnl"] == 80.0


def test_selling_entire_position_removes_it(tmp_path):
    ledger = PortfolioLedger(
        starting_equity=100_000.0,
        path=tmp_path / "ledger.json",
        fresh=True,
    )

    ledger.record_fill(
        "AAPL",
        "buy",
        qty=10,
        price=100.0,
        sector="Tech",
    )

    ledger.record_fill(
        "AAPL",
        "sell",
        qty=10,
        price=120.0,
        sector="Tech",
    )

    assert ledger.get_positions() == {}
    assert ledger.get_cash() == 100_200.0
    assert ledger._data["realized_pnl"] == 200.0


def test_selling_unheld_position_raises(tmp_path):
    ledger = PortfolioLedger(
        starting_equity=100_000.0,
        path=tmp_path / "ledger.json",
        fresh=True,
    )

    with pytest.raises(ValueError, match="no position exists"):
        ledger.record_fill(
            "AAPL",
            "sell",
            qty=1,
            price=100.0,
            sector="Tech",
        )


def test_overselling_position_raises(tmp_path):
    ledger = PortfolioLedger(
        starting_equity=100_000.0,
        path=tmp_path / "ledger.json",
        fresh=True,
    )

    ledger.record_fill(
        "AAPL",
        "buy",
        qty=10,
        price=100.0,
        sector="Tech",
    )

    with pytest.raises(ValueError, match="only 10.0 shares held"):
        ledger.record_fill(
            "AAPL",
            "sell",
            qty=11,
            price=120.0,
            sector="Tech",
        )


# ======================================================================
# Persistence
# ======================================================================


def test_ledger_persists_and_reloads(tmp_path):
    path = tmp_path / "ledger.json"

    ledger = PortfolioLedger(
        starting_equity=100_000.0,
        path=path,
        fresh=True,
    )

    ledger.record_fill(
        "MSFT",
        "buy",
        qty=5,
        price=400.0,
        sector="Tech",
    )

    reloaded = PortfolioLedger(
        starting_equity=100_000.0,
        path=path,
    )

    assert reloaded.get_cash() == 98_000.0

    positions = reloaded.get_positions()

    assert positions["MSFT"]["shares"] == 5
    assert positions["MSFT"]["average_cost"] == 400.0
    assert reloaded._data["realized_pnl"] == 0.0


def test_session_starting_equity_is_preserved(tmp_path):
    path = tmp_path / "ledger.json"

    ledger = PortfolioLedger(
        starting_equity=100_000.0,
        path=path,
        fresh=True,
    )

    ledger.record_fill(
        "AAPL",
        "buy",
        qty=10,
        price=100.0,
        sector="Tech",
    )

    reloaded = PortfolioLedger(
        starting_equity=50_000.0,
        path=path,
    )

    assert reloaded._data["session_starting_equity"] == 100_000.0


def test_realized_pnl_persists_and_reloads(tmp_path):
    path = tmp_path / "ledger.json"

    ledger = PortfolioLedger(
        starting_equity=100_000.0,
        path=path,
        fresh=True,
    )

    ledger.record_fill(
        "AAPL",
        "buy",
        qty=10,
        price=100.0,
        sector="Tech",
    )

    ledger.record_fill(
        "AAPL",
        "sell",
        qty=5,
        price=140.0,
        sector="Tech",
    )

    assert ledger._data["realized_pnl"] == 200.0

    reloaded = PortfolioLedger(
        starting_equity=100_000.0,
        path=path,
    )

    assert reloaded._data["realized_pnl"] == 200.0


# ======================================================================
# Cost-Basis Regression Tests
# ======================================================================


def test_position_preserves_average_cost_after_mark_to_market(
    tmp_path,
    monkeypatch,
):
    """
    Regression test for position cost-basis preservation.

    Scenario:

        BUY 10 AAPL @ $100
        Market price later becomes $120

    Expected:

        shares       = 10
        market_value = $1200
        average_cost = $100

    Mark-to-market must update current market value without
    destroying historical acquisition cost.
    """

    ledger = PortfolioLedger(
        starting_equity=10_000.0,
        path=tmp_path / "ledger.json",
    )

    ledger.record_fill(
        "AAPL",
        "buy",
        qty=10,
        price=100.0,
        sector="Tech",
    )

    class FakeSeries:
        is_empty = False

        class Latest:
            close = 120.0

        latest = Latest()

    def fake_fetch(ticker, **kwargs):
        return FakeSeries()

    monkeypatch.setattr(
        "data_sources.fetch_ohlcv",
        fake_fetch,
    )

    monkeypatch.setattr(
        "portfolio.ledger.build_correlation_matrix",
        lambda tickers, lookback_days=30: pd.DataFrame(),
    )

    ledger.snapshot()

    position = ledger._data["positions"]["AAPL"]

    assert position["shares"] == 10
    assert position["market_value"] == 1_200.0
    assert position["average_cost"] == 100.0


def test_multiple_buys_preserve_weighted_average_cost_after_mark_to_market(
    tmp_path,
    monkeypatch,
):
    """
    Regression test for weighted average acquisition cost.

    Scenario:

        BUY 10 AAPL @ $100
        Market moves to $120
        BUY 10 AAPL @ $140

    Expected:

        shares       = 20
        average_cost = $120
        market_value = $2800

    The second buy must use historical acquisition cost rather than
    the mark-to-market value of the existing position.
    """

    ledger = PortfolioLedger(
        starting_equity=10_000.0,
        path=tmp_path / "ledger.json",
    )

    ledger.record_fill(
        "AAPL",
        "buy",
        qty=10,
        price=100.0,
        sector="Tech",
    )

    class FirstMarketSeries:
        is_empty = False

        class Latest:
            close = 120.0

        latest = Latest()

    def first_fetch(ticker, **kwargs):
        return FirstMarketSeries()

    monkeypatch.setattr(
        "data_sources.fetch_ohlcv",
        first_fetch,
    )

    monkeypatch.setattr(
        "portfolio.ledger.build_correlation_matrix",
        lambda tickers, lookback_days=30: pd.DataFrame(),
    )

    ledger.snapshot()

    ledger.record_fill(
        "AAPL",
        "buy",
        qty=10,
        price=140.0,
        sector="Tech",
    )

    position = ledger._data["positions"]["AAPL"]

    assert position["shares"] == 20
    assert position["average_cost"] == 120.0
    assert position["market_value"] == 2_800.0


def test_partial_sell_preserves_average_cost(
    tmp_path,
    monkeypatch,
):
    """
    Regression test for partial-sell cost-basis preservation.

    Scenario:

        BUY 10 AAPL @ $100
        BUY 10 AAPL @ $120
        SELL 5 AAPL @ $140

    Expected:

        remaining shares = 15
        average_cost     = $110

    A partial sale changes the number of shares held but must not
    change the historical average acquisition cost of the remaining
    position.
    """

    ledger = PortfolioLedger(
        starting_equity=10_000.0,
        path=tmp_path / "ledger.json",
    )

    ledger.record_fill(
        "AAPL",
        "buy",
        qty=10,
        price=100.0,
        sector="Tech",
    )

    ledger.record_fill(
        "AAPL",
        "buy",
        qty=10,
        price=120.0,
        sector="Tech",
    )

    class MarketSeries:
        is_empty = False

        class Latest:
            close = 140.0

        latest = Latest()

    def fake_fetch(ticker, **kwargs):
        return MarketSeries()

    monkeypatch.setattr(
        "data_sources.fetch_ohlcv",
        fake_fetch,
    )

    monkeypatch.setattr(
        "portfolio.ledger.build_correlation_matrix",
        lambda tickers, lookback_days=30: pd.DataFrame(),
    )

    ledger.record_fill(
        "AAPL",
        "sell",
        qty=5,
        price=140.0,
        sector="Tech",
    )

    position = ledger._data["positions"]["AAPL"]

    assert position["shares"] == 15
    assert position["average_cost"] == 110.0


# ======================================================================
# Realized P&L Regression Tests
# ======================================================================


def test_partial_sell_records_realized_pnl(tmp_path):
    """
    Regression test for realized P&L accounting.

    Scenario:

        BUY 10 AAPL @ $100
        SELL 5 AAPL @ $140

    Expected:

        realized_pnl = 5 * ($140 - $100) = $200
        remaining shares = 5
        remaining average_cost = $100

    Realized P&L must be calculated from historical acquisition cost,
    not from current market value.
    """

    ledger = PortfolioLedger(
        starting_equity=100_000.0,
        path=tmp_path / "ledger.json",
        fresh=True,
    )

    ledger.record_fill(
        "AAPL",
        "buy",
        qty=10,
        price=100.0,
        sector="Tech",
    )

    ledger.record_fill(
        "AAPL",
        "sell",
        qty=5,
        price=140.0,
        sector="Tech",
    )

    positions = ledger.get_positions()

    assert positions["AAPL"]["shares"] == 5
    assert positions["AAPL"]["average_cost"] == 100.0

    assert ledger._data["realized_pnl"] == 200.0


def test_weighted_average_cost_drives_realized_pnl(tmp_path):
    """
    Realized P&L must use the weighted average acquisition cost.

    Scenario:

        BUY 10 AAPL @ $100
        BUY 10 AAPL @ $120
        SELL 5 AAPL @ $140

    Weighted average cost:

        ((10 * 100) + (10 * 120)) / 20 = $110

    Expected realized P&L:

        5 * (140 - 110) = $150

    Remaining:

        shares       = 15
        average_cost = $110
    """

    ledger = PortfolioLedger(
        starting_equity=100_000.0,
        path=tmp_path / "ledger.json",
        fresh=True,
    )

    ledger.record_fill(
        "AAPL",
        "buy",
        qty=10,
        price=100.0,
        sector="Tech",
    )

    ledger.record_fill(
        "AAPL",
        "buy",
        qty=10,
        price=120.0,
        sector="Tech",
    )

    ledger.record_fill(
        "AAPL",
        "sell",
        qty=5,
        price=140.0,
        sector="Tech",
    )

    positions = ledger.get_positions()

    assert positions["AAPL"]["shares"] == 15
    assert positions["AAPL"]["average_cost"] == 110.0

    assert ledger._data["realized_pnl"] == 150.0


def test_multiple_sells_accumulate_realized_pnl(tmp_path):
    """
    Realized P&L must accumulate across multiple sell fills.

    Scenario:

        BUY 10 AAPL @ $100
        SELL 3 AAPL @ $120 -> +$60
        SELL 2 AAPL @ $80  -> -$40

    Expected:

        cumulative realized P&L = $20
        remaining shares       = 5
        remaining average cost = $100
    """

    ledger = PortfolioLedger(
        starting_equity=100_000.0,
        path=tmp_path / "ledger.json",
        fresh=True,
    )

    ledger.record_fill(
        "AAPL",
        "buy",
        qty=10,
        price=100.0,
        sector="Tech",
    )

    ledger.record_fill(
        "AAPL",
        "sell",
        qty=3,
        price=120.0,
        sector="Tech",
    )

    ledger.record_fill(
        "AAPL",
        "sell",
        qty=2,
        price=80.0,
        sector="Tech",
    )

    positions = ledger.get_positions()

    assert positions["AAPL"]["shares"] == 5
    assert positions["AAPL"]["average_cost"] == 100.0

    assert ledger._data["realized_pnl"] == 20.0
