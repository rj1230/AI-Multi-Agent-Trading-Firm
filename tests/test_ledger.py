from datetime import date, datetime, timedelta, timezone

import portfolio.ledger as ledger_module
from data_sources.schemas import OHLCVBar, OHLCVSeries, DataSourceMode
from portfolio.ledger import PortfolioLedger


def _series(ticker: str, close: float) -> OHLCVSeries:
    ts = datetime(2026, 1, 1, tzinfo=timezone.utc)
    bar = OHLCVBar(timestamp=ts, open=close, high=close + 1, low=close - 1, close=close, volume=1000)
    return OHLCVSeries(ticker=ticker, bars=[bar], source="test", mode=DataSourceMode.LIVE, as_of=ts)


def test_fresh_ledger_starts_at_configured_equity(tmp_path):
    ledger = PortfolioLedger(starting_equity=100_000.0, path=tmp_path / "ledger.json")
    snap = ledger.snapshot()
    assert snap.equity == 100_000.0
    assert snap.starting_equity == 100_000.0
    assert snap.positions == {}


def test_record_buy_updates_cash_and_positions(tmp_path, monkeypatch):
    monkeypatch.setattr(ledger_module, "fetch_ohlcv", lambda t, lookback_days=1: _series(t, 105.0))
    ledger = PortfolioLedger(starting_equity=10_000.0, path=tmp_path / "ledger.json")
    ledger.record_fill("AAPL", "buy", qty=10, price=100.0, sector="Tech")

    snap = ledger.snapshot()
    assert snap.equity == 10_000.0 - 1_000.0 + (10 * 105.0)  # cash spent + marked-to-market value
    assert snap.positions["AAPL"].shares == 10
    assert snap.positions["AAPL"].sector == "Tech"


def test_record_sell_reduces_position(tmp_path, monkeypatch):
    monkeypatch.setattr(ledger_module, "fetch_ohlcv", lambda t, lookback_days=1: _series(t, 100.0))
    ledger = PortfolioLedger(starting_equity=10_000.0, path=tmp_path / "ledger.json")
    ledger.record_fill("AAPL", "buy", qty=10, price=100.0, sector="Tech")
    ledger.record_fill("AAPL", "sell", qty=4, price=100.0, sector="Tech")

    snap = ledger.snapshot()
    assert snap.positions["AAPL"].shares == 6


def test_selling_entire_position_removes_it(tmp_path, monkeypatch):
    monkeypatch.setattr(ledger_module, "fetch_ohlcv", lambda t, lookback_days=1: _series(t, 100.0))
    ledger = PortfolioLedger(starting_equity=10_000.0, path=tmp_path / "ledger.json")
    ledger.record_fill("AAPL", "buy", qty=10, price=100.0, sector="Tech")
    ledger.record_fill("AAPL", "sell", qty=10, price=100.0, sector="Tech")

    snap = ledger.snapshot()
    assert "AAPL" not in snap.positions


def test_selling_unheld_ticker_raises(tmp_path):
    ledger = PortfolioLedger(starting_equity=10_000.0, path=tmp_path / "ledger.json")
    try:
        ledger.record_fill("AAPL", "sell", qty=1, price=100.0, sector="Tech")
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_ledger_persists_across_instances(tmp_path, monkeypatch):
    monkeypatch.setattr(ledger_module, "fetch_ohlcv", lambda t, lookback_days=1: _series(t, 100.0))
    path = tmp_path / "ledger.json"
    ledger1 = PortfolioLedger(starting_equity=10_000.0, path=path)
    ledger1.record_fill("AAPL", "buy", qty=5, price=100.0, sector="Tech")

    ledger2 = PortfolioLedger(starting_equity=10_000.0, path=path)  # re-opens same file
    snap = ledger2.snapshot()
    assert snap.positions["AAPL"].shares == 5


def test_session_starting_equity_rolls_forward_on_new_day(tmp_path, monkeypatch):
    monkeypatch.setattr(ledger_module, "fetch_ohlcv", lambda t, lookback_days=1: _series(t, 100.0))
    path = tmp_path / "ledger.json"
    ledger = PortfolioLedger(starting_equity=10_000.0, path=path)
    ledger._data["session_date"] = "2020-01-01"  # simulate an old session
    ledger._save()

    snap = ledger.snapshot()  # should detect "today" != "2020-01-01" and roll
    assert ledger._data["session_date"] == date.today().isoformat()
    assert snap.starting_equity == 10_000.0  # no positions -> equity unchanged
