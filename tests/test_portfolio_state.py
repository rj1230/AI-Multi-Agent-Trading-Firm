from datetime import datetime, timedelta, timezone

import portfolio.state as state_module
from data_sources.schemas import OHLCVBar, OHLCVSeries, DataSourceMode


def _make_series(ticker: str, closes: list[float]) -> OHLCVSeries:
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    bars = [
        OHLCVBar(timestamp=start + timedelta(days=i), open=c, high=c + 1, low=c - 1, close=c, volume=1000)
        for i, c in enumerate(closes)
    ]
    as_of = bars[-1].timestamp if bars else start
    return OHLCVSeries(ticker=ticker, bars=bars, source="test", mode=DataSourceMode.LIVE, as_of=as_of)


def test_build_correlation_matrix_calls_fetch_ohlcv_per_ticker(monkeypatch):
    fake_data = {
        "AAPL": _make_series("AAPL", [100, 102, 101, 105, 107]),
        "MSFT": _make_series("MSFT", [200, 199, 202, 198, 205]),
    }
    calls = []

    def fake_fetch_ohlcv(ticker, simulated_date=None, lookback_days=30):
        calls.append(ticker)
        return fake_data[ticker]

    monkeypatch.setattr(state_module, "fetch_ohlcv", fake_fetch_ohlcv)

    matrix = state_module.build_correlation_matrix(["AAPL", "MSFT"])

    assert calls == ["AAPL", "MSFT"]
    assert "AAPL" in matrix.columns and "MSFT" in matrix.columns


def test_build_correlation_matrix_skips_empty_series(monkeypatch):
    def fake_fetch_ohlcv(ticker, simulated_date=None, lookback_days=30):
        return _make_series(ticker, [])  # empty -- e.g. brand-new listing

    monkeypatch.setattr(state_module, "fetch_ohlcv", fake_fetch_ohlcv)

    matrix = state_module.build_correlation_matrix(["AAPL"])
    assert matrix.empty
