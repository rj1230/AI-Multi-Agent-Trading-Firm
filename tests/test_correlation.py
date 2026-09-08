import pandas as pd
import pytest
from datetime import datetime, timedelta, timezone

from data_sources.schemas import OHLCVBar, OHLCVSeries, DataSourceMode
from portfolio.correlation import returns_from_ohlcv, update_correlation_matrix, check_correlation


def _make_series(ticker: str, closes: list[float]) -> OHLCVSeries:
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    bars = [
        OHLCVBar(
            timestamp=start + timedelta(days=i),
            open=c, high=c + 1, low=c - 1, close=c, volume=1000,
        )
        for i, c in enumerate(closes)
    ]
    as_of = bars[-1].timestamp if bars else start
    return OHLCVSeries(ticker=ticker, bars=bars, source="test", mode=DataSourceMode.LIVE, as_of=as_of)


def test_returns_from_ohlcv_drops_first_nan_row():
    series = _make_series("AAPL", [100, 101, 99, 102])
    returns = returns_from_ohlcv(series)
    assert len(returns) == 3  # 4 bars -> 3 pct-change returns
    assert not returns.isna().any()


def test_returns_from_empty_series_is_empty():
    series = _make_series("AAPL", [])
    returns = returns_from_ohlcv(series)
    assert returns.empty


def test_two_tickers_moving_identically_are_perfectly_correlated():
    closes = [100 + i for i in range(10)]  # steady uptrend
    aapl = _make_series("AAPL", closes)
    msft = _make_series("MSFT", closes)  # identical pattern
    matrix = update_correlation_matrix({
        "AAPL": returns_from_ohlcv(aapl),
        "MSFT": returns_from_ohlcv(msft),
    })
    assert matrix.loc["AAPL", "MSFT"] == pytest.approx(1.0, abs=1e-6)


def test_check_correlation_flags_above_threshold():
    matrix = pd.DataFrame({"AAPL": [1.0, 0.85], "MSFT": [0.85, 1.0]}, index=["AAPL", "MSFT"])
    result = check_correlation("AAPL", ["MSFT"], matrix, max_correlation=0.7)
    assert result["flagged"] is True
    assert result["against"] == "MSFT"
    assert result["max_correlation"] == pytest.approx(0.85)


def test_check_correlation_passes_below_threshold():
    matrix = pd.DataFrame({"AAPL": [1.0, 0.3], "MSFT": [0.3, 1.0]}, index=["AAPL", "MSFT"])
    result = check_correlation("AAPL", ["MSFT"], matrix, max_correlation=0.7)
    assert result["flagged"] is False


def test_check_correlation_missing_ticker_fails_open():
    matrix = pd.DataFrame({"MSFT": [1.0]}, index=["MSFT"])
    result = check_correlation("AAPL", ["MSFT"], matrix, max_correlation=0.7)
    assert result["flagged"] is False
    assert result["max_correlation"] == 0.0


def test_check_correlation_empty_matrix_fails_open():
    result = check_correlation("AAPL", [], pd.DataFrame(), max_correlation=0.7)
    assert result["flagged"] is False
