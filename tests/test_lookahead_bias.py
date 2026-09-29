from __future__ import annotations

from datetime import UTC, datetime, timedelta

import data_sources.historical as historical_module
from data_sources.schemas import OHLCVBar


def test_historical_ohlcv_filters_future_bars(monkeypatch):
    """
    Regression test for OHLCV lookahead protection.

    The underlying historical source intentionally contains a future bar.

    Given simulated date T, historical replay must expose only:

        bar.timestamp <= T

    The future bar must never reach the backtest consumer.
    """

    simulated_date = datetime(
        2026,
        8,
        4,
        tzinfo=UTC,
    )

    future_date = simulated_date + timedelta(days=1)

    historical_bars = [
        OHLCVBar(
            timestamp=simulated_date - timedelta(days=1),
            open=149.0,
            high=151.0,
            low=148.0,
            close=150.0,
            volume=1_000_000,
        ),
        OHLCVBar(
            timestamp=future_date,
            open=199.0,
            high=201.0,
            low=198.0,
            close=200.0,
            volume=1_000_000,
        ),
    ]

    monkeypatch.setattr(
        historical_module,
        "_load_or_fetch_full_history",
        lambda ticker: historical_bars,
    )

    series = historical_module.fetch_ohlcv(
        ticker="AAPL",
        simulated_date=simulated_date,
        lookback_days=30,
    )

    assert len(series.bars) == 1

    assert series.bars[0].timestamp == (simulated_date - timedelta(days=1))

    assert all(bar.timestamp <= simulated_date for bar in series.bars)

    assert all(bar.timestamp != future_date for bar in series.bars)

    assert series.as_of == simulated_date
