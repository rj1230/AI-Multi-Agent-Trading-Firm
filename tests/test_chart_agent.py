"""
ChartAgent tests. Indicator math is tested directly (pure functions, no
mocking needed); run_chart_agent's data-fetch integration is tested with
fetch_ohlcv mocked.
"""

from datetime import datetime, timedelta, timezone
from unittest.mock import patch

from agents.chart_agent import (
    compute_rsi,
    compute_sma,
    compute_volume_spike,
    run_chart_agent,
)
from data_sources.schemas import DataSourceMode, OHLCVBar, OHLCVSeries


def _make_bars(closes: list[float], volumes: list[int] | None = None) -> list[OHLCVBar]:
    now = datetime.now(timezone.utc)
    volumes = volumes or [1000] * len(closes)
    return [
        OHLCVBar(
            timestamp=now - timedelta(days=len(closes) - i),
            open=c,
            high=c + 1,
            low=c - 1,
            close=c,
            volume=v,
        )
        for i, (c, v) in enumerate(zip(closes, volumes))
    ]


def test_rsi_insufficient_data_returns_none():
    assert compute_rsi([100, 101, 102], window=14) is None


def test_rsi_all_gains_is_100():
    closes = [100 + i for i in range(20)]  # strictly increasing
    rsi = compute_rsi(closes, window=14)
    assert rsi == 100.0


def test_rsi_all_losses_is_0():
    closes = [100 - i for i in range(20)]  # strictly decreasing
    rsi = compute_rsi(closes, window=14)
    assert rsi == 0.0


def test_sma_insufficient_data_returns_none():
    assert compute_sma([1, 2, 3], window=10) is None


def test_sma_computes_average_of_window():
    closes = [10] * 5 + [20] * 5  # last 5 are all 20
    assert compute_sma(closes, window=5) == 20.0


def test_volume_spike_detected():
    volumes = [1000] * 20 + [5000]  # last bar 5x the trailing average
    assert compute_volume_spike(volumes, lookback=20) is True


def test_volume_spike_not_detected():
    volumes = [1000] * 21
    assert compute_volume_spike(volumes, lookback=20) is False


def test_run_chart_agent_empty_data_returns_neutral():
    empty = OHLCVSeries(
        ticker="AAPL",
        bars=[],
        source="none",
        mode=DataSourceMode.LIVE,
        as_of=datetime.now(timezone.utc),
    )
    with patch("agents.chart_agent.fetch_ohlcv", return_value=empty):
        signal = run_chart_agent("AAPL")

    assert signal.direction == "neutral"
    assert signal.confidence == 0.0
    assert "no ohlcv data" in signal.rationale.lower()


def test_run_chart_agent_uptrend_produces_bullish():
    # 60 bars, uptrending with periodic pullbacks (2 up days, 1 down day)
    # so RSI stays in a moderate range instead of pinning to overbought —
    # a pure straight-line uptrend triggers RSI's overbought/reversal vote,
    # which correctly ties against SMA's bullish vote and nets to neutral.
    # This pattern isolates the "clear uptrend, moderate momentum" case
    # that should net bullish.
    closes = [100.0]
    for i in range(59):
        closes.append(closes[-1] - 1.2 if i % 3 == 2 else closes[-1] + 0.9)
    bars = _make_bars(closes)
    series = OHLCVSeries(
        ticker="AAPL",
        bars=bars,
        source="test",
        mode=DataSourceMode.LIVE,
        as_of=datetime.now(timezone.utc),
    )
    with patch("agents.chart_agent.fetch_ohlcv", return_value=series):
        signal = run_chart_agent("AAPL")

    assert signal.direction == "bullish"
    assert signal.confidence > 0
    assert "SMA" in signal.rationale
