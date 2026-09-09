from datetime import datetime, timedelta, timezone

from agents.chart_agent import compute_atr
from data_sources.schemas import OHLCVBar


def _bars(closes: list[float]) -> list[OHLCVBar]:
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    return [
        OHLCVBar(timestamp=start + timedelta(days=i), open=c, high=c + 1, low=c - 1, close=c, volume=1000)
        for i, c in enumerate(closes)
    ]


def test_atr_none_with_insufficient_bars():
    bars = _bars([100, 101, 102])  # far fewer than window+1
    assert compute_atr(bars, window=14) is None


def test_atr_constant_true_range():
    # closes step by 1 each bar, high=close+1, low=close-1 -> every bar's
    # true range is exactly 2.0 (see test docstring math), so ATR must be
    # exactly 2.0 regardless of window placement.
    closes = [100 + i for i in range(16)]
    bars = _bars(closes)
    atr = compute_atr(bars, window=14)
    assert atr == 2.0


def test_atr_reacts_to_a_gap():
    closes = [100 + i for i in range(15)] + [130]  # big gap up on the last bar
    bars = _bars(closes)
    atr_with_gap = compute_atr(bars, window=14)
    atr_without_gap = compute_atr(bars[:-1], window=14)
    assert atr_with_gap > atr_without_gap
