"""
ChartAgent (Phase 4).

Deliberately zero LLM calls, per the doc's explicit guidance: technical
pattern detection is a formula, not a judgment call, so this stays plain
code. Computes three indicators and combines them with a simple, auditable
rule into {direction, confidence, rationale}.

Indicators:
  - RSI(14): momentum / overbought-oversold
  - SMA(10) vs SMA(50) crossover: trend direction
  - Volume spike: latest volume vs its 20-bar average

Empty-data handling: if there aren't enough bars for a given indicator's
window, that indicator is skipped (not defaulted to a fake value) and
noted in the rationale -- mirrors NewsAgent's zero-articles handling.

compute_atr() was added for Phase 5/6 wiring: RiskAgent's position sizing
needs Average True Range, and this is the module that already owns bar-
level indicator math -- see graph/nodes.py for why it's called separately
from run_chart_agent() rather than folded into Signal's output.
"""

from __future__ import annotations

from data_sources import fetch_ohlcv
from data_sources.schemas import OHLCVBar
from graph.state import Signal

RSI_WINDOW = 14
SMA_FAST = 10
SMA_SLOW = 50
VOLUME_LOOKBACK = 20
VOLUME_SPIKE_MULTIPLIER = 1.5
ATR_WINDOW = 14


def _closes(bars: list[OHLCVBar]) -> list[float]:
    return [b.close for b in bars]


def _volumes(bars: list[OHLCVBar]) -> list[int]:
    return [b.volume for b in bars]


def compute_rsi(closes: list[float], window: int = RSI_WINDOW) -> float | None:
    if len(closes) < window + 1:
        return None

    gains, losses = [], []
    for i in range(1, len(closes)):
        change = closes[i] - closes[i - 1]
        gains.append(max(change, 0.0))
        losses.append(max(-change, 0.0))

    avg_gain = sum(gains[-window:]) / window
    avg_loss = sum(losses[-window:]) / window

    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return 100.0 - (100.0 / (1.0 + rs))


def compute_sma(closes: list[float], window: int) -> float | None:
    if len(closes) < window:
        return None
    return sum(closes[-window:]) / window


def compute_volume_spike(
    volumes: list[int], lookback: int = VOLUME_LOOKBACK
) -> bool | None:
    if len(volumes) < lookback + 1:
        return None
    avg = sum(volumes[-lookback - 1 : -1]) / lookback
    if avg == 0:
        return None
    return volumes[-1] >= avg * VOLUME_SPIKE_MULTIPLIER


def compute_atr(bars: list[OHLCVBar], window: int = ATR_WINDOW) -> float | None:
    """
    Average True Range over `window` periods. True Range for a bar is the
    largest of: high-low, |high - prev_close|, |low - prev_close| -- this
    is what makes ATR react to gaps, not just intrabar range.

    Returns None (not a fake 0.0) when there aren't enough bars, same
    empty-data convention as compute_rsi/compute_sma above -- a caller
    must not silently treat "no data" as "zero volatility".
    """
    if len(bars) < window + 1:
        return None

    true_ranges = []
    for i in range(1, len(bars)):
        high, low = bars[i].high, bars[i].low
        prev_close = bars[i - 1].close
        true_range = max(
            high - low,
            abs(high - prev_close),
            abs(low - prev_close),
        )
        true_ranges.append(true_range)

    return sum(true_ranges[-window:]) / window


def _combine(
    rsi: float | None,
    sma_fast: float | None,
    sma_slow: float | None,
    volume_spike: bool | None,
) -> Signal:
    """Simple, auditable combination rule -- every direction call must be
    traceable to specific indicator values in the rationale."""
    votes = []  # list of ("bullish"|"bearish"|"neutral", weight)
    notes = []

    if rsi is not None:
        if rsi >= 70:
            votes.append(("bearish", 0.5))
            notes.append(f"RSI={rsi:.1f} (overbought)")
        elif rsi <= 30:
            votes.append(("bullish", 0.5))
            notes.append(f"RSI={rsi:.1f} (oversold)")
        else:
            notes.append(f"RSI={rsi:.1f} (neutral range)")
    else:
        notes.append("RSI unavailable (insufficient bars)")

    if sma_fast is not None and sma_slow is not None:
        if sma_fast > sma_slow:
            votes.append(("bullish", 0.5))
            notes.append(f"SMA{SMA_FAST}>{SMA_SLOW} (uptrend)")
        elif sma_fast < sma_slow:
            votes.append(("bearish", 0.5))
            notes.append(f"SMA{SMA_FAST}<{SMA_SLOW} (downtrend)")
        else:
            notes.append("SMA crossover flat")
    else:
        notes.append("SMA crossover unavailable (insufficient bars)")

    if volume_spike:
        notes.append("volume spike detected")
    elif volume_spike is False:
        notes.append("no volume spike")
    else:
        notes.append("volume spike unavailable (insufficient bars)")

    if not votes:
        return Signal(
            direction="neutral",
            confidence=0.0,
            rationale="; ".join(notes),
        )

    bullish_weight = sum(w for d, w in votes if d == "bullish")
    bearish_weight = sum(w for d, w in votes if d == "bearish")

    if bullish_weight > bearish_weight:
        direction = "bullish"
        confidence = min(bullish_weight / len(votes), 1.0)
    elif bearish_weight > bullish_weight:
        direction = "bearish"
        confidence = min(bearish_weight / len(votes), 1.0)
    else:
        direction = "neutral"
        confidence = 0.3

    return Signal(
        direction=direction, confidence=confidence, rationale="; ".join(notes)
    )


def run_chart_agent(ticker: str) -> Signal:
    """Core logic, decoupled from the graph node wrapper for direct unit
    testing without a TradingState."""
    series = fetch_ohlcv(ticker, lookback_days=max(SMA_SLOW, VOLUME_LOOKBACK) + 10)

    if series.is_empty:
        return Signal(
            direction="neutral", confidence=0.0, rationale="no OHLCV data available"
        )

    closes = _closes(series.bars)
    volumes = _volumes(series.bars)

    rsi = compute_rsi(closes)
    sma_fast = compute_sma(closes, SMA_FAST)
    sma_slow = compute_sma(closes, SMA_SLOW)
    volume_spike = compute_volume_spike(volumes)

    return _combine(rsi, sma_fast, sma_slow, volume_spike)
