"""
Technical analysis functions used by ChartAgent: moving averages, RSI, ATR,
and simple pattern detection (support/resistance, MA crossovers).
"""


def sma(prices: list[float], window: int) -> list[float]:
    raise NotImplementedError


def rsi(prices: list[float], window: int = 14) -> list[float]:
    raise NotImplementedError


def atr(ohlcv: list[dict], window: int = 14) -> list[float]:
    raise NotImplementedError


def detect_patterns(ohlcv: list[dict]) -> list[str]:
    """Returns detected pattern names, e.g. ['golden_cross', 'support_bounce']."""
    raise NotImplementedError
