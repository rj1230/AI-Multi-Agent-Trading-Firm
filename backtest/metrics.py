"""
Computes performance metrics from a backtest ledger/equity curve.
"""


def sharpe_ratio(equity_curve: list[float], risk_free_rate: float = 0.0) -> float:
    raise NotImplementedError


def win_rate(trades: list[dict]) -> float:
    raise NotImplementedError


def max_drawdown(equity_curve: list[float]) -> float:
    raise NotImplementedError


def summarize(equity_curve: list[float], trades: list[dict], buy_and_hold_curve: list[float]) -> dict:
    """Returns Sharpe, win rate, max drawdown, total return vs. buy-and-hold."""
    raise NotImplementedError
