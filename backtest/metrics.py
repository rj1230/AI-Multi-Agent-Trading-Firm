"""
Performance metrics from a backtest's equity curve / trade log.

Metric semantics:
- Sharpe is calculated from the strategy equity curve.
- Win rate is calculated only from trades containing a numeric ``pnl``.
- If there are no scored/closed trades, win rate is ``None`` rather than
  misleadingly reporting 0%.
- Buy-and-hold metrics are calculated from the benchmark curve supplied
  by the backtest runner.
"""

from __future__ import annotations

import math
from typing import Optional


def sharpe_ratio(
    equity_curve: list[float],
    risk_free_rate: float = 0.0,
) -> float:
    """
    Annualized Sharpe ratio from a daily equity curve.

    Returns 0.0 when there are insufficient observations or zero volatility.
    """
    if len(equity_curve) < 2:
        return 0.0

    daily_returns = [
        (equity_curve[i] / equity_curve[i - 1]) - 1
        for i in range(1, len(equity_curve))
        if equity_curve[i - 1] != 0
    ]

    if not daily_returns:
        return 0.0

    mean_return = sum(daily_returns) / len(daily_returns)

    variance = sum((r - mean_return) ** 2 for r in daily_returns) / len(daily_returns)

    std_dev = math.sqrt(variance)

    if std_dev == 0:
        return 0.0

    daily_rf = risk_free_rate / 252

    return ((mean_return - daily_rf) / std_dev) * math.sqrt(252)


def win_rate(trades: list[dict]) -> Optional[float]:
    """
    Calculate win rate from trades with realized P&L.

    A trade is considered scored only when it contains a numeric ``pnl``.

    Returns:
        float:
            Fraction of scored trades with positive P&L.
        None:
            When no scored/closed trades exist yet.

    ``None`` is intentional. It distinguishes:

        no closed trades

    from:

        closed trades exist, but 0% were profitable.
    """
    scored = [trade for trade in trades if isinstance(trade.get("pnl"), (int, float))]

    if not scored:
        return None

    wins = sum(1 for trade in scored if trade["pnl"] > 0)

    return wins / len(scored)


def closed_trade_count(trades: list[dict]) -> int:
    """
    Count trades for which realized P&L is available.
    """
    return sum(1 for trade in trades if isinstance(trade.get("pnl"), (int, float)))


def max_drawdown(equity_curve: list[float]) -> float:
    """
    Largest peak-to-trough decline.

    Returned as a negative fraction:
        -0.15 == 15% drawdown.
    """
    if not equity_curve:
        return 0.0

    peak = equity_curve[0]
    worst = 0.0

    for value in equity_curve:
        peak = max(peak, value)

        if peak > 0:
            drawdown = (value - peak) / peak
            worst = min(worst, drawdown)

    return worst


def total_return(equity_curve: list[float]) -> float:
    """Calculate total return from an equity curve."""
    if len(equity_curve) < 2:
        return 0.0

    if equity_curve[0] == 0:
        return 0.0

    return (equity_curve[-1] / equity_curve[0]) - 1


def buy_and_hold_return(buy_and_hold_curve: list[float]) -> float:
    """Calculate total return from the benchmark curve."""
    if len(buy_and_hold_curve) < 2:
        return 0.0

    if buy_and_hold_curve[0] == 0:
        return 0.0

    return (buy_and_hold_curve[-1] / buy_and_hold_curve[0]) - 1


def summarize(
    equity_curve: list[float],
    trades: list[dict],
    buy_and_hold_curve: list[float],
) -> dict:
    """
    Produce the complete performance-metrics payload.
    """
    strategy_return = total_return(equity_curve)
    benchmark_return = buy_and_hold_return(buy_and_hold_curve)

    closed_trades = closed_trade_count(trades)

    return {
        "sharpe_ratio": sharpe_ratio(equity_curve),
        "win_rate": win_rate(trades),
        "closed_trades": closed_trades,
        "max_drawdown": max_drawdown(equity_curve),
        "total_return": strategy_return,
        "buy_and_hold_return": benchmark_return,
        "excess_return_vs_buy_and_hold": (strategy_return - benchmark_return),
        "num_trades": len(trades),
    }
