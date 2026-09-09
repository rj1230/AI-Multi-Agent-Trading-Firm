"""
Performance metrics from a backtest's equity curve / trade log.
"""

from __future__ import annotations

import math


def sharpe_ratio(equity_curve: list[float], risk_free_rate: float = 0.0) -> float:
    """Annualized Sharpe from a daily equity curve. Needs >= 2 points to
    compute a single return, and non-zero volatility to avoid a
    divide-by-zero on a perfectly flat (e.g. all-hold) curve."""
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


def win_rate(trades: list[dict]) -> float:
    """Fraction of trades with pnl > 0. Trades without a "pnl" key are
    skipped (e.g. this runner's raw executed-trade log doesn't compute
    per-trade P&L yet -- pass a trade list with "pnl" populated, or treat
    0.0 as "not yet computable" rather than a real 0% win rate)."""
    scored = [t for t in trades if "pnl" in t]
    if not scored:
        return 0.0
    wins = sum(1 for t in scored if t["pnl"] > 0)
    return wins / len(scored)


def max_drawdown(equity_curve: list[float]) -> float:
    """Largest peak-to-trough decline, as a negative fraction (e.g.
    -0.15 = a 15% drawdown from the running peak)."""
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


def summarize(
    equity_curve: list[float], trades: list[dict], buy_and_hold_curve: list[float]
) -> dict:
    total_return = (
        (equity_curve[-1] / equity_curve[0]) - 1
        if len(equity_curve) >= 2 and equity_curve[0]
        else 0.0
    )
    bh_return = (
        (buy_and_hold_curve[-1] / buy_and_hold_curve[0]) - 1
        if len(buy_and_hold_curve) >= 2 and buy_and_hold_curve[0]
        else 0.0
    )
    return {
        "sharpe_ratio": sharpe_ratio(equity_curve),
        "win_rate": win_rate(trades),
        "max_drawdown": max_drawdown(equity_curve),
        "total_return": total_return,
        "buy_and_hold_return": bh_return,
        "excess_return_vs_buy_and_hold": total_return - bh_return,
        "num_trades": len(trades),
    }
