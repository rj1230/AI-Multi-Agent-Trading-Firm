"""
Performance metrics from a backtest's equity curve and trade log.

Metric semantics:

Equity-curve metrics:
- Sharpe ratio is calculated from the strategy equity curve.
- Max drawdown is calculated from the strategy equity curve.
- Total return is calculated from the strategy equity curve.
- Buy-and-hold return is calculated from the supplied benchmark curve.
- Excess return is strategy return minus benchmark return.

Trade metrics:
- ``trades`` represents execution events/fills.
- ``closed_trades`` represents completed position round trips.
- New closed-trade records use ``realized_pnl``.
- Legacy trade records may use ``pnl`` and remain supported.
- Win rate is calculated only from trades containing numeric realized P&L.
- Profit factor is gross profit divided by absolute gross loss.
- Realized P&L is calculated from completed trades.
- When no completed trades exist, trade metrics return ``None`` where
  appropriate rather than misleadingly reporting zero performance.

The metrics layer is deterministic and does not make trading decisions.
"""

from __future__ import annotations

import math


def sharpe_ratio(
    equity_curve: list[float],
    risk_free_rate: float = 0.0,
) -> float:
    """
    Annualized Sharpe ratio from a daily equity curve.

    Returns:
        0.0 when there are insufficient observations or zero volatility.
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


def _pnl_value(trade: dict) -> float | None:
    """
    Extract the realized P&L from a trade.

    New schema:
        realized_pnl

    Legacy schema:
        pnl

    ``realized_pnl`` takes precedence when both fields exist.

    Returns None for missing, non-numeric, boolean, or non-finite values.
    """

    if "realized_pnl" in trade:
        value = trade.get("realized_pnl")
    else:
        value = trade.get("pnl")

    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(float(value))
    ):
        return None

    return float(value)


def scored_closed_trades(
    closed_trades: list[dict],
) -> list[dict]:
    """
    Return closed trades containing valid realized P&L.

    Both the new ``realized_pnl`` field and the legacy ``pnl`` field
    are supported.
    """

    return [trade for trade in closed_trades if _pnl_value(trade) is not None]


def win_rate(
    closed_trades: list[dict],
) -> float | None:
    """
    Calculate win rate from completed trades.

    A trade is a win when realized P&L is greater than zero.

    Zero-P&L trades are counted as non-wins.

    Returns:
        Fraction of profitable completed trades.
        None when no scored completed trades exist.
    """

    scored = scored_closed_trades(closed_trades)

    if not scored:
        return None

    wins = sum(1 for trade in scored if _pnl_value(trade) > 0)

    return wins / len(scored)


def closed_trade_count(
    closed_trades: list[dict],
) -> int:
    """
    Count completed trades with valid realized P&L.
    """

    return len(scored_closed_trades(closed_trades))


def realized_pnl(
    closed_trades: list[dict],
) -> float | None:
    """
    Calculate cumulative realized P&L from completed trades.

    Returns None when no scored completed trades exist.
    """

    scored = scored_closed_trades(closed_trades)

    if not scored:
        return None

    return sum(_pnl_value(trade) for trade in scored)


def gross_profit(
    closed_trades: list[dict],
) -> float | None:
    """
    Sum all positive realized P&L values.

    Returns None when there are no scored completed trades.
    """

    scored = scored_closed_trades(closed_trades)

    if not scored:
        return None

    return sum(_pnl_value(trade) for trade in scored if _pnl_value(trade) > 0)


def gross_loss(
    closed_trades: list[dict],
) -> float | None:
    """
    Sum all negative realized P&L values.

    The result is negative.

    Example:
        -100 + -50 = -150
    """

    scored = scored_closed_trades(closed_trades)

    if not scored:
        return None

    return sum(_pnl_value(trade) for trade in scored if _pnl_value(trade) < 0)


def profit_factor(
    closed_trades: list[dict],
) -> float | None:
    """
    Calculate profit factor.

    Formula:

        gross_profit / abs(gross_loss)

    Returns None when:
    - no completed trades exist, or
    - no losing trades exist.

    A zero-loss sample is intentionally not represented as infinity.
    """

    scored = scored_closed_trades(closed_trades)

    if not scored:
        return None

    total_profit = sum(_pnl_value(trade) for trade in scored if _pnl_value(trade) > 0)

    total_loss = sum(_pnl_value(trade) for trade in scored if _pnl_value(trade) < 0)

    if total_loss == 0:
        return None

    return total_profit / abs(total_loss)


def average_winning_trade(
    closed_trades: list[dict],
) -> float | None:
    """Calculate average realized P&L of profitable completed trades."""

    scored = scored_closed_trades(closed_trades)

    wins = [_pnl_value(trade) for trade in scored if _pnl_value(trade) > 0]

    if not wins:
        return None

    return sum(wins) / len(wins)


def average_losing_trade(
    closed_trades: list[dict],
) -> float | None:
    """Calculate average realized P&L of losing completed trades."""

    scored = scored_closed_trades(closed_trades)

    losses = [_pnl_value(trade) for trade in scored if _pnl_value(trade) < 0]

    if not losses:
        return None

    return sum(losses) / len(losses)


def max_drawdown(
    equity_curve: list[float],
) -> float:
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


def total_return(
    equity_curve: list[float],
) -> float:
    """Calculate total return from an equity curve."""

    if len(equity_curve) < 2:
        return 0.0

    if equity_curve[0] == 0:
        return 0.0

    return (equity_curve[-1] / equity_curve[0]) - 1


def buy_and_hold_return(
    buy_and_hold_curve: list[float],
) -> float:
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
    closed_trades: list[dict] | None = None,
) -> dict:
    """
    Produce the complete performance-metrics payload.

    Args:
        equity_curve:
            Strategy equity values.

        trades:
            Execution events/fills.

        buy_and_hold_curve:
            Benchmark equity values.

        closed_trades:
            Completed position round trips.

    Backward compatibility:

    When ``closed_trades`` is omitted, ``trades`` is treated as the
    scored trade collection. This preserves the original API used by
    the existing metrics tests and older callers.

    New backtest code should explicitly provide ``closed_trades``.
    """

    strategy_return = total_return(equity_curve)

    benchmark_return = buy_and_hold_return(buy_and_hold_curve)

    completed = closed_trades if closed_trades is not None else trades

    return {
        # Equity-curve metrics.
        "sharpe_ratio": sharpe_ratio(equity_curve),
        "max_drawdown": max_drawdown(equity_curve),
        "total_return": strategy_return,
        "buy_and_hold_return": benchmark_return,
        "excess_return_vs_buy_and_hold": (strategy_return - benchmark_return),
        # Execution metrics.
        "num_trades": len(trades),
        # Completed-trade metrics.
        "closed_trades": closed_trade_count(completed),
        "win_rate": win_rate(completed),
        "realized_pnl": realized_pnl(completed),
        "gross_profit": gross_profit(completed),
        "gross_loss": gross_loss(completed),
        "profit_factor": profit_factor(completed),
        "average_winning_trade": average_winning_trade(completed),
        "average_losing_trade": average_losing_trade(completed),
    }
