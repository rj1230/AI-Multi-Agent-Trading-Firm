"""
Regression tests for backtest performance metrics.
"""

from __future__ import annotations

import pytest

from backtest.metrics import (
    buy_and_hold_return,
    closed_trade_count,
    max_drawdown,
    sharpe_ratio,
    summarize,
    total_return,
    win_rate,
)

# ============================================================
# Win rate
# ============================================================


def test_win_rate_returns_none_when_no_trades_exist() -> None:
    assert win_rate([]) is None


def test_win_rate_returns_none_when_no_trades_are_closed() -> None:
    trades = [
        {"ticker": "AAPL", "side": "buy", "qty": 10},
        {"ticker": "MSFT", "side": "buy", "qty": 5},
    ]

    assert win_rate(trades) is None


def test_win_rate_uses_only_realized_pnl() -> None:
    trades = [
        {"ticker": "AAPL", "pnl": 100.0},
        {"ticker": "MSFT", "pnl": -50.0},
        {"ticker": "GOOGL", "side": "buy", "qty": 5},
    ]

    assert win_rate(trades) == pytest.approx(0.5)


def test_win_rate_counts_zero_pnl_as_not_a_win() -> None:
    trades = [
        {"ticker": "AAPL", "pnl": 100.0},
        {"ticker": "MSFT", "pnl": 0.0},
        {"ticker": "GOOGL", "pnl": -50.0},
    ]

    assert win_rate(trades) == pytest.approx(1 / 3)


# ============================================================
# Closed trade count
# ============================================================


def test_closed_trade_count_only_counts_realized_pnl() -> None:
    trades = [
        {"ticker": "AAPL", "pnl": 100.0},
        {"ticker": "MSFT", "pnl": -50.0},
        {"ticker": "GOOGL", "side": "buy", "qty": 10},
    ]

    assert closed_trade_count(trades) == 2


def test_closed_trade_count_returns_zero_when_no_realized_pnl() -> None:
    trades = [
        {"ticker": "AAPL", "side": "buy", "qty": 10},
        {"ticker": "MSFT", "side": "buy", "qty": 5},
    ]

    assert closed_trade_count(trades) == 0


# ============================================================
# Returns
# ============================================================


def test_total_return() -> None:
    equity_curve = [100_000.0, 101_000.0, 105_000.0]

    assert total_return(equity_curve) == pytest.approx(0.05)


def test_total_return_handles_insufficient_curve() -> None:
    assert total_return([]) == 0.0
    assert total_return([100_000.0]) == 0.0


def test_buy_and_hold_return() -> None:
    benchmark_curve = [100_000.0, 102_000.0, 110_000.0]

    assert buy_and_hold_return(benchmark_curve) == pytest.approx(0.10)


def test_buy_and_hold_return_handles_insufficient_curve() -> None:
    assert buy_and_hold_return([]) == 0.0
    assert buy_and_hold_return([100_000.0]) == 0.0


# ============================================================
# Drawdown
# ============================================================


def test_max_drawdown() -> None:
    equity_curve = [
        100_000.0,
        110_000.0,
        105_000.0,
        99_000.0,
        108_000.0,
    ]

    assert max_drawdown(equity_curve) == pytest.approx(
        (99_000.0 - 110_000.0) / 110_000.0
    )


def test_max_drawdown_returns_zero_for_empty_curve() -> None:
    assert max_drawdown([]) == 0.0


# ============================================================
# Sharpe ratio
# ============================================================


def test_sharpe_ratio_returns_zero_for_insufficient_data() -> None:
    assert sharpe_ratio([]) == 0.0
    assert sharpe_ratio([100_000.0]) == 0.0


def test_sharpe_ratio_returns_zero_for_zero_volatility() -> None:
    equity_curve = [100_000.0] * 10

    assert sharpe_ratio(equity_curve) == 0.0


# ============================================================
# summarize()
# ============================================================


def test_summarize_with_no_closed_trades() -> None:
    equity_curve = [
        100_000.0,
        99_500.0,
        99_750.0,
    ]

    benchmark_curve = [
        100_000.0,
        101_000.0,
        102_000.0,
    ]

    trades = [
        {"ticker": "AAPL", "side": "buy", "qty": 10},
    ]

    metrics = summarize(
        equity_curve,
        trades,
        benchmark_curve,
    )

    # Critical regression:
    # no realized trades must NOT be reported as 0% win rate.
    assert metrics["win_rate"] is None

    assert metrics["closed_trades"] == 0

    assert metrics["num_trades"] == 1

    assert metrics["total_return"] == pytest.approx(-0.0025)

    assert metrics["buy_and_hold_return"] == pytest.approx(0.02)

    assert metrics["excess_return_vs_buy_and_hold"] == pytest.approx(-0.0225)


def test_summarize_with_closed_trades() -> None:
    equity_curve = [
        100_000.0,
        101_000.0,
        102_000.0,
    ]

    benchmark_curve = [
        100_000.0,
        100_500.0,
        101_000.0,
    ]

    trades = [
        {"ticker": "AAPL", "pnl": 100.0},
        {"ticker": "MSFT", "pnl": -50.0},
        {"ticker": "GOOGL", "pnl": 25.0},
    ]

    metrics = summarize(
        equity_curve,
        trades,
        benchmark_curve,
    )

    assert metrics["win_rate"] == pytest.approx(2 / 3)

    assert metrics["closed_trades"] == 3

    assert metrics["num_trades"] == 3

    assert metrics["total_return"] == pytest.approx(0.02)

    assert metrics["buy_and_hold_return"] == pytest.approx(0.01)

    assert metrics["excess_return_vs_buy_and_hold"] == pytest.approx(0.01)
