"""
backtest/vbt_crosscheck.py

Cross-checks metrics.py's Sharpe/max-drawdown against vectorbt,
using the actual BacktestResult.equity_curve.

Usage:
    uv run python backtest/vbt_crosscheck.py
"""

from __future__ import annotations

import pandas as pd
import vectorbt as vbt

from backtest.runner import run_backtest
from backtest.metrics import sharpe_ratio, max_drawdown as own_max_drawdown


def crosscheck(equity_curve: list[tuple[str, float]]) -> None:
    dates = pd.to_datetime([d for d, _ in equity_curve])
    values = [v for _, v in equity_curve]
    series = pd.Series(values, index=dates).sort_index()

    # --- Raw (calendar-day) comparison ---
    own_sharpe_raw = sharpe_ratio(values)
    own_dd_raw = own_max_drawdown(values)

    vbt_returns_raw = series.pct_change().dropna()
    vbt_ret_acc_raw = vbt_returns_raw.vbt.returns(freq="D")

    print("=== RAW (calendar days, includes weekends) ===")
    print(
        f"own  sharpe: {own_sharpe_raw:.4f}   vbt sharpe: {vbt_ret_acc_raw.sharpe_ratio():.4f}"
    )
    print(
        f"own  max_dd: {own_dd_raw:.4f}       vbt max_dd: {vbt_ret_acc_raw.max_drawdown():.4f}"
    )

    # --- Weekday-filtered comparison (closer to a real trading calendar) ---
    weekday_series = series[series.index.dayofweek < 5]
    weekday_values = weekday_series.tolist()

    own_sharpe_wd = sharpe_ratio(weekday_values)
    own_dd_wd = own_max_drawdown(weekday_values)

    vbt_returns_wd = weekday_series.pct_change().dropna()
    vbt_ret_acc_wd = vbt_returns_wd.vbt.returns(freq="D")

    print("\n=== WEEKDAYS ONLY (approximates trading days) ===")
    print(
        f"own  sharpe: {own_sharpe_wd:.4f}   vbt sharpe: {vbt_ret_acc_wd.sharpe_ratio():.4f}"
    )
    print(
        f"own  max_dd: {own_dd_wd:.4f}       vbt max_dd: {vbt_ret_acc_wd.max_drawdown():.4f}"
    )

    print(
        f"\nRaw sample count: {len(values)}   Weekday sample count: {len(weekday_values)}"
    )


if __name__ == "__main__":
    result = run_backtest(
        tickers=["AAPL"],
        start_date="2026-01-01",
        end_date="2026-03-01",
        tick_delay_seconds=1.5,
    )

    # --- diagnostic block: paste this in ---
    outcomes = {}
    for entry in result.tick_log:
        outcomes[entry["outcome"]] = outcomes.get(entry["outcome"], 0) + 1

    print("Outcome counts:", outcomes)
    print("Num trades recorded:", len(result.trades))
    print("First 5 equity points:", result.equity_curve[:5])
    print("Last 5 equity points:", result.equity_curve[-5:])
    if result.trades:
        print("Sample trade:", result.trades[0])
    # --- end diagnostic block ---

    crosscheck(result.equity_curve)
