"""
Phase 9 cross-check: runs a real backtest, then computes Sharpe ratio and
max drawdown two independent ways on the IDENTICAL equity curve --
backtest/metrics.py's own formulas, and vectorbt's. If they agree
(within a small tolerance for methodology differences -- see notes
below), that's real evidence the hand-written metrics math is correct,
not just "didn't crash."

This does NOT compare trading strategies or re-run vectorbt's own
backtesting engine -- that would be comparing two different decision
processes, not validating metrics math. It takes the one true equity
curve your system actually produced and asks: do two different Sharpe/
drawdown implementations agree on what that curve means?

DEBUG LOGGING: enabled at INFO level below so that any logger.info()
calls inside backtest/runner.py, orchestrator/tick_orchestrator.py, or
elsewhere in the call chain actually print instead of being silently
discarded (Python's logging module does nothing until a handler is
configured somewhere in the process).

Run: python run_vectorbt_crosscheck.py
"""

from __future__ import annotations

import logging

import pandas as pd
import vectorbt as vbt

from backtest.metrics import max_drawdown, sharpe_ratio
from backtest.runner import run_backtest

logging.basicConfig(level=logging.INFO, format="%(message)s")

TICKERS = ["AAPL"]
START_DATE = "2026-07-01"
END_DATE = "2026-07-31"


def main() -> None:
    print(f"Running real backtest: {TICKERS} from {START_DATE} to {END_DATE}...")
    result = run_backtest(
        tickers=TICKERS,
        start_date=START_DATE,
        end_date=END_DATE,
        starting_equity=100_000.0,
        ledger_path="backtest/crosscheck_ledger.json",
    )

    if len(result.equity_curve) < 2:
        print(
            f"Only {len(result.equity_curve)} equity points produced -- "
            "not enough to compute Sharpe/drawdown meaningfully. "
            "Check that OHLCV/news caches cover this date range."
        )
        return

    dates = [d for d, _ in result.equity_curve]
    equity_values = [e for _, e in result.equity_curve]

    print(
        f"\n{len(equity_values)} equity points, {len(result.trades)} trades executed."
    )
    print(f"Equity: {equity_values[0]:.2f} -> {equity_values[-1]:.2f}")

    # --- Your own metrics.py ---
    own_sharpe = sharpe_ratio(equity_values)
    own_drawdown = max_drawdown(equity_values)

    # --- vectorbt, on the identical curve ---
    equity_series = pd.Series(equity_values, index=pd.to_datetime(dates))
    returns = equity_series.pct_change().dropna()

    if returns.std() == 0:
        print(
            "\nNo variation in returns (flat equity curve -- likely zero "
            "trades executed this window). Sharpe is undefined/zero for "
            "both implementations by construction; nothing to cross-check."
        )
        vbt_sharpe = 0.0
    else:
        vbt_sharpe = returns.vbt.returns(freq="1D").sharpe_ratio()

    vbt_drawdown = abs(equity_series.vbt.drawdown().min())

    print("\n--- Sharpe Ratio ---")
    print(f"  backtest/metrics.py : {own_sharpe:.4f}")
    print(f"  vectorbt            : {vbt_sharpe:.4f}")
    print(f"  difference          : {abs(own_sharpe - vbt_sharpe):.4f}")

    print("\n--- Max Drawdown ---")
    print(f"  backtest/metrics.py : {own_drawdown:.4f}")
    print(f"  vectorbt            : {vbt_drawdown:.4f}")
    print(f"  difference          : {abs(own_drawdown - vbt_drawdown):.4f}")

    # Small tolerance, not exact-equality: vectorbt's Sharpe uses a
    # slightly different annualization convention in some versions, and
    # floating point sums over different code paths never match bit-for-
    # bit. This checks "the same story," not "the same float."
    sharpe_ok = abs(own_sharpe - vbt_sharpe) < 0.05
    drawdown_ok = abs(own_drawdown - vbt_drawdown) < 0.005

    print("\n--- Verdict ---")
    print(f"  Sharpe agrees within tolerance   : {sharpe_ok}")
    print(f"  Drawdown agrees within tolerance : {drawdown_ok}")

    if not (sharpe_ok and drawdown_ok):
        print(
            "\nDisagreement found -- do not treat this as passing. Check "
            "which formula differs (annualization factor, population vs "
            "sample std dev, simple vs log returns) before trusting either."
        )


if __name__ == "__main__":
    main()
