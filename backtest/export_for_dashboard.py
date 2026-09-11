"""
Writes a BacktestResult to CSVs the dashboard reads. Run this after any
backtest you want reflected in the Backtest Report tab.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from backtest.runner import BacktestResult

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def export_result(result: BacktestResult, project_root: Path = PROJECT_ROOT) -> None:
    data_dir = project_root / "data"
    data_dir.mkdir(exist_ok=True)

    pd.DataFrame(result.equity_curve, columns=["date", "equity"]).to_csv(
        data_dir / "backtest_results.csv", index=False
    )
    pd.DataFrame(result.trades).to_csv(data_dir / "backtest_trades.csv", index=False)
    pd.DataFrame(result.tick_log).to_csv(data_dir / "backtest_tick_log.csv", index=False)
    print(f"Exported {len(result.equity_curve)} days, {len(result.trades)} trades, "
          f"{len(result.tick_log)} tick-log entries to {data_dir}")