"""
harness/runner.py — Phase 1: single-variant runner.

Calls the existing `python -m backtest.runner` CLI per variant (matches
the invocation confirmed working in dashboard-debug.json) and locates the
resulting JSON. `evaluate_run` is a placeholder here — Phase 3
(evaluators.py) replaces it with real financial/execution/risk/signal
metric extraction; for now it passes through the backtest summary fields
so Phase 1's own success check is verifiable end to end.

KNOWN GAP: variant.overrides (e.g. no_news_agent, strict_risk) are not
yet applied anywhere below. backtest.runner's CLI has no flags for them
today, so right now every variant in a config runs identically regardless
of its overrides. Wiring overrides through (either as new CLI flags or by
having run_backtest write a temporary config file backtest.runner reads)
is required before Phase 2 variants will actually differ in behavior —
worth doing before trusting a multi-variant comparison.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

from harness.experiment import ExperimentConfig

RESULTS_DIR = Path("backtest/results")


def run_backtest(run_id: str, tickers: list[str], start: str, end: str) -> Path:
    command = [
        "python",
        "-m",
        "backtest.runner",
        "--tickers",
        ",".join(tickers),
        "--start",
        start,
        "--end",
        end,
        "--run-id",
        run_id,
    ]
    subprocess.run(command, check=True)
    result_path = RESULTS_DIR / f"{run_id}.json"
    if not result_path.exists():
        raise FileNotFoundError(f"Missing backtest result: {result_path}")
    return result_path


def evaluate_run(result: dict) -> dict:
    """Placeholder pass-through — Phase 3 replaces this with real metrics
    (Sharpe, max drawdown, agreement rate, news availability rate, etc.)."""
    return {
        "final_equity": result.get("final_equity"),
        "executed": result.get("executed"),
        "exec_successes": result.get("exec_successes"),
        "exec_failures": result.get("exec_failures"),
        "risk_approvals": result.get("risk_approvals"),
    }


def run_experiment(config: ExperimentConfig, experiment_id: str) -> list[dict]:
    evaluations = []
    for variant in config.variants:
        run_id = f"{experiment_id}__{variant.name}"
        result_path = run_backtest(
            run_id=run_id,
            tickers=config.data.tickers,
            start=config.data.start,
            end=config.data.end,
        )
        result = json.loads(result_path.read_text(encoding="utf-8"))
        evaluation = evaluate_run(result)
        evaluation["variant"] = variant.name
        evaluation["run_id"] = run_id
        evaluations.append(evaluation)
    return evaluations


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()

    cfg = ExperimentConfig.from_yaml(args.config)
    results = run_experiment(cfg, args.run_id)
    for r in results:
        print(r)
