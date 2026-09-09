"""
Walks HistoricalDataSource's simulated clock forward day by day, invoking
run_tick() (the real multi-ticker orchestrator: local RiskAgent ->
PortfolioRiskCoordinator -> ExecutionAgent, same code path live/paper
trading uses) at each step -- Phase 4.1's dual-mode principle applied to
the whole system, not just data fetching.

Uses a dedicated backtest ledger/broker, NOT the live singletons --
running a backtest must never mutate portfolio/ledger.json, the real
paper-trading ledger. This module swaps graph.nodes' singleton globals
for the duration of the run and restores them afterward, the same
override pattern tests/test_nodes.py and tests/test_tick_runner.py
already use via monkeypatch (this isn't a test, so it does the
save/restore manually in a try/finally instead).
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone

import graph.nodes as nodes_module
from broker.sim_broker import SimBroker
from config.risk_config import load_risk_config
from data_sources import fetch_ohlcv, set_simulated_date
from orchestrator.tick_runner import run_tick
from portfolio.ledger import PortfolioLedger

logger = logging.getLogger(__name__)


@dataclass
class BacktestResult:
    equity_curve: list[tuple[str, float]] = field(
        default_factory=list
    )  # (date_iso, equity)
    trades: list[dict] = field(default_factory=list)  # one dict per executed TickResult
    tick_log: list[dict] = field(
        default_factory=list
    )  # every ticker's outcome, every day


def _price_lookup_factory(as_of_holder: dict):
    """Backtest SimBroker fills need a price for whatever ticker is being
    traded, as of the CURRENT simulated date (as_of_holder is mutated by
    the loop below each step, since SimBroker's price_lookup signature is
    just (ticker) -> float, no date parameter)."""

    def _lookup(ticker: str) -> float:
        series = fetch_ohlcv(
            ticker, simulated_date=as_of_holder["date"], lookback_days=1
        )
        if series.is_empty or series.latest is None:
            raise ValueError(
                f"No historical price for {ticker} as of {as_of_holder['date']}"
            )
        return series.latest.close

    return _lookup


def run_backtest(
    tickers: list[str],
    start_date: str,
    end_date: str,
    step: str = "1d",
    starting_equity: float = 100_000.0,
    ledger_path: str = "backtest/backtest_ledger.json",
) -> BacktestResult:
    """Returns a full trade log + per-step equity curve. Uses its own
    ledger file (ledger_path) -- never the live paper-trading ledger."""
    if step != "1d":
        raise NotImplementedError("Only daily steps are supported for now.")

    start = datetime.fromisoformat(start_date).replace(tzinfo=timezone.utc)
    end = datetime.fromisoformat(end_date).replace(tzinfo=timezone.utc)

    # Save real singletons so this run can restore them afterward --
    # never leave graph.nodes pointed at backtest state after returning.
    saved_config = nodes_module._risk_config_singleton
    saved_ledger = nodes_module._ledger_singleton
    saved_broker = nodes_module._broker_singleton

    as_of_holder = {"date": start}
    ledger = PortfolioLedger(starting_equity=starting_equity, path=ledger_path)
    broker = SimBroker(
        starting_cash=starting_equity, price_lookup=_price_lookup_factory(as_of_holder)
    )

    nodes_module._risk_config_singleton = load_risk_config()
    nodes_module._ledger_singleton = ledger
    nodes_module._broker_singleton = broker

    result = BacktestResult()

    try:
        current = start
        while current <= end:
            as_of_holder["date"] = current
            ledger.simulated_date = current.date()
            set_simulated_date(current)

            tick_results = asyncio.run(run_tick(tickers))

            for ticker, tick_result in tick_results.items():
                result.tick_log.append(
                    {
                        "date": current.date().isoformat(),
                        "ticker": ticker,
                        "outcome": tick_result.outcome,
                        "notes": tick_result.notes,
                    }
                )
                if tick_result.outcome == "executed":
                    result.trades.append(
                        {
                            "date": current.date().isoformat(),
                            "ticker": ticker,
                            "notes": tick_result.notes,
                        }
                    )

            equity = ledger.snapshot().equity
            result.equity_curve.append((current.date().isoformat(), equity))
            logger.info("Backtest %s: equity=%.2f", current.date().isoformat(), equity)

            current += timedelta(days=1)
    finally:
        set_simulated_date(None)
        nodes_module._risk_config_singleton = saved_config
        nodes_module._ledger_singleton = saved_ledger
        nodes_module._broker_singleton = saved_broker

    return result
