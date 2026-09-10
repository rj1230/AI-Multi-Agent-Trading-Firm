"""
Walks HistoricalDataSource's simulated clock forward day by day, invoking
run_tick() (the real multi-ticker orchestrator: local RiskAgent ->
PortfolioRiskCoordinator -> ExecutionAgent) at each step.

Uses a dedicated backtest ledger/broker, NOT the live singletons.

Rate-limit protection:
- Adds a configurable delay between daily ticks.
- Prevents repeated backtest days from immediately hammering the LLM API.
- Keeps the existing singleton save/restore behavior intact.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone

import graph.nodes as nodes_module
from broker.sim_broker import SimBroker
from config.risk_config import load_risk_config
from data_sources import fetch_ohlcv, set_simulated_date
from orchestrator.tick_runner import run_tick
from portfolio.ledger import PortfolioLedger

logger = logging.getLogger(__name__)


# ---------------------------------------------------------
# Backtest rate-limit configuration
# ---------------------------------------------------------

# Delay between simulated days.
#
# 1.5 seconds is deliberately conservative for a free-tier LLM
# during debugging. Increase this to 2-3 seconds if you still
# encounter HTTP 429 responses.
BACKTEST_TICK_DELAY_SECONDS = 1.5


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
    traded, as of the CURRENT simulated date."""

    def _lookup(ticker: str) -> float:
        series = fetch_ohlcv(
            ticker,
            simulated_date=as_of_holder["date"],
            lookback_days=1,
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
    tick_delay_seconds: float = BACKTEST_TICK_DELAY_SECONDS,
) -> BacktestResult:
    """Returns a full trade log + per-step equity curve.

    Uses its own ledger file (ledger_path) and never mutates
    the live paper-trading ledger.

    Parameters
    ----------
    tickers:
        Tickers to process on every simulated day.

    start_date:
        First simulated date, e.g. "2026-01-01".

    end_date:
        Last simulated date, inclusive.

    step:
        Currently only "1d" is supported.

    starting_equity:
        Initial simulated portfolio equity.

    ledger_path:
        Dedicated backtest ledger path.

    tick_delay_seconds:
        Delay between simulated daily ticks.

        This is primarily used to protect free-tier LLM APIs
        such as Groq from rate limiting during backtests.

        Set to 0 to disable the delay.
    """

    if step != "1d":
        raise NotImplementedError("Only daily steps are supported for now.")

    if tick_delay_seconds < 0:
        raise ValueError("tick_delay_seconds cannot be negative.")

    start = datetime.fromisoformat(start_date).replace(tzinfo=timezone.utc)

    end = datetime.fromisoformat(end_date).replace(tzinfo=timezone.utc)

    if start > end:
        raise ValueError(f"start_date ({start_date}) must be <= end_date ({end_date})")

    # Save real singletons so this run can restore them afterward.
    #
    # IMPORTANT:
    # Never leave graph.nodes pointed at backtest state after
    # this function returns.
    saved_config = nodes_module._risk_config_singleton
    saved_ledger = nodes_module._ledger_singleton
    saved_broker = nodes_module._broker_singleton

    as_of_holder = {"date": start}

    ledger = PortfolioLedger(
        starting_equity=starting_equity,
        path=ledger_path,
    )

    broker = SimBroker(
        starting_cash=starting_equity,
        price_lookup=_price_lookup_factory(as_of_holder),
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

            logger.info(
                "Backtest tick starting: %s",
                current.date().isoformat(),
            )

            # -------------------------------------------------
            # Run the REAL orchestration path.
            # -------------------------------------------------

            tick_results = asyncio.run(run_tick(tickers))

            # -------------------------------------------------
            # Record results.
            # -------------------------------------------------

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

            # -------------------------------------------------
            # Record equity.
            # -------------------------------------------------

            equity = ledger.snapshot().equity

            result.equity_curve.append(
                (
                    current.date().isoformat(),
                    equity,
                )
            )

            logger.info(
                "Backtest %s: equity=%.2f",
                current.date().isoformat(),
                equity,
            )

            # -------------------------------------------------
            # Move to next simulated day.
            #
            # IMPORTANT:
            # This sleep is REAL wall-clock time. It does not
            # affect the simulated trading date.
            #
            # It exists purely to prevent the LLM provider from
            # receiving requests too rapidly during backtests.
            # -------------------------------------------------

            current += timedelta(days=1)

            if current <= end and tick_delay_seconds > 0:
                logger.info(
                    "Backtest rate-limit delay: %.2fs",
                    tick_delay_seconds,
                )

                time.sleep(tick_delay_seconds)

    finally:
        # Always restore live state, even if:
        # - Groq raises HTTP 429
        # - run_tick() raises
        # - the user presses Ctrl+C
        # - another exception occurs
        set_simulated_date(None)

        nodes_module._risk_config_singleton = saved_config
        nodes_module._ledger_singleton = saved_ledger
        nodes_module._broker_singleton = saved_broker

    return result
