"""
Historical multi-agent trading backtester.

Design:
- Replays only sessions that actually exist in cached OHLCV data.
- Never simulates weekends or exchange holidays.
- Uses the real multi-agent orchestration pipeline.
- Uses a dedicated backtest ledger and SimBroker.
- Preserves the existing CLI.
- Keeps simulated_date synchronized with the historical session.
- Produces an equal-weight buy-and-hold benchmark.
- Persists structured NewsAgent data-availability metadata.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import data_sources
import graph.nodes as nodes_module
from broker.sim_broker import SimBroker
from config.risk_config import load_risk_config
from data_sources import set_simulated_date
from orchestrator.tick_runner import run_tick
from portfolio.ledger import PortfolioLedger

logger = logging.getLogger(__name__)

BACKTEST_TICK_DELAY_SECONDS = 1.5


@dataclass
class BacktestResult:
    """Complete output produced by one historical backtest run."""

    equity_curve: list[tuple[str, float]] = field(default_factory=list)

    buy_hold_curve: list[tuple[str, float]] = field(default_factory=list)

    trades: list[dict] = field(default_factory=list)

    tick_log: list[dict] = field(default_factory=list)

    # Per-ticker historical news provenance.
    #
    # Example:
    #
    # {
    #     "AAPL": {
    #         "sessions": 12,
    #         "available_sessions": 12,
    #         "unavailable_sessions": 0,
    #         "error_sessions": 0,
    #         "articles_seen": 0,
    #         "status": "available",
    #     }
    # }
    news_availability: dict[str, dict] = field(default_factory=dict)


def _price_lookup_factory(
    as_of_holder: dict,
):
    """
    Create a historical price lookup bound to the current simulated date.

    SimBroker calls this function when an order is submitted. The price
    therefore comes from the same historical session currently being
    replayed.
    """

    def _lookup(
        ticker: str,
    ) -> float:
        series = data_sources.fetch_ohlcv(
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


def _normalize_timestamp(
    timestamp: datetime,
) -> datetime:
    """Normalize a timestamp to timezone-aware UTC."""

    if timestamp.tzinfo is None:
        return timestamp.replace(tzinfo=timezone.utc)

    return timestamp.astimezone(timezone.utc)


def _load_sessions(
    tickers: list[str],
    start: datetime,
    end: datetime,
) -> list[datetime]:
    """
    Build the backtest calendar from actual cached OHLCV bars.

    A session is included if at least one requested ticker has a bar
    between start and end.
    """

    sessions: set[datetime] = set()

    for ticker in tickers:
        series = data_sources.fetch_ohlcv(
            ticker,
            simulated_date=end,
            lookback_days=10_000,
        )

        for bar in series.bars:
            timestamp = _normalize_timestamp(bar.timestamp)

            if start <= timestamp <= end:
                sessions.add(timestamp)

    return sorted(sessions)


def _build_buy_and_hold_curve(
    tickers: list[str],
    sessions: list[datetime],
    starting_equity: float,
) -> list[tuple[str, float]]:
    """
    Build an equal-weight buy-and-hold benchmark.

    Each ticker receives an equal fraction of starting equity at the first
    available historical session.

    No future data is used:

        price(ticker, session) <= session timestamp

    If a ticker has no price on a particular session, its most recent
    available historical price is carried forward.

    This is a benchmark, not an executable strategy.
    """

    if not tickers or not sessions or starting_equity <= 0:
        return []

    ticker_prices: dict[
        str,
        dict[datetime, float],
    ] = {}

    for ticker in tickers:
        series = data_sources.fetch_ohlcv(
            ticker,
            simulated_date=sessions[-1],
            lookback_days=10_000,
        )

        prices: dict[datetime, float] = {}

        for bar in series.bars:
            timestamp = _normalize_timestamp(bar.timestamp)

            if timestamp <= sessions[-1] and bar.close > 0:
                prices[timestamp] = bar.close

        ticker_prices[ticker] = prices

    # Determine the first usable price for each ticker.
    initial_prices: dict[str, float] = {}

    for ticker in tickers:
        prices = ticker_prices[ticker]

        for session in sessions:
            eligible_prices = [
                (timestamp, price)
                for timestamp, price in prices.items()
                if timestamp <= session
            ]

            if not eligible_prices:
                continue

            (
                _initial_timestamp,
                initial_price,
            ) = max(
                eligible_prices,
                key=lambda item: item[0],
            )

            if initial_price > 0:
                initial_prices[ticker] = initial_price

            break

    if not initial_prices:
        return []

    # Equal-weight only across tickers that actually have an initial
    # historical price.
    benchmark_tickers = list(initial_prices)

    allocation_per_ticker = starting_equity / len(benchmark_tickers)

    shares: dict[str, float] = {
        ticker: (allocation_per_ticker / initial_prices[ticker])
        for ticker in benchmark_tickers
    }

    latest_prices = dict(initial_prices)

    benchmark_curve: list[tuple[str, float]] = []

    for session in sessions:
        total_value = 0.0

        for ticker in benchmark_tickers:
            prices = ticker_prices[ticker]

            eligible_prices = [
                (timestamp, price)
                for timestamp, price in prices.items()
                if timestamp <= session
            ]

            if eligible_prices:
                (
                    _latest_timestamp,
                    latest_price,
                ) = max(
                    eligible_prices,
                    key=lambda item: item[0],
                )

                if latest_price > 0:
                    latest_prices[ticker] = latest_price

            if ticker in latest_prices:
                total_value += shares[ticker] * latest_prices[ticker]

        benchmark_curve.append(
            (
                session.date().isoformat(),
                total_value,
            )
        )

    return benchmark_curve


def _record_news_metadata(
    result: BacktestResult,
    ticker: str,
    tick_result,
) -> None:
    """
    Aggregate one ticker's NewsAgent metadata into the backtest result.
    """

    ticker_news = result.news_availability.setdefault(
        ticker,
        {
            "sessions": 0,
            "available_sessions": 0,
            "unavailable_sessions": 0,
            "error_sessions": 0,
            "articles_seen": 0,
            "status": "unknown",
        },
    )

    ticker_news["sessions"] += 1

    availability = tick_result.news_availability

    if availability == "available":
        ticker_news["available_sessions"] += 1

    elif availability == "unavailable":
        ticker_news["unavailable_sessions"] += 1

    elif availability == "error":
        ticker_news["error_sessions"] += 1

    ticker_news["articles_seen"] += tick_result.news_article_count


def _finalize_news_status(
    result: BacktestResult,
) -> None:
    """
    Resolve the final run-level NewsAgent status per ticker.

    Status semantics:

        available
            Every observed session had available news data.

        partial
            At least one session had available data, but other sessions
            were unavailable or errored.

        unavailable
            No session had available data and at least one session was
            explicitly unavailable.

        error
            No session had available data and at least one session
            returned an error without an explicit unavailable state.

        unknown
            No usable availability classification was observed.
    """

    for (
        ticker,
        news_data,
    ) in result.news_availability.items():
        sessions_count = news_data["sessions"]

        available = news_data["available_sessions"]

        unavailable = news_data["unavailable_sessions"]

        errors = news_data["error_sessions"]

        if sessions_count == 0:
            news_data["status"] = "unknown"

        elif available == sessions_count:
            news_data["status"] = "available"

        elif available > 0:
            news_data["status"] = "partial"

        elif errors > 0 and unavailable == 0:
            news_data["status"] = "error"

        elif unavailable > 0:
            news_data["status"] = "unavailable"

        else:
            news_data["status"] = "unknown"


def run_backtest(
    tickers: list[str],
    start_date: str,
    end_date: str,
    step: str = "1d",
    starting_equity: float = 100_000.0,
    ledger_path: str = "backtest/backtest_ledger.json",
    tick_delay_seconds: float = BACKTEST_TICK_DELAY_SECONDS,
) -> BacktestResult:
    """
    Run the historical multi-agent trading pipeline.

    The backtest advances through actual OHLCV sessions rather than
    calendar days.
    """

    if not tickers:
        raise ValueError("At least one ticker is required.")

    if step != "1d":
        raise NotImplementedError("Only daily steps are supported for now.")

    if tick_delay_seconds < 0:
        raise ValueError("tick_delay_seconds cannot be negative.")

    if starting_equity <= 0:
        raise ValueError("starting_equity must be greater than zero.")

    start = datetime.fromisoformat(start_date).replace(tzinfo=timezone.utc)

    end = datetime.fromisoformat(end_date).replace(tzinfo=timezone.utc)

    if start > end:
        raise ValueError(f"start_date ({start_date}) must be <= end_date ({end_date})")

    logger.info(
        "Preparing historical sessions: %s -> %s",
        start.date(),
        end.date(),
    )

    sessions = _load_sessions(
        tickers,
        start,
        end,
    )

    if not sessions:
        raise ValueError(
            "No historical OHLCV sessions found for the "
            f"requested tickers/date range: {tickers}, "
            f"{start_date} -> {end_date}"
        )

    logger.info(
        "Historical sessions found: %d",
        len(sessions),
    )

    logger.info(
        "First session: %s",
        sessions[0].date().isoformat(),
    )

    logger.info(
        "Last session: %s",
        sessions[-1].date().isoformat(),
    )

    saved_config = nodes_module._risk_config_singleton

    saved_ledger = nodes_module._ledger_singleton

    saved_broker = nodes_module._broker_singleton

    as_of_holder = {"date": sessions[0]}

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
        for (
            session_index,
            current,
        ) in enumerate(
            sessions,
            start=1,
        ):
            as_of_holder["date"] = current

            # Keep ledger historical session state synchronized.
            ledger.simulated_date = current.date()

            # Keep every historical data source synchronized.
            set_simulated_date(current)

            logger.info(
                "Backtest session %d/%d starting: %s",
                session_index,
                len(sessions),
                current.date().isoformat(),
            )

            tick_results = asyncio.run(run_tick(tickers))

            for (
                ticker,
                tick_result,
            ) in tick_results.items():
                # --------------------------------------------------
                # Per-tick log.
                # --------------------------------------------------

                result.tick_log.append(
                    {
                        "date": (current.date().isoformat()),
                        "ticker": ticker,
                        "outcome": (tick_result.outcome),
                        "notes": (tick_result.notes),
                        # Structured news provenance.
                        "news_availability": (tick_result.news_availability),
                        "news_article_count": (tick_result.news_article_count),
                        "news_source": (tick_result.news_source),
                        "news_as_of": (tick_result.news_as_of),
                    }
                )

                # --------------------------------------------------
                # Aggregate news quality.
                # --------------------------------------------------

                _record_news_metadata(
                    result,
                    ticker,
                    tick_result,
                )

                # --------------------------------------------------
                # Executed trades.
                # --------------------------------------------------

                if tick_result.outcome == "executed":
                    result.trades.append(
                        {
                            "date": (current.date().isoformat()),
                            "ticker": ticker,
                            "notes": (tick_result.notes),
                            "shares": (tick_result.shares),
                            "price": (tick_result.price),
                        }
                    )

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

            if session_index < len(sessions) and tick_delay_seconds > 0:
                logger.info(
                    "Backtest rate-limit delay: %.2fs",
                    tick_delay_seconds,
                )

                time.sleep(tick_delay_seconds)

        # Resolve final news status before
        # serializing the result.
        _finalize_news_status(result)

        # Build the benchmark after the historical
        # session calendar is known.
        result.buy_hold_curve = _build_buy_and_hold_curve(
            tickers=tickers,
            sessions=sessions,
            starting_equity=starting_equity,
        )

    finally:
        # Always clear historical replay state.
        set_simulated_date(None)

        # Always restore the previous process-level
        # singletons.
        nodes_module._risk_config_singleton = saved_config

        nodes_module._ledger_singleton = saved_ledger

        nodes_module._broker_singleton = saved_broker

    return result


def _serialize_result(
    result: BacktestResult,
) -> dict:
    """
    Convert the backtest result into the JSON result schema.
    """

    from backtest.metrics import summarize

    equity_values = [equity for _date, equity in result.equity_curve]

    buy_hold_values = [equity for _date, equity in result.buy_hold_curve]

    metrics = summarize(
        equity_values,
        result.trades,
        buy_hold_values,
    )

    return {
        "sharpe_ratio": metrics["sharpe_ratio"],
        "win_rate": metrics["win_rate"],
        "closed_trades": metrics["closed_trades"],
        "max_drawdown": metrics["max_drawdown"],
        "total_return": metrics["total_return"],
        "buy_hold_return": metrics["buy_and_hold_return"],
        "excess_return_vs_buy_and_hold": (metrics["excess_return_vs_buy_and_hold"]),
        "equity_curve": (result.equity_curve),
        "buy_hold_curve": (result.buy_hold_curve),
        "trades": result.trades,
        "tick_log": result.tick_log,
        # ----------------------------------------------------------
        # Structured NewsAgent data-quality metadata.
        # ----------------------------------------------------------
        "news_availability": (result.news_availability),
    }


def main() -> None:
    import argparse
    import json

    parser = argparse.ArgumentParser(
        description=("Run the multi-agent trading backtest.")
    )

    parser.add_argument(
        "--tickers",
        required=True,
        help=("Comma-separated ticker symbols, e.g. AAPL,MSFT,GOOGL"),
    )

    parser.add_argument(
        "--start",
        required=True,
        help=("Backtest start date, e.g. 2024-10-15"),
    )

    parser.add_argument(
        "--end",
        required=True,
        help=("Backtest end date, e.g. 2024-10-31"),
    )

    parser.add_argument(
        "--starting-equity",
        type=float,
        default=100_000.0,
    )

    parser.add_argument(
        "--tick-delay",
        type=float,
        default=BACKTEST_TICK_DELAY_SECONDS,
        help=("Delay between historical sessions in seconds."),
    )

    parser.add_argument(
        "--run-id",
        default="latest",
        help=("Identifier used for the result JSON filename."),
    )

    parser.add_argument(
        "--ledger-path",
        default=("backtest/backtest_ledger.json"),
        help=("Path to the dedicated backtest ledger."),
    )

    args = parser.parse_args()

    tickers = [
        ticker.strip().upper() for ticker in args.tickers.split(",") if ticker.strip()
    ]

    result = run_backtest(
        tickers=tickers,
        start_date=args.start,
        end_date=args.end,
        starting_equity=args.starting_equity,
        ledger_path=args.ledger_path,
        tick_delay_seconds=args.tick_delay,
    )

    payload = _serialize_result(result)

    results_dir = Path("backtest/results")

    results_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = results_dir / f"{args.run_id}.json"

    output_path.write_text(
        json.dumps(
            payload,
            indent=2,
        ),
        encoding="utf-8",
    )

    print()
    print("=" * 60)
    print("BACKTEST COMPLETE")
    print("=" * 60)
    print(f"Sessions      : {len(result.equity_curve)}")
    print(f"Executed      : {len(result.trades)}")

    print(
        f"Final equity  : ${result.equity_curve[-1][1]:,.2f}"
        if result.equity_curve
        else "Final equity  : N/A"
    )

    print(f"Benchmark     : {len(result.buy_hold_curve)} points")

    print(f"News tickers  : {len(result.news_availability)}")

    print(f"Result file   : {output_path}")

    print("=" * 60)


if __name__ == "__main__":
    main()
