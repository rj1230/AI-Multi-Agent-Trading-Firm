"""
Historical multi-agent trading backtester.

Design:

* Replays only sessions that actually exist in cached OHLCV data.
* Never simulates weekends or exchange holidays.
* Uses the real multi-agent orchestration pipeline.
* Uses a dedicated backtest ledger and SimBroker.
* Preserves the existing CLI.
* Keeps simulated_date synchronized with the historical session.
* Produces an equal-weight buy-and-hold benchmark.
* Persists structured NewsAgent provenance.
* Persists structured decision telemetry for every ticker/session.
* Persists structured execution telemetry for every ticker/session.
* Persists structured position-accounting telemetry.
* Persists deterministic TradeTrace audit telemetry for every ticker/session.
* Separates execution events from completed/closed trades.
* Produces run-level orchestration diagnostics.
* Produces realized-P&L and trade-quality performance metrics.
* Captures final portfolio state before dependency cleanup.

The telemetry and diagnostics in this module are observational.
They do not alter Strategy V1 decision logic.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections import Counter
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import data_sources
import graph.nodes as nodes_module
from broker.sim_broker import SimBroker
from config.risk_config import load_risk_config
from data_sources import set_simulated_date
from orchestrator.tick_runner import run_tick
from portfolio.ledger import PortfolioLedger
from telemetry.trade_trace import TradeTrace

logger = logging.getLogger(__name__)

BACKTEST_TICK_DELAY_SECONDS = 1.5


@dataclass
class BacktestResult:
    """Complete output produced by one historical backtest run."""

    run_id: str = "latest"

    # Performance curves.
    equity_curve: list[tuple[str, float]] = field(default_factory=list)
    buy_hold_curve: list[tuple[str, float]] = field(default_factory=list)

    # Canonical deterministic audit traces.
    #
    # One trace is produced for every ticker/session, including:
    #   - executed trades
    #   - local risk rejects
    #   - portfolio-book rejects
    #   - execution rejects
    trade_traces: list[dict] = field(default_factory=list)

    # Execution events.
    #
    # One entry represents one successful broker execution/fill.
    trades: list[dict] = field(default_factory=list)

    # Completed position round trips.
    #
    # An entry is created only when a fill completely closes the position.
    closed_trades: list[dict] = field(default_factory=list)

    # Backward-compatible per ticker/session telemetry.
    tick_log: list[dict] = field(default_factory=list)

    # Aggregated NewsAgent availability/provenance.
    news_availability: dict[str, dict] = field(default_factory=dict)

    # ------------------------------------------------------------------
    # Final run state.
    #
    # These fields are deliberately stored on BacktestResult rather than
    # exposed through graph.nodes singletons.
    #
    # run_backtest() restores its temporary broker/ledger dependencies
    # before returning, so callers must inspect these fields for final
    # backtest state.
    # ------------------------------------------------------------------
    final_positions: dict[str, dict] = field(default_factory=dict)
    final_equity: float = 0.0
    final_cash: float = 0.0
    final_realized_pnl: float = 0.0


def _price_lookup_factory(as_of_holder: dict):
    """
    Create a historical price lookup bound to the current simulated date.

    SimBroker calls this function when an order is submitted. The price
    therefore comes from the same historical session currently being
    replayed.
    """

    def _lookup(ticker: str) -> float:
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


def _normalize_timestamp(timestamp: datetime) -> datetime:
    """Normalize a timestamp to timezone-aware UTC."""

    if timestamp.tzinfo is None:
        return timestamp.replace(tzinfo=UTC)

    return timestamp.astimezone(UTC)


def _parse_backtest_datetime(value: str) -> datetime:
    """
    Parse a backtest datetime/date into timezone-aware UTC.

    Date-only values are interpreted as midnight UTC.
    Explicit offsets are preserved and normalized to UTC.
    """

    parsed = datetime.fromisoformat(value)

    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)

    return parsed.astimezone(UTC)


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

    ticker_prices: dict[str, dict[datetime, float]] = {}

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

            _initial_timestamp, initial_price = max(
                eligible_prices,
                key=lambda item: item[0],
            )

            if initial_price > 0:
                initial_prices[ticker] = initial_price

            break

    if not initial_prices:
        return []

    benchmark_tickers = list(initial_prices)

    allocation_per_ticker = starting_equity / len(benchmark_tickers)

    shares: dict[str, float] = {
        ticker: allocation_per_ticker / initial_prices[ticker]
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
                _latest_timestamp, latest_price = max(
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
    """Aggregate one ticker's NewsAgent metadata."""

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


def _finalize_news_status(result: BacktestResult) -> None:
    """Resolve final run-level NewsAgent status per ticker."""

    for news_data in result.news_availability.values():
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


def _build_diagnostics(result: BacktestResult) -> dict:
    """
    Build run-level orchestration diagnostics.

    These metrics describe what happened inside the orchestration funnel.

    They are observational only and do not alter trading decisions.
    """

    outcome_counts = Counter(tick.get("outcome") for tick in result.tick_log)

    execution_attempts = sum(
        bool(tick.get("execution_attempted")) for tick in result.tick_log
    )

    execution_successes = sum(
        tick.get("execution_success") is True for tick in result.tick_log
    )

    execution_failures = sum(
        tick.get("execution_attempted") is True
        and tick.get("execution_success") is False
        for tick in result.tick_log
    )

    risk_approvals = sum(
        tick.get("risk_decision") == "approved" for tick in result.tick_log
    )

    coordinator_approvals = sum(
        tick.get("outcome") == "executed"
        or tick.get("outcome") == "held_execution_reject"
        for tick in result.tick_log
    )

    local_rejections = sum(
        tick.get("outcome") == "held_local_reject" for tick in result.tick_log
    )

    book_rejections = sum(
        tick.get("outcome") == "held_book_reject" for tick in result.tick_log
    )

    trace_count = len(result.trade_traces)

    trace_tickers = len(
        {trace.get("ticker") for trace in result.trade_traces if trace.get("ticker")}
    )

    return {
        "run_id": result.run_id,
        "tick_count": len(result.tick_log),
        "trade_trace_count": trace_count,
        "trade_trace_tickers": trace_tickers,
        "trace_coverage_complete": trace_count == len(result.tick_log),
        "outcome_counts": dict(outcome_counts),
        "risk_approvals": risk_approvals,
        "local_rejections": local_rejections,
        "coordinator_approvals": coordinator_approvals,
        "book_rejections": book_rejections,
        "execution_attempts": execution_attempts,
        "execution_successes": execution_successes,
        "execution_failures": execution_failures,
        "execution_events": len(result.trades),
        "closed_trades": len(result.closed_trades),
    }


def _build_execution_event(
    current: datetime,
    ticker: str,
    tick_result,
) -> dict:
    """
    Build one execution-event record.

    `trades` intentionally remains an execution-event collection rather
    than a completed round-trip collection.
    """

    return {
        "date": current.date().isoformat(),
        "ticker": ticker,
        "notes": tick_result.notes,
        "shares": tick_result.shares,
        "price": tick_result.price,
        "signal_direction": tick_result.signal_direction,
        "signal_confidence": tick_result.signal_confidence,
        "merge_agreement": tick_result.merge_agreement,
        "atr": tick_result.atr,
        "entry_price": tick_result.entry_price,
        "sector": tick_result.sector,
        "proposed_shares": tick_result.proposed_shares,
        "risk_decision": tick_result.risk_decision,
        "execution_attempted": tick_result.execution_attempted,
        "execution_success": tick_result.execution_success,
        "execution_side": tick_result.execution_side,
        "average_cost": tick_result.average_cost,
        "realized_pnl": tick_result.realized_pnl,
        "remaining_shares": tick_result.remaining_shares,
        "position_closed": tick_result.position_closed,
        "news_availability": tick_result.news_availability,
        "news_article_count": tick_result.news_article_count,
        "news_source": tick_result.news_source,
        "news_as_of": tick_result.news_as_of,
    }


def _build_closed_trade_event(
    current: datetime,
    ticker: str,
    tick_result,
) -> dict:
    """
    Build a completed position/round-trip record.

    This function should only be called when `position_closed` is true.
    """

    return {
        "date": current.date().isoformat(),
        "ticker": ticker,
        "side": tick_result.execution_side,
        "shares": tick_result.shares,
        "exit_price": tick_result.price,
        "average_cost": tick_result.average_cost,
        "realized_pnl": tick_result.realized_pnl,
        "sector": tick_result.sector,
        "signal_direction": tick_result.signal_direction,
        "signal_confidence": tick_result.signal_confidence,
        "merge_agreement": tick_result.merge_agreement,
        "notes": tick_result.notes,
        "news_availability": tick_result.news_availability,
        "news_article_count": tick_result.news_article_count,
        "news_source": tick_result.news_source,
        "news_as_of": tick_result.news_as_of,
    }


def _capture_trade_trace(
    result: BacktestResult,
    tick_result,
) -> None:
    """
    Persist the canonical TradeTrace produced by run_tick().

    TradeTrace is intentionally observational. The backtester does not
    construct or mutate trading decisions from the trace.
    """

    trace = getattr(tick_result, "trade_trace", None)

    if trace is None:
        return

    if not isinstance(trace, TradeTrace):
        raise TypeError(
            "TickResult.trade_trace must be a TradeTrace instance or None, "
            f"got {type(trace).__name__}"
        )

    result.trade_traces.append(trace.to_dict())


def _capture_final_state(
    result: BacktestResult,
    ledger: PortfolioLedger,
) -> None:
    """Capture authoritative final portfolio state before cleanup."""

    snapshot = ledger.snapshot()

    result.final_positions = {
        ticker: {
            "shares": float(position.shares),
            "market_value": float(position.market_value),
        }
        for ticker, position in snapshot.positions.items()
    }

    result.final_equity = float(snapshot.equity)
    result.final_cash = float(ledger.get_cash())
    result.final_realized_pnl = float(ledger.get_realized_pnl())


def run_backtest(
    tickers: list[str],
    start_date: str,
    end_date: str,
    step: str = "1d",
    starting_equity: float = 100_000.0,
    ledger_path: str = "backtest/backtest_ledger.json",
    tick_delay_seconds: float = BACKTEST_TICK_DELAY_SECONDS,
    run_id: str = "latest",
) -> BacktestResult:
    """
    Run the historical multi-agent trading pipeline.

    The backtest advances through actual OHLCV sessions rather than
    calendar days.

    This function preserves Strategy V1 behavior while recording richer
    decision, execution, position-accounting, and TradeTrace telemetry.

    Runtime lifecycle:

        1. Save existing graph dependencies.
        2. Install dedicated backtest ledger/broker/config.
        3. Replay historical sessions.
        4. Capture final portfolio state.
        5. Restore original graph dependencies.
        6. Return BacktestResult.

    This guarantees state isolation between backtest runs while still
    exposing the completed run state to callers.
    """

    if not tickers:
        raise ValueError("At least one ticker is required.")

    if step != "1d":
        raise NotImplementedError("Only daily steps are supported for now.")

    if tick_delay_seconds < 0:
        raise ValueError("tick_delay_seconds cannot be negative.")

    if starting_equity <= 0:
        raise ValueError("starting_equity must be greater than zero.")

    if not run_id or not run_id.strip():
        raise ValueError("run_id cannot be empty.")

    tickers = [ticker.strip().upper() for ticker in tickers if ticker.strip()]

    if not tickers:
        raise ValueError("At least one non-empty ticker is required.")

    run_id = run_id.strip()

    start = _parse_backtest_datetime(start_date)
    end = _parse_backtest_datetime(end_date)

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

    # Preserve the process-level dependencies exactly as they were
    # before the backtest.
    saved_config = nodes_module._risk_config_singleton
    saved_ledger = nodes_module._ledger_singleton
    saved_broker = nodes_module._broker_singleton

    as_of_holder = {
        "date": sessions[0],
    }

    # Dedicated state for this backtest run.
    ledger = PortfolioLedger(
        starting_equity=starting_equity,
        path=ledger_path,
        fresh=True,
    )

    broker = SimBroker(
        starting_cash=starting_equity,
        price_lookup=_price_lookup_factory(as_of_holder),
    )

    nodes_module._risk_config_singleton = load_risk_config()
    nodes_module._ledger_singleton = ledger
    nodes_module._broker_singleton = broker

    result = BacktestResult(
        run_id=run_id,
    )

    try:
        for session_index, current in enumerate(
            sessions,
            start=1,
        ):
            as_of_holder["date"] = current

            ledger.simulated_date = current.date()

            set_simulated_date(current)

            logger.info(
                "Backtest session %d/%d starting: %s",
                session_index,
                len(sessions),
                current.date().isoformat(),
            )

            tick_results = asyncio.run(
                run_tick(
                    tickers,
                    run_id=run_id,
                    simulated_date=current,
                )
            )

            for ticker, tick_result in tick_results.items():
                # ------------------------------------------------------
                # Canonical TradeTrace.
                #
                # Capture this for every ticker/session, regardless of
                # whether a trade was ultimately executed.
                # ------------------------------------------------------
                _capture_trade_trace(
                    result=result,
                    tick_result=tick_result,
                )

                result.tick_log.append(
                    {
                        "date": current.date().isoformat(),
                        "ticker": ticker,
                        "outcome": tick_result.outcome,
                        "notes": tick_result.notes,
                        "shares": tick_result.shares,
                        "price": tick_result.price,
                        "signal_direction": tick_result.signal_direction,
                        "signal_confidence": tick_result.signal_confidence,
                        "merge_agreement": tick_result.merge_agreement,
                        "atr": tick_result.atr,
                        "entry_price": tick_result.entry_price,
                        "sector": tick_result.sector,
                        "proposed_shares": tick_result.proposed_shares,
                        "risk_decision": tick_result.risk_decision,
                        "risk_notes": tick_result.risk_notes or [],
                        "execution_attempted": tick_result.execution_attempted,
                        "execution_success": tick_result.execution_success,
                        "execution_side": tick_result.execution_side,
                        "average_cost": tick_result.average_cost,
                        "realized_pnl": tick_result.realized_pnl,
                        "remaining_shares": tick_result.remaining_shares,
                        "position_closed": tick_result.position_closed,
                        "news_availability": tick_result.news_availability,
                        "news_article_count": tick_result.news_article_count,
                        "news_source": tick_result.news_source,
                        "news_as_of": tick_result.news_as_of,
                    }
                )

                _record_news_metadata(
                    result,
                    ticker,
                    tick_result,
                )

                if tick_result.outcome == "executed":
                    result.trades.append(
                        _build_execution_event(
                            current=current,
                            ticker=ticker,
                            tick_result=tick_result,
                        )
                    )

                    # A closed trade is fundamentally different from
                    # an execution event. Only create it when the
                    # accounting layer confirms that the position
                    # reached zero.
                    if tick_result.position_closed:
                        result.closed_trades.append(
                            _build_closed_trade_event(
                                current=current,
                                ticker=ticker,
                                tick_result=tick_result,
                            )
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

        _finalize_news_status(result)

        result.buy_hold_curve = _build_buy_and_hold_curve(
            tickers=tickers,
            sessions=sessions,
            starting_equity=starting_equity,
        )

        # IMPORTANT:
        #
        # Capture the final state BEFORE finally restores the temporary
        # broker and ledger singletons.
        #
        # This is the authoritative final state of this backtest run.
        _capture_final_state(
            result=result,
            ledger=ledger,
        )

    finally:
        # Always clear historical time-travel state.
        set_simulated_date(None)

        # Always restore the process state that existed before the run.
        nodes_module._risk_config_singleton = saved_config
        nodes_module._ledger_singleton = saved_ledger
        nodes_module._broker_singleton = saved_broker

    return result


def _serialize_result(result: BacktestResult) -> dict:
    """
    Convert the backtest result into the JSON result schema.

    The schema deliberately contains:

        1. execution-level events
        2. closed-trade events
        3. detailed tick-level telemetry
        4. canonical TradeTrace audit telemetry
        5. aggregated run-level diagnostics
        6. strategy performance metrics
        7. final portfolio state

    This allows the UI and research tooling to inspect the decision
    funnel and performance without parsing human-readable notes.
    """

    from backtest.metrics import summarize

    equity_values = [equity for _date, equity in result.equity_curve]

    buy_hold_values = [equity for _date, equity in result.buy_hold_curve]

    # IMPORTANT:
    #
    # `trades` contains execution events.
    # `closed_trades` contains completed positions.
    #
    # Performance metrics such as win rate, realized P&L and profit
    # factor must therefore use the explicit closed-trade collection.
    metrics = summarize(
        equity_values,
        result.trades,
        buy_hold_values,
        result.closed_trades,
    )

    diagnostics = _build_diagnostics(result)

    return {
        # --------------------------------------------------------------
        # Run identity.
        # --------------------------------------------------------------
        "run_id": result.run_id,
        # --------------------------------------------------------------
        # Core performance metrics.
        # --------------------------------------------------------------
        "sharpe_ratio": metrics["sharpe_ratio"],
        "win_rate": metrics["win_rate"],
        "closed_trades": metrics["closed_trades"],
        "max_drawdown": metrics["max_drawdown"],
        "total_return": metrics["total_return"],
        "buy_hold_return": metrics["buy_and_hold_return"],
        "excess_return_vs_buy_and_hold": (metrics["excess_return_vs_buy_and_hold"]),
        # --------------------------------------------------------------
        # Realized trade-quality metrics.
        # --------------------------------------------------------------
        "realized_pnl": metrics["realized_pnl"],
        "gross_profit": metrics["gross_profit"],
        "gross_loss": metrics["gross_loss"],
        "profit_factor": metrics["profit_factor"],
        "average_winning_trade": metrics["average_winning_trade"],
        "average_losing_trade": metrics["average_losing_trade"],
        # --------------------------------------------------------------
        # Final authoritative backtest state.
        #
        # These values come from the dedicated backtest ledger captured
        # immediately before runtime cleanup.
        # --------------------------------------------------------------
        "final_positions": result.final_positions,
        "final_equity": result.final_equity,
        "final_cash": result.final_cash,
        "final_realized_pnl": result.final_realized_pnl,
        # --------------------------------------------------------------
        # Equity curves.
        # --------------------------------------------------------------
        "equity_curve": result.equity_curve,
        "buy_hold_curve": result.buy_hold_curve,
        # --------------------------------------------------------------
        # Execution events.
        #
        # One record per successful broker execution/fill.
        # --------------------------------------------------------------
        "trades": result.trades,
        # --------------------------------------------------------------
        # Completed position events.
        #
        # One record per fully closed position.
        # --------------------------------------------------------------
        "closed_trade_events": result.closed_trades,
        # --------------------------------------------------------------
        # Backward-compatible tick-level telemetry.
        # --------------------------------------------------------------
        "tick_log": result.tick_log,
        # --------------------------------------------------------------
        # Canonical deterministic TradeTrace telemetry.
        #
        # One trace per ticker/session when run_tick() provides one.
        # --------------------------------------------------------------
        "trade_traces": result.trade_traces,
        # --------------------------------------------------------------
        # Run-level orchestration diagnostics.
        # --------------------------------------------------------------
        "diagnostics": diagnostics,
        # --------------------------------------------------------------
        # NewsAgent provenance/availability.
        # --------------------------------------------------------------
        "news_availability": result.news_availability,
    }


def main() -> None:
    import argparse
    import json

    parser = argparse.ArgumentParser(
        description="Run the multi-agent trading backtest."
    )

    parser.add_argument(
        "--tickers",
        required=True,
        help="Comma-separated ticker symbols, e.g. AAPL,MSFT,GOOGL",
    )

    parser.add_argument(
        "--start",
        required=True,
        help="Backtest start date, e.g. 2024-10-15",
    )

    parser.add_argument(
        "--end",
        required=True,
        help="Backtest end date, e.g. 2024-10-31",
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
        help="Delay between historical sessions in seconds.",
    )

    parser.add_argument(
        "--run-id",
        default="latest",
        help="Unique identifier used for the run and result JSON filename.",
    )

    parser.add_argument(
        "--ledger-path",
        default="backtest/backtest_ledger.json",
        help="Path to the dedicated backtest ledger.",
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
        run_id=args.run_id,
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
            default=str,
        ),
        encoding="utf-8",
    )

    diagnostics = payload["diagnostics"]

    # Performance metrics can legitimately be None when there are
    # no completed/closed trades. Format those values safely rather
    # than treating undefined metrics as zero.
    realized_pnl = payload["realized_pnl"]
    profit_factor = payload["profit_factor"]
    win_rate = payload["win_rate"]

    realized_pnl_display = (
        f"${realized_pnl:,.2f}" if realized_pnl is not None else "N/A"
    )

    profit_factor_display = (
        f"{profit_factor:.4f}" if profit_factor is not None else "N/A"
    )

    win_rate_display = f"{win_rate:.2%}" if win_rate is not None else "N/A"

    print()
    print("=" * 60)
    print("BACKTEST COMPLETE")
    print("=" * 60)

    print(f"Run ID        : {result.run_id}")

    print(f"Sessions      : {len(result.equity_curve)}")

    print(f"Executed      : {len(result.trades)}")

    print(f"Closed trades : {len(result.closed_trades)}")

    print(f"Trade traces  : {len(result.trade_traces)}")

    print(f"Trace coverage: {diagnostics['trace_coverage_complete']}")

    print(f"Realized P&L  : {realized_pnl_display}")

    print(f"Profit factor : {profit_factor_display}")

    print(f"Win rate      : {win_rate_display}")

    print(f"Final equity  : ${result.final_equity:,.2f}")

    print(f"Final cash    : ${result.final_cash:,.2f}")

    print(f"Final positions: {len(result.final_positions)}")

    print(f"Benchmark     : {len(result.buy_hold_curve)} points")

    print(f"News tickers  : {len(result.news_availability)}")

    print(f"Risk approvals: {diagnostics['risk_approvals']}")

    print(f"Exec attempts : {diagnostics['execution_attempts']}")

    print(f"Exec successes: {diagnostics['execution_successes']}")

    print(f"Exec failures : {diagnostics['execution_failures']}")

    print(f"Result file   : {output_path}")

    print("=" * 60)


if __name__ == "__main__":
    main()
