from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import datetime
from typing import Literal

from graph import nodes
from graph.state import RiskDecision, TradingState
from portfolio.coordinator import (
    TickerProposal,
    run_portfolio_coordinator,
)
from portfolio.state import build_correlation_matrix
from telemetry.trade_trace import (
    AccountingTrace,
    ChartTrace,
    CoordinatorTrace,
    ExecutionTrace,
    NewsTrace,
    RiskTrace,
    SignalTrace,
    TradeTrace,
)


@dataclass
class TickResult:
    outcome: Literal[
        "executed",
        "held_local_reject",
        "held_book_reject",
        "held_execution_reject",
    ]

    notes: str
    shares: float
    price: float | None

    signal_direction: str | None = None
    signal_confidence: float = 0.0
    merge_agreement: bool | None = None

    atr: float | None = None
    entry_price: float | None = None
    sector: str | None = None
    proposed_shares: float = 0.0

    risk_decision: str | None = None
    risk_notes: list[str] | None = None

    execution_attempted: bool = False
    execution_success: bool | None = None
    execution_side: str | None = None

    average_cost: float | None = None
    realized_pnl: float | None = None
    remaining_shares: float | None = None
    position_closed: bool = False

    news_availability: str | None = None
    news_article_count: int = 0
    news_source: str | None = None
    news_as_of: str | None = None

    trade_trace: TradeTrace | None = None


def _apply(
    state: TradingState,
    updates: dict,
) -> TradingState:
    merged = dict(updates)

    if "agent_logs" in merged:
        merged["agent_logs"] = state.agent_logs + merged["agent_logs"]

    return state.model_copy(update=merged)


def _run_local_pipeline(
    ticker: str,
    run_id: str | None = None,
    tick_id: int | None = None,
) -> TradingState:
    state = TradingState(
        ticker=ticker,
        run_id=run_id,
        tick_id=tick_id,
    )

    state = _apply(
        state,
        nodes.news_agent_node(state),
    )

    state = _apply(
        state,
        nodes.chart_agent_node(state),
    )

    state = _apply(
        state,
        nodes.signal_merger_node(state),
    )

    state = _apply(
        state,
        nodes.risk_agent_node(state),
    )

    return state


def _news_metadata(
    state: TradingState,
) -> dict:
    return {
        "news_availability": state.news_availability,
        "news_article_count": state.news_article_count,
        "news_source": state.news_source,
        "news_as_of": (
            state.news_as_of.isoformat() if state.news_as_of is not None else None
        ),
    }


def _decision_metadata(
    state: TradingState,
) -> dict:
    merged_signal = state.merged_signal

    return {
        "signal_direction": (
            merged_signal.direction if merged_signal is not None else None
        ),
        "signal_confidence": (
            merged_signal.confidence if merged_signal is not None else 0.0
        ),
        "merge_agreement": state.merge_agreement,
        "atr": state.atr,
        "entry_price": state.entry_price,
        "sector": state.sector,
        "proposed_shares": state.proposed_shares,
        "risk_decision": state.risk_decision.value,
        "risk_notes": list(state.risk_notes),
    }


def _position_metadata(
    ticker: str,
) -> dict:
    portfolio = nodes.get_ledger().snapshot()
    position = portfolio.positions.get(ticker)

    if position is None:
        return {
            "remaining_shares": 0.0,
            "position_closed": True,
        }

    remaining_shares = float(position.shares)

    return {
        "remaining_shares": remaining_shares,
        "position_closed": remaining_shares <= 1e-9,
    }


def _trade_side(
    state: TradingState,
) -> Literal["buy", "sell"]:
    if state.merged_signal is not None and state.merged_signal.direction == "bullish":
        return "buy"

    return "sell"


def _accounting_metadata(
    accounting,
) -> dict:
    if accounting is None:
        return {
            "average_cost": None,
            "realized_pnl": None,
        }

    return {
        "average_cost": accounting.average_cost,
        "realized_pnl": accounting.realized_pnl,
    }


def _build_trade_trace(
    state: TradingState,
    *,
    run_id: str,
    simulated_date: datetime | None,
    outcome: str,
    risk_proposed_shares: float,
    execution_attempted: bool,
    execution_success: bool | None,
    execution_side: str | None = None,
    execution_quantity: float | None = None,
    execution_price: float | None = None,
    execution_order_id: str | None = None,
    execution_notes: list[str] | None = None,
    accounting=None,
    remaining_shares: float | None = None,
    position_closed: bool | None = None,
    coordinator_approved: bool | None = None,
    coordinator_shares: float | None = None,
    coordinator_notes: list[str] | None = None,
) -> TradeTrace:
    news_signal = state.news_signal
    chart_signal = state.chart_signal
    merged_signal = state.merged_signal

    accounting_trace = AccountingTrace(
        average_cost=(accounting.average_cost if accounting is not None else None),
        realized_pnl=(accounting.realized_pnl if accounting is not None else None),
        remaining_shares=remaining_shares,
        position_closed=position_closed,
    )

    return TradeTrace(
        run_id=run_id,
        ticker=state.ticker,
        timestamp=state.timestamp,
        simulated_date=simulated_date,
        news=NewsTrace(
            direction=(news_signal.direction if news_signal is not None else None),
            confidence=(news_signal.confidence if news_signal is not None else None),
            availability=state.news_availability,
            article_count=state.news_article_count,
            source=state.news_source,
            as_of=state.news_as_of,
        ),
        chart=ChartTrace(
            direction=(chart_signal.direction if chart_signal is not None else None),
            confidence=(chart_signal.confidence if chart_signal is not None else None),
        ),
        signal=SignalTrace(
            direction=(merged_signal.direction if merged_signal is not None else None),
            confidence=(
                merged_signal.confidence if merged_signal is not None else None
            ),
            agreement=state.merge_agreement,
        ),
        risk=RiskTrace(
            approved=(state.risk_decision == RiskDecision.APPROVED),
            raw_shares=risk_proposed_shares,
            proposed_shares=risk_proposed_shares,
            final_shares=(coordinator_shares if coordinator_approved else 0.0),
            checks=list(state.risk_notes),
        ),
        coordinator=CoordinatorTrace(
            approved=coordinator_approved,
            shares=coordinator_shares,
            notes=list(coordinator_notes or []),
        ),
        execution=ExecutionTrace(
            attempted=execution_attempted,
            success=execution_success,
            side=execution_side,
            quantity=execution_quantity,
            price=execution_price,
            order_id=execution_order_id,
            notes=list(execution_notes or []),
        ),
        accounting=accounting_trace,
        metadata={
            "outcome": outcome,
            "tick_id": state.tick_id,
            "sector": state.sector,
            "atr": state.atr,
            "entry_price": state.entry_price,
        },
    )


async def run_tick(
    tickers: list[str],
    *,
    run_id: str = "runtime",
    simulated_date: datetime | None = None,
) -> dict[str, TickResult]:
    states = await asyncio.gather(
        *(
            asyncio.to_thread(
                _run_local_pipeline,
                ticker,
                run_id,
                index,
            )
            for index, ticker in enumerate(tickers)
        )
    )

    corr_matrix = build_correlation_matrix(tickers)

    portfolio = nodes.get_ledger().snapshot()

    portfolio = portfolio.model_copy(
        update={
            "correlation_matrix": corr_matrix,
        }
    )

    config = nodes.get_risk_config()

    proposals = [
        TickerProposal(
            ticker=state.ticker,
            sector=state.sector,
            entry_price=state.entry_price,
            proposed_shares=state.proposed_shares,
            combined_confidence=(
                state.merged_signal.confidence if state.merged_signal else 0.0
            ),
            side=_trade_side(state),
        )
        for state in states
        if state.risk_decision == RiskDecision.APPROVED
    ]

    coordinator_decisions = run_portfolio_coordinator(
        proposals,
        portfolio,
        config,
    )

    results: dict[str, TickResult] = {}

    for state in states:
        news = _news_metadata(state)
        telemetry = _decision_metadata(state)

        risk_proposed_shares = state.proposed_shares

        if state.risk_decision != RiskDecision.APPROVED:
            nodes.hold_node(state)

            position_metadata = _position_metadata(
                state.ticker,
            )

            trace = _build_trade_trace(
                state,
                run_id=run_id,
                simulated_date=simulated_date,
                outcome="held_local_reject",
                risk_proposed_shares=risk_proposed_shares,
                execution_attempted=False,
                execution_success=None,
                remaining_shares=position_metadata["remaining_shares"],
                position_closed=position_metadata["position_closed"],
                coordinator_approved=None,
                coordinator_shares=None,
                coordinator_notes=[],
            )

            results[state.ticker] = TickResult(
                outcome="held_local_reject",
                notes=("; ".join(state.risk_notes) or "held"),
                shares=0.0,
                price=state.entry_price,
                **telemetry,
                execution_attempted=False,
                execution_success=None,
                **position_metadata,
                **news,
                trade_trace=trace,
            )

            continue

        decision = coordinator_decisions.get(state.ticker)

        if decision is not None and decision.approved:
            state = _apply(
                state,
                {
                    "proposed_shares": decision.shares,
                },
            )

            exec_updates = nodes.execution_agent_node(state)

            execution_success = bool(
                exec_updates.get(
                    "execution_success",
                    False,
                )
            )

            execution_notes = exec_updates.get(
                "execution_notes",
                "",
            )

            execution_side = exec_updates.get(
                "execution_side",
            )

            execution_quantity = exec_updates.get(
                "execution_quantity",
            )

            execution_price = exec_updates.get(
                "execution_price",
            )

            execution_order_id = exec_updates.get(
                "execution_order_id",
            )

            accounting = exec_updates.get(
                "accounting",
            )

            accounting_metadata = _accounting_metadata(accounting)

            position_metadata = _position_metadata(
                state.ticker,
            )

            telemetry = _decision_metadata(state)

            outcome = "executed" if execution_success else "held_execution_reject"

            trace = _build_trade_trace(
                state,
                run_id=run_id,
                simulated_date=simulated_date,
                outcome=outcome,
                risk_proposed_shares=risk_proposed_shares,
                execution_attempted=True,
                execution_success=execution_success,
                execution_side=execution_side,
                execution_quantity=execution_quantity,
                execution_price=execution_price,
                execution_order_id=execution_order_id,
                execution_notes=([execution_notes] if execution_notes else []),
                accounting=accounting,
                remaining_shares=position_metadata["remaining_shares"],
                position_closed=position_metadata["position_closed"],
                coordinator_approved=True,
                coordinator_shares=decision.shares,
                coordinator_notes=list(decision.notes),
            )

            if execution_success:
                results[state.ticker] = TickResult(
                    outcome="executed",
                    notes=execution_notes,
                    shares=decision.shares,
                    price=state.entry_price,
                    **telemetry,
                    execution_attempted=True,
                    execution_success=True,
                    execution_side=execution_side,
                    **accounting_metadata,
                    **position_metadata,
                    **news,
                    trade_trace=trace,
                )

            else:
                results[state.ticker] = TickResult(
                    outcome="held_execution_reject",
                    notes=(
                        execution_notes
                        or "Broker execution failed; trade not executed."
                    ),
                    shares=0.0,
                    price=state.entry_price,
                    **telemetry,
                    execution_attempted=True,
                    execution_success=False,
                    execution_side=execution_side,
                    **accounting_metadata,
                    **position_metadata,
                    **news,
                    trade_trace=trace,
                )

        else:
            book_notes = state.risk_notes + (decision.notes if decision else [])

            state = _apply(
                state,
                {
                    "risk_notes": book_notes,
                },
            )

            nodes.hold_node(state)

            position_metadata = _position_metadata(
                state.ticker,
            )

            telemetry = _decision_metadata(state)

            trace = _build_trade_trace(
                state,
                run_id=run_id,
                simulated_date=simulated_date,
                outcome="held_book_reject",
                risk_proposed_shares=risk_proposed_shares,
                execution_attempted=False,
                execution_success=None,
                remaining_shares=position_metadata["remaining_shares"],
                position_closed=position_metadata["position_closed"],
                coordinator_approved=False,
                coordinator_shares=(decision.shares if decision is not None else None),
                coordinator_notes=(
                    list(decision.notes) if decision is not None else []
                ),
            )

            results[state.ticker] = TickResult(
                outcome="held_book_reject",
                notes=("; ".join(book_notes) or "held at book level"),
                shares=0.0,
                price=state.entry_price,
                **telemetry,
                execution_attempted=False,
                execution_success=None,
                **position_metadata,
                **news,
                trade_trace=trace,
            )

    return results
