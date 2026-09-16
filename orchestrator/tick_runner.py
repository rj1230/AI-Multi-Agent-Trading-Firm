from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Literal

import graph.nodes as nodes
from graph.state import RiskDecision, TradingState
from portfolio.coordinator import TickerProposal, run_portfolio_coordinator
from portfolio.state import build_correlation_matrix


@dataclass
class TickResult:
    """
    Outcome of one ticker's pass through a single orchestration tick.

    The result contains both execution outcome and structured NewsAgent
    provenance so backtests can report data quality without parsing logs.
    """

    outcome: Literal[
        "executed",
        "held_local_reject",
        "held_book_reject",
        "held_execution_reject",
    ]

    notes: str
    shares: float
    price: float | None

    # --------------------------------------------------------------
    # News provenance
    # --------------------------------------------------------------

    news_availability: str | None = None
    news_article_count: int = 0
    news_source: str | None = None
    news_as_of: str | None = None


def _apply(
    state: TradingState,
    updates: dict,
) -> TradingState:
    """
    Merge node updates into TradingState.

    agent_logs are additive and therefore concatenated manually.
    """

    merged = dict(updates)

    if "agent_logs" in merged:
        merged["agent_logs"] = state.agent_logs + merged["agent_logs"]

    return state.model_copy(update=merged)


def _run_local_pipeline(
    ticker: str,
) -> TradingState:
    """
    Run:

        NewsAgent -> ChartAgent -> SignalMerger -> RiskAgent
    """

    state = TradingState(ticker=ticker)

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
    """
    Extract structured NewsAgent metadata from TradingState.
    """

    return {
        "news_availability": state.news_availability,
        "news_article_count": state.news_article_count,
        "news_source": state.news_source,
        "news_as_of": (
            state.news_as_of.isoformat() if state.news_as_of is not None else None
        ),
    }


async def run_tick(
    tickers: list[str],
) -> dict[str, TickResult]:
    """
    Run one multi-ticker orchestration cycle.

    Pipeline:

        Local RiskAgent
            ->
        Portfolio Coordinator
            ->
        ExecutionAgent
            ->
        Broker
            ->
        TickResult
    """

    # 1. Run local pipelines concurrently.
    states = await asyncio.gather(
        *(
            asyncio.to_thread(
                _run_local_pipeline,
                ticker,
            )
            for ticker in tickers
        )
    )

    # 2. Build portfolio-wide correlation matrix.
    corr_matrix = build_correlation_matrix(tickers)

    portfolio = nodes.get_ledger().snapshot()

    portfolio = portfolio.model_copy(update={"correlation_matrix": corr_matrix})

    config = nodes.get_risk_config()

    # 3. Convert locally-approved states into proposals.
    proposals = [
        TickerProposal(
            ticker=state.ticker,
            sector=state.sector,
            entry_price=state.entry_price,
            proposed_shares=state.proposed_shares,
            combined_confidence=(
                state.merged_signal.confidence if state.merged_signal else 0.0
            ),
        )
        for state in states
        if state.risk_decision == RiskDecision.APPROVED
    ]

    # 4. Run portfolio-level coordinator.
    coordinator_decisions = run_portfolio_coordinator(
        proposals,
        portfolio,
        config,
    )

    results: dict[str, TickResult] = {}

    # 5. Resolve each ticker's final outcome.
    for state in states:
        news = _news_metadata(state)

        # ----------------------------------------------------------
        # Local RiskAgent rejection
        # ----------------------------------------------------------

        if state.risk_decision != RiskDecision.APPROVED:
            nodes.hold_node(state)

            results[state.ticker] = TickResult(
                outcome="held_local_reject",
                notes=("; ".join(state.risk_notes) or "held"),
                shares=0.0,
                price=state.entry_price,
                **news,
            )

            continue

        decision = coordinator_decisions.get(state.ticker)

        # ----------------------------------------------------------
        # Portfolio approval -> execution
        # ----------------------------------------------------------

        if decision is not None and decision.approved:
            state = _apply(
                state,
                {"proposed_shares": decision.shares},
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

            if execution_success:
                results[state.ticker] = TickResult(
                    outcome="executed",
                    notes=execution_notes,
                    shares=decision.shares,
                    price=state.entry_price,
                    **news,
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
                    **news,
                )

        # ----------------------------------------------------------
        # Portfolio rejection
        # ----------------------------------------------------------

        else:
            book_notes = state.risk_notes + (decision.notes if decision else [])

            state = _apply(
                state,
                {"risk_notes": book_notes},
            )

            nodes.hold_node(state)

            results[state.ticker] = TickResult(
                outcome="held_book_reject",
                notes=("; ".join(book_notes) or "held at book level"),
                shares=0.0,
                price=state.entry_price,
                **news,
            )

    return results
