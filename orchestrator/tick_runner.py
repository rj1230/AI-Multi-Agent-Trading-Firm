from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Literal

import graph.nodes as nodes
from broker.sim_broker import SimBroker
from config.risk_config import load_risk_config
from data_sources import fetch_ohlcv, set_simulated_date
from graph.state import RiskDecision, TradingState
from portfolio.coordinator import TickerProposal, run_portfolio_coordinator
from portfolio.ledger import PortfolioLedger
from portfolio.state import build_correlation_matrix

# ---------------------------------------------------------
# TickResult
# ---------------------------------------------------------


@dataclass
class TickResult:
    """Outcome of one ticker's pass through a single tick."""

    outcome: Literal["executed", "held_local_reject", "held_book_reject"]
    notes: str
    shares: float
    price: float | None


def _apply(state: TradingState, updates: dict) -> TradingState:
    """Merge one node's returned updates into state, by hand -- we aren't
    going through a compiled LangGraph here, so agent_logs' operator.add
    reducer has to be replicated manually (concatenate, don't overwrite)."""
    merged = dict(updates)
    if "agent_logs" in merged:
        merged["agent_logs"] = state.agent_logs + merged["agent_logs"]
    return state.model_copy(update=merged)


def _run_local_pipeline(ticker: str) -> TradingState:
    """NewsAgent -> ChartAgent -> SignalMerger -> local RiskAgent for one
    ticker. Mirrors graph/nodes.py's node sequence exactly, so mocks
    patched onto graph.nodes (run_news_agent, fetch_ohlcv, etc.) are
    picked up the same way they would be inside the real compiled graph."""
    state = TradingState(ticker=ticker)
    state = _apply(state, nodes.news_agent_node(state))
    state = _apply(state, nodes.chart_agent_node(state))
    state = _apply(state, nodes.signal_merger_node(state))
    state = _apply(state, nodes.risk_agent_node(state))
    return state


async def run_tick(tickers: list[str]) -> dict[str, TickResult]:
    """The real multi-ticker orchestrator: local RiskAgent ->
    PortfolioRiskCoordinator -> ExecutionAgent, fanned out across
    `tickers` concurrently.

    Only locally-approved proposals reach the coordinator (section 4.2 --
    a per-ticker RiskAgent can't see the whole book, so its approval is
    necessary but not sufficient). The coordinator then admits proposals
    highest-conviction-first against a running simulated portfolio, so
    when the shared risk budget runs out, it's the lowest-conviction
    tickers that lose their slot.
    """
    # Each ticker's News/Chart/Merger/local-Risk pipeline runs
    # concurrently (I/O-bound: LLM calls + OHLCV fetches).
    states = await asyncio.gather(
        *(asyncio.to_thread(_run_local_pipeline, t) for t in tickers)
    )
    states_by_ticker = {s.ticker: s for s in states}

    # Built once per tick across the full ticker set (see portfolio/state.py
    # docstring) -- not once per ticker.
    corr_matrix = build_correlation_matrix(tickers)

    portfolio = nodes.get_ledger().snapshot()
    portfolio = portfolio.model_copy(update={"correlation_matrix": corr_matrix})
    config = nodes.get_risk_config()

    proposals = [
        TickerProposal(
            ticker=s.ticker,
            sector=s.sector,
            entry_price=s.entry_price,
            proposed_shares=s.proposed_shares,
            combined_confidence=(
                s.merged_signal.confidence if s.merged_signal else 0.0
            ),
        )
        for s in states
        if s.risk_decision == RiskDecision.APPROVED
    ]

    coordinator_decisions = run_portfolio_coordinator(proposals, portfolio, config)

    results: dict[str, TickResult] = {}

    for s in states:
        if s.risk_decision != RiskDecision.APPROVED:
            # Never reached the coordinator -- local RiskAgent already
            # said no (insufficient history, sizing failure, etc.).
            nodes.hold_node(s)
            results[s.ticker] = TickResult(
                outcome="held_local_reject",
                notes="; ".join(s.risk_notes) or "held",
                shares=0.0,
                price=s.entry_price,
            )
            continue

        decision = coordinator_decisions.get(s.ticker)

        if decision is not None and decision.approved:
            s = _apply(s, {"proposed_shares": decision.shares})
            exec_updates = nodes.execution_agent_node(s)
            results[s.ticker] = TickResult(
                outcome="executed",
                notes=exec_updates.get("execution_notes", ""),
                shares=decision.shares,
                price=s.entry_price,
            )
        else:
            book_notes = s.risk_notes + (decision.notes if decision else [])
            s = _apply(s, {"risk_notes": book_notes})
            nodes.hold_node(s)
            results[s.ticker] = TickResult(
                outcome="held_book_reject",
                notes="; ".join(book_notes) or "held at book level",
                shares=0.0,
                price=s.entry_price,
            )

    return results
