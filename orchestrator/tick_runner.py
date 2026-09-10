"""
Multi-ticker tick orchestrator.

Matches graph/nodes.py's singleton architecture: RiskConfig, the
PortfolioLedger, and the Broker are process-lifetime singletons owned by
graph.nodes (get_risk_config/get_ledger/get_broker), not parameters this
file threads through. Reusing those exact singletons means a per-ticker
risk_agent_node call and this file's book-level coordinator check are
always looking at the identical portfolio state -- not two independently
derived copies that could silently drift.

Sector lookup: each ticker's local risk_agent_node result already carries
its own "sector" (from config.sectors.get_sector()) -- reused directly,
not re-looked-up here.

Known gap, not silently worked around: PortfolioLedger.snapshot()'s own
correlation matrix only covers currently HELD tickers. For the "two
brand-new, never-held tickers correlated with each other" case
(coordinator's second self-check) to work, this file rebuilds the matrix
over held-plus-candidate tickers before calling the coordinator -- local
risk_agent_node calls still use the ledger's own (held-only) matrix,
which is correct for what they're checking.

Stage 1: every ticker's local News->Chart->Merge->RiskAgent subgraph runs
concurrently (build_coordination_subgraph) -- nothing executes yet.
Stage 2: locally-approved tickers become TickerProposals; the coordinator
runs once on the full batch, book-level.
Stage 3: only coordinator-approved trades execute, via the same broker
singleton every node already uses -- and the ledger is updated the same
way execution_agent_node updates it, so ledger and broker never drift
apart just because this path bypassed that node.

DEBUG LOGGING: this file logs at each stage (local subgraph result ->
proposal -> coordinator decision -> execution) so a zero-trades run can
be traced to the exact stage it stalled at. Enable with:

    import logging
    logging.basicConfig(level=logging.INFO)

or set the "tick_orchestrator" logger's level directly if your project
already configures logging elsewhere.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass

from agents.execution_agent import ExecutionResult, run_execution_agent
from agents.execution_agent import RiskDecision as ExecAgentRiskDecision
from agents.signal_merger import MergedSignal
from graph.build import build_coordination_subgraph
from graph.nodes import get_broker, get_ledger, get_risk_config
from graph.state import RiskDecision, TradingState
from portfolio.coordinator import TickerProposal, run_portfolio_coordinator
from portfolio.state import PortfolioSnapshot, build_correlation_matrix

logger = logging.getLogger(__name__)


@dataclass
class TickResult:
    ticker: str
    outcome: str  # "executed" | "held_local_reject" | "held_book_reject" | "held_execution_failed"
    notes: str


def _portfolio_for_coordinator(tickers: list[str]) -> PortfolioSnapshot:
    base = get_ledger().snapshot()
    universe = sorted(set(tickers) | set(base.positions.keys()))
    correlation_matrix = build_correlation_matrix(universe)
    logger.info(
        "[Portfolio] equity=%s starting_equity=%s held=%s universe=%s",
        base.equity,
        base.starting_equity,
        list(base.positions.keys()),
        universe,
    )
    return PortfolioSnapshot(
        equity=base.equity,
        starting_equity=base.starting_equity,
        positions=base.positions,
        correlation_matrix=correlation_matrix,
    )


async def _run_one_ticker(ticker: str) -> dict:
    graph = build_coordination_subgraph()
    return await graph.ainvoke(TradingState(ticker=ticker))


async def run_tick(tickers: list[str]) -> dict[str, TickResult]:
    config = get_risk_config()
    broker = get_broker()
    ledger = get_ledger()

    local_results = await asyncio.gather(*(_run_one_ticker(t) for t in tickers))

    results: dict[str, TickResult] = {}
    proposals: list[TickerProposal] = []
    result_by_ticker: dict[str, dict] = {}

    for result in local_results:
        ticker = result["ticker"]
        result_by_ticker[ticker] = result

        merged_signal = result.get("merged_signal")
        logger.info(
            "[%s] local subgraph result: risk_decision=%s signal_direction=%s "
            "confidence=%s entry_price=%s proposed_shares=%s risk_notes=%s",
            ticker,
            result["risk_decision"],
            getattr(merged_signal, "direction", "n/a"),
            getattr(merged_signal, "confidence", "n/a"),
            result.get("entry_price"),
            result.get("proposed_shares"),
            result.get("risk_notes"),
        )

        if result["risk_decision"] != RiskDecision.APPROVED:
            results[ticker] = TickResult(
                ticker=ticker,
                outcome="held_local_reject",
                notes="; ".join(result["risk_notes"]) or "rejected locally",
            )
            logger.info("[%s] HELD (local reject): %s", ticker, results[ticker].notes)
            continue

        proposals.append(
            TickerProposal(
                ticker=ticker,
                sector=result["sector"],
                entry_price=result["entry_price"],
                proposed_shares=result["proposed_shares"],
                combined_confidence=merged_signal.confidence or 0.0,
            )
        )
        logger.info(
            "[%s] proposal created: shares=%s confidence=%s",
            ticker,
            result["proposed_shares"],
            merged_signal.confidence,
        )

    logger.info(
        "Locally-approved proposals this tick: %s", [p.ticker for p in proposals]
    )

    portfolio = _portfolio_for_coordinator(tickers)
    coordinator_decisions = (
        run_portfolio_coordinator(proposals, portfolio, config) if proposals else {}
    )

    if not proposals:
        logger.info(
            "No proposals reached the coordinator -- every ticker was locally rejected."
        )

    for proposal in proposals:
        decision = coordinator_decisions[proposal.ticker]
        logger.info(
            "[%s] coordinator decision: approved=%s shares=%s notes=%s",
            proposal.ticker,
            decision.approved,
            getattr(decision, "shares", "n/a"),
            decision.notes,
        )

        if not decision.approved:
            results[proposal.ticker] = TickResult(
                ticker=proposal.ticker,
                outcome="held_book_reject",
                notes="; ".join(decision.notes),
            )
            logger.info(
                "[%s] HELD (book reject): %s",
                proposal.ticker,
                results[proposal.ticker].notes,
            )
            continue

        merged_signal = result_by_ticker[proposal.ticker]["merged_signal"]
        risk_result = ExecAgentRiskDecision(
            approved=True,
            ticker=proposal.ticker,
            proposed_shares=decision.shares,
        )
        merged_for_exec = MergedSignal(
            direction=merged_signal.direction,
            combined_confidence=merged_signal.confidence or 0.0,
            agreement=True,
            rationale=merged_signal.rationale or "",
        )
        exec_result: ExecutionResult = run_execution_agent(
            risk_result, merged_for_exec, broker
        )
        logger.info(
            "[%s] execution result: executed=%s order=%s notes=%s",
            proposal.ticker,
            exec_result.executed,
            exec_result.order,
            exec_result.notes,
        )

        if exec_result.executed and exec_result.order is not None:
            ledger.record_fill(
                ticker=proposal.ticker,
                side=exec_result.order.side,
                qty=exec_result.order.qty,
                price=exec_result.order.filled_avg_price or proposal.entry_price or 0.0,
                sector=proposal.sector,
            )
            logger.info(
                "[%s] ledger updated: side=%s qty=%s price=%s",
                proposal.ticker,
                exec_result.order.side,
                exec_result.order.qty,
                exec_result.order.filled_avg_price,
            )

        results[proposal.ticker] = TickResult(
            ticker=proposal.ticker,
            outcome="executed" if exec_result.executed else "held_execution_failed",
            notes=exec_result.notes,
        )

    logger.info("Tick complete: %s", {t: r.outcome for t, r in results.items()})
    return results
