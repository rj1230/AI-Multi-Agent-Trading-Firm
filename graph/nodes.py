"""
Graph nodes.

Phase 3 stubs proved the wiring; Phase 4 wired real News/Chart agents.
This pass wires SignalMerger, local RiskAgent, ExecutionAgent, and
HoldNode for real, replacing signal_merger_stub / risk_agent_stub /
execution_agent_stub / hold_node_stub.

PortfolioRiskCoordinator (portfolio/coordinator.py) is deliberately NOT
wired in here yet — it operates on a batch of proposals across ALL
tickers in one tick, which doesn't fit this single-ticker TradingState
shape. That's a separate, larger change to graph/build.py (multi-ticker
fan-out + a batch coordinator step before routing), tracked as the next
step after this one lands and passes tests.
"""

from __future__ import annotations

from agents.chart_agent import run_chart_agent
from agents.execution_agent import ExecutionResult, run_execution_agent
from agents.execution_agent import RiskDecision as ExecAgentRiskDecision
from agents.news_agent import run_news_agent
from agents.risk_agent import run_risk_agent
from agents.signal_merger import MergedSignal
from agents.signal_merger import Signal as MergerSignal
from agents.signal_merger import merge_signals
from broker.protocol import Broker
from config.risk_config import RiskConfig
from config.settings import ATR_PERIOD
from data_sources import fetch_ohlcv
from graph.state import AgentLogEntry, RiskDecision, Signal, TradingState
from portfolio.state import PortfolioSnapshot


def _log(node: str, message: str) -> list[AgentLogEntry]:
    # Return only the new entry — agent_logs is Annotated with operator.add
    # in TradingState, so LangGraph concatenates this onto existing state
    # rather than us needing to read-then-append manually.
    return [AgentLogEntry(node=node, message=message)]


# --- News / Chart (Phase 4, unchanged) --------------------------------------


def news_agent_node(state: TradingState) -> dict:
    signal = run_news_agent(state.ticker)
    return {
        "news_signal": signal,
        "agent_logs": _log(
            "NewsAgent",
            f"{signal.direction} (conf={signal.confidence:.2f}): {signal.rationale}",
        ),
    }


def chart_agent_node(state: TradingState) -> dict:
    signal = run_chart_agent(state.ticker)
    return {
        "chart_signal": signal,
        "agent_logs": _log(
            "ChartAgent",
            f"{signal.direction} (conf={signal.confidence:.2f}): {signal.rationale}",
        ),
    }


# --- SignalMerger (Phase 5) --------------------------------------------------


def signal_merger_node(state: TradingState) -> dict:
    """Real deterministic merge, replacing signal_merger_stub. Explicitly
    reconstructs agents.signal_merger.Signal from state's News/Chart
    signals rather than duck-typing across the module boundary — this
    re-validates direction/confidence through Pydantic before the merge
    runs, which is cheap insurance against a malformed upstream signal
    slipping through as a plain object."""
    news = MergerSignal(
        direction=state.news_signal.direction,
        confidence=state.news_signal.confidence,
        rationale=state.news_signal.rationale,
    )
    chart = MergerSignal(
        direction=state.chart_signal.direction,
        confidence=state.chart_signal.confidence,
        rationale=state.chart_signal.rationale,
    )
    merged: MergedSignal = merge_signals(news, chart)

    return {
        "merged_signal": Signal(
            direction=merged.direction,
            confidence=merged.combined_confidence,
            rationale=merged.rationale,
        ),
        "agent_logs": _log(
            "SignalMerger",
            f"{merged.direction} (conf={merged.combined_confidence:.2f}, "
            f"agreement={merged.agreement}): {merged.rationale}",
        ),
    }


# --- RiskAgent (Phase 5/6) ---------------------------------------------------


def _compute_atr_and_entry_price(
    ticker: str, period: int = ATR_PERIOD
) -> tuple[float, float]:
    """Computed here, not in ChartAgent -- ChartAgent's job is direction,
    RiskAgent's job is sizing. Uses the same dual-mode fetch_ohlcv() every
    other agent uses, so LIVE/BACKTEST stay identical. Standard true-range
    ATR: TR = max(high-low, |high-prev_close|, |low-prev_close|), averaged
    over `period` bars."""
    series = fetch_ohlcv(ticker, lookback_days=period + 5)
    if series.is_empty or len(series.bars) < 2:
        return 0.0, 0.0

    bars = series.bars
    entry_price = bars[-1].close

    true_ranges = []
    for i in range(1, len(bars)):
        high, low, prev_close = bars[i].high, bars[i].low, bars[i - 1].close
        true_ranges.append(
            max(high - low, abs(high - prev_close), abs(low - prev_close))
        )

    recent = true_ranges[-period:] if len(true_ranges) >= period else true_ranges
    atr = sum(recent) / len(recent) if recent else 0.0
    return atr, entry_price


def make_risk_agent_node(config: RiskConfig, portfolio: PortfolioSnapshot):
    """Factory, not a bare function -- config and portfolio are bound once
    at graph-construction time (see graph/build.py), not re-fetched per
    node call. portfolio must already have a fresh correlation_matrix
    (via build_correlation_matrix()) built once per tick by the caller,
    per portfolio/state.py's docstring."""

    def risk_agent_node(state: TradingState) -> dict:
        sector = config.sector_map.get(state.ticker)
        if sector is None:
            note = f"No sector mapping for {state.ticker} -- trade blocked pending config update."
            return {
                "risk_decision": RiskDecision.REJECTED,
                "risk_notes": [note],
                "proposed_shares": 0.0,
                "agent_logs": _log("RiskAgent", f"rejected: {note}"),
            }

        atr, entry_price = _compute_atr_and_entry_price(state.ticker)

        merged_for_risk = MergerSignal(
            direction=state.merged_signal.direction,
            confidence=state.merged_signal.confidence or 0.0,
            rationale=state.merged_signal.rationale or "",
        )
        # run_risk_agent's type hint says MergedSignal but only reads
        # .direction -- MergerSignal (the pre-merge Signal shape) satisfies
        # that structurally. Kept explicit rather than passing MergedSignal
        # itself since state doesn't retain the .agreement flag separately.

        decision = run_risk_agent(
            merged_signal=merged_for_risk,
            ticker=state.ticker,
            sector=sector,
            entry_price=entry_price,
            atr=atr,
            portfolio=portfolio,
            config=config,
        )

        return {
            "risk_decision": RiskDecision.APPROVED
            if decision.approved
            else RiskDecision.REJECTED,
            "risk_notes": decision.risk_notes,
            "proposed_shares": decision.proposed_shares,
            "agent_logs": _log(
                "RiskAgent",
                f"{'approved' if decision.approved else 'rejected'}: "
                + "; ".join(decision.risk_notes),
            ),
        }

    return risk_agent_node


# --- ExecutionAgent / HoldNode (Phase 6) -------------------------------------


def make_execution_agent_node(broker: Broker):
    def execution_agent_node(state: TradingState) -> dict:
        risk_result = ExecAgentRiskDecision(
            approved=(state.risk_decision == RiskDecision.APPROVED),
            ticker=state.ticker,
            proposed_shares=state.proposed_shares,
        )
        merged_for_exec = MergedSignal(
            direction=state.merged_signal.direction,
            combined_confidence=state.merged_signal.confidence or 0.0,
            agreement=True,  # unused by run_execution_agent; only .direction is read
            rationale=state.merged_signal.rationale or "",
        )

        result: ExecutionResult = run_execution_agent(
            risk_result, merged_for_exec, broker
        )

        return {
            "agent_logs": _log("ExecutionAgent", result.notes),
        }

    return execution_agent_node


def hold_node(state: TradingState) -> dict:
    return {
        "agent_logs": _log(
            "HoldNode",
            f"holding {state.ticker}, reason: {'; '.join(state.risk_notes) or 'no notes'}",
        ),
    }


def route_after_risk(state: TradingState) -> str:
    """Conditional edge: unchanged from Phase 3."""
    if state.risk_decision == RiskDecision.APPROVED:
        return "execution_agent"
    return "hold_node"
