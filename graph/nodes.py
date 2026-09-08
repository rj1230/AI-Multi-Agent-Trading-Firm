"""
Phase 3 stub nodes.

Each function here is a placeholder for real agent logic (Phase 4/5/6).
All they do is append one AgentLogEntry and pass state through — the goal
is proving the graph's wiring and state-flow are correct BEFORE real logic
exists, per the doc's Phase 3 checkpoint.

Note on the conditional edge: RiskAgent's real logic doesn't exist yet, so
this stub uses a simple deterministic hook (ticker == "REJECT_TEST") purely
so both branches (ExecutionAgent / HoldNode) can be exercised in tests.
Replace this with real risk rule evaluation in Phase 5.
"""

from __future__ import annotations

from agents.chart_agent import run_chart_agent
from agents.news_agent import run_news_agent
from graph.state import AgentLogEntry, RiskDecision, Signal, TradingState


def _log(node: str, message: str) -> list[AgentLogEntry]:
    # Return only the new entry — agent_logs is Annotated with operator.add
    # in TradingState, so LangGraph concatenates this onto existing state
    # rather than us needing to read-then-append manually. Returning the
    # full accumulated list here would double it up.
    return [AgentLogEntry(node=node, message=message)]


def news_agent_node(state: TradingState) -> dict:
    """Phase 4: real NewsAgent, replacing news_agent_stub in the graph."""
    signal = run_news_agent(state.ticker)
    return {
        "news_signal": signal,
        "agent_logs": _log(
            "NewsAgent",
            f"{signal.direction} (conf={signal.confidence:.2f}): {signal.rationale}",
        ),
    }


def chart_agent_node(state: TradingState) -> dict:
    """Phase 4: real ChartAgent, replacing chart_agent_stub in the graph."""
    signal = run_chart_agent(state.ticker)
    return {
        "chart_signal": signal,
        "agent_logs": _log(
            "ChartAgent",
            f"{signal.direction} (conf={signal.confidence:.2f}): {signal.rationale}",
        ),
    }


def news_agent_stub(state: TradingState) -> dict:
    return {
        "news_signal": Signal(direction="neutral", confidence=0.0, rationale="stub"),
        "agent_logs": _log("NewsAgent", f"stub signal for {state.ticker}"),
    }


def chart_agent_stub(state: TradingState) -> dict:
    return {
        "chart_signal": Signal(direction="neutral", confidence=0.0, rationale="stub"),
        "agent_logs": _log("ChartAgent", f"stub signal for {state.ticker}"),
    }


def signal_merger_stub(state: TradingState) -> dict:
    return {
        "merged_signal": Signal(
            direction="neutral", confidence=0.0, rationale="stub merge"
        ),
        "agent_logs": _log("SignalMerger", "stub merge (deterministic, no LLM)"),
    }


def risk_agent_stub(state: TradingState) -> dict:
    # Deterministic test hook only — real rule evaluation arrives in Phase 5.
    if state.ticker == "REJECT_TEST":
        decision = RiskDecision.REJECTED
        notes = ["stub: forced rejection for REJECT_TEST ticker"]
    else:
        decision = RiskDecision.APPROVED
        notes = ["stub: auto-approved, no real risk rules yet"]

    return {
        "risk_decision": decision,
        "risk_notes": notes,
        "agent_logs": _log("RiskAgent", f"stub decision: {decision.value}"),
    }


def execution_agent_stub(state: TradingState) -> dict:
    return {
        "agent_logs": _log(
            "ExecutionAgent", f"stub: would place order for {state.ticker}"
        ),
    }


def hold_node_stub(state: TradingState) -> dict:
    return {
        "agent_logs": _log("HoldNode", f"stub: holding, reason={state.risk_notes}"),
    }


def route_after_risk(state: TradingState) -> str:
    """Conditional edge: RiskAgent's decision routes to ExecutionAgent or
    HoldNode. Kept as a plain function (not a node) per LangGraph convention
    for conditional edges."""
    if state.risk_decision == RiskDecision.APPROVED:
        return "execution_agent"
    return "hold_node"
