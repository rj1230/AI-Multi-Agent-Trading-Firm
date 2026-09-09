"""
Graph wiring.

Matches section 1.1 of the architecture doc:

    START -> [NewsAgent, ChartAgent] (parallel fan-out)
          -> SignalMerger (deterministic)
          -> RiskAgent
          -> conditional edge -> ExecutionAgent | HoldNode -> END

Phase 3's stub nodes have been replaced with real logic (graph.nodes).
The graph's SHAPE is unchanged from Phase 3 -- exactly as that phase's
docstring said would happen: "changing graph.nodes' implementations, NOT
this file."

Scope note: this is still the single-ticker graph. PortfolioRiskCoordinator
(portfolio/coordinator.py, Phase 7) runs across ALL tickers' proposals
AFTER their individual subgraphs complete -- that's an orchestration layer
that wraps N invocations of this graph, not a node inside it. Wiring that
concurrent multi-ticker runner is separate work from this file.
"""

from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from graph.nodes import (
    chart_agent_node,
    execution_agent_node,
    hold_node,
    news_agent_node,
    risk_agent_node,
    route_after_risk,
    signal_merger_node,
)
from graph.state import TradingState


def build_graph():
    builder = StateGraph(TradingState)

    builder.add_node("news_agent", news_agent_node)
    builder.add_node("chart_agent", chart_agent_node)
    builder.add_node("signal_merger", signal_merger_node)
    builder.add_node("risk_agent", risk_agent_node)
    builder.add_node("execution_agent", execution_agent_node)
    builder.add_node("hold_node", hold_node)

    builder.add_edge(START, "news_agent")
    builder.add_edge(START, "chart_agent")

    builder.add_edge("news_agent", "signal_merger")
    builder.add_edge("chart_agent", "signal_merger")

    builder.add_edge("signal_merger", "risk_agent")

    builder.add_conditional_edges(
        "risk_agent",
        route_after_risk,
        {
            "execution_agent": "execution_agent",
            "hold_node": "hold_node",
        },
    )

    builder.add_edge("execution_agent", END)
    builder.add_edge("hold_node", END)

    return builder.compile()


def build_coordination_subgraph():
    """Same nodes as build_graph() through RiskAgent, but stops there and
    routes to END -- no execution. Used by orchestrator/tick_runner.py:
    a per-ticker RiskAgent can't see other tickers' proposals this same
    tick, so nothing should execute until PortfolioRiskCoordinator has
    seen the full batch."""
    builder = StateGraph(TradingState)

    builder.add_node("news_agent", news_agent_node)
    builder.add_node("chart_agent", chart_agent_node)
    builder.add_node("signal_merger", signal_merger_node)
    builder.add_node("risk_agent", risk_agent_node)

    builder.add_edge(START, "news_agent")
    builder.add_edge(START, "chart_agent")
    builder.add_edge("news_agent", "signal_merger")
    builder.add_edge("chart_agent", "signal_merger")
    builder.add_edge("signal_merger", "risk_agent")
    builder.add_edge("risk_agent", END)

    return builder.compile()


if __name__ == "__main__":
    graph = build_graph()
    result = graph.invoke(TradingState(ticker="AAPL"))
    for entry in result["agent_logs"]:
        print(f"[{entry.node}] {entry.message}")
