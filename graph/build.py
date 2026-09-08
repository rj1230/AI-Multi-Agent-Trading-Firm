"""
Graph skeleton wiring — Phase 3.

Matches §1.1 of the architecture doc:

    START -> [NewsAgent, ChartAgent] (parallel fan-out)
          -> SignalMerger (deterministic)
          -> RiskAgent
          -> conditional edge -> ExecutionAgent | HoldNode -> END

All nodes are stubs (graph.nodes) for this phase. Swapping in real agent
logic later means changing graph.nodes' implementations, NOT this file —
the graph's shape stays fixed once Phase 3 is validated.
"""

from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from graph.nodes import (
    chart_agent_node,
    execution_agent_stub,
    hold_node_stub,
    news_agent_node,
    risk_agent_stub,
    route_after_risk,
    signal_merger_stub,
)
from graph.state import TradingState


def build_graph():
    builder = StateGraph(TradingState)

    builder.add_node("news_agent", news_agent_node)
    builder.add_node("chart_agent", chart_agent_node)
    builder.add_node("signal_merger", signal_merger_stub)
    builder.add_node("risk_agent", risk_agent_stub)
    builder.add_node("execution_agent", execution_agent_stub)
    builder.add_node("hold_node", hold_node_stub)

    # Parallel fan-out from START
    builder.add_edge(START, "news_agent")
    builder.add_edge(START, "chart_agent")

    # Both feed into SignalMerger (LangGraph waits for both before firing)
    builder.add_edge("news_agent", "signal_merger")
    builder.add_edge("chart_agent", "signal_merger")

    builder.add_edge("signal_merger", "risk_agent")

    # Conditional edge: RiskAgent's decision routes to one of two branches
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


if __name__ == "__main__":
    # Quick manual run — mirrors the doc's Phase 3 checkpoint: confirm state
    # flows through every node in the right order.
    graph = build_graph()
    result = graph.invoke(TradingState(ticker="AAPL"))
    for entry in result["agent_logs"]:
        print(f"[{entry.node}] {entry.message}")
