"""
Graph wiring.

Matches §1.1 of the architecture doc:

    START -> [NewsAgent, ChartAgent] (parallel fan-out)
          -> SignalMerger (deterministic)
          -> RiskAgent
          -> conditional edge -> ExecutionAgent | HoldNode -> END

Phase 3 proved this shape with stubs; Phase 5/6 wires real
SignalMerger/RiskAgent/ExecutionAgent/HoldNode logic. The shape itself is
unchanged from Phase 3 -- what's new is that risk_agent and
execution_agent are now factories needing config/portfolio/broker bound
at construction time, since LangGraph nodes only ever receive `state`.

PortfolioRiskCoordinator is NOT wired into this graph yet -- see
graph/nodes.py module docstring. This remains a single-ticker pipeline
until that multi-ticker restructuring happens as its own change.
"""

from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from broker.protocol import Broker
from config.risk_config import RiskConfig
from graph.nodes import (
    chart_agent_node,
    hold_node,
    make_execution_agent_node,
    make_risk_agent_node,
    news_agent_node,
    route_after_risk,
    signal_merger_node,
)
from graph.state import TradingState
from portfolio.state import PortfolioSnapshot


def build_graph(config: RiskConfig, portfolio: PortfolioSnapshot, broker: Broker):
    builder = StateGraph(TradingState)

    builder.add_node("news_agent", news_agent_node)
    builder.add_node("chart_agent", chart_agent_node)
    builder.add_node("signal_merger", signal_merger_node)
    builder.add_node("risk_agent", make_risk_agent_node(config, portfolio))
    builder.add_node("execution_agent", make_execution_agent_node(broker))
    builder.add_node("hold_node", hold_node)

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
    # Manual smoke run. Uses SimBroker (no network calls, no real orders)
    # and a minimal fresh-account PortfolioSnapshot with an empty
    # correlation matrix -- fine for a wiring check, NOT a substitute for
    # the real per-tick correlation_matrix build (build_correlation_matrix)
    # a scheduler would do before invoking this for real.
    from broker.sim_broker import SimBroker
    from config.risk_config import load_risk_config

    config = load_risk_config()
    portfolio = PortfolioSnapshot(
        equity=100_000.0,
        starting_equity=100_000.0,
        positions={},
        correlation_matrix=None,
    )
    broker = SimBroker()

    graph = build_graph(config, portfolio, broker)
    result = graph.invoke(TradingState(ticker="AAPL"))
    for entry in result["agent_logs"]:
        print(f"[{entry.node}] {entry.message}")
