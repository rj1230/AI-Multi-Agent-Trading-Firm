"""
Conditional edge logic.

Graph flow:
START -> [NewsAgent || ChartAgent] -> SignalMerger -> RiskAgent (local)
      -> conditional edge -> {ExecutionAgent | HoldNode} -> END
(PortfolioRiskCoordinator runs across all per-ticker results when
multiple tickers are active — wired in build_graph.py, not here.)
"""


def route_after_risk_agent(state: dict) -> str:
    """Returns 'execution' or 'hold' based on state['risk_decision']['risk_approved']."""
    raise NotImplementedError
