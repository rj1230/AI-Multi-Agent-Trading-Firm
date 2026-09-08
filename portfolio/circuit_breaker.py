"""
Tracks daily account equity drawdown. Trips at -3% and halts all NEW
executions for the session (existing positions may still be closed).
"""
from __future__ import annotations


class CircuitBreaker:
    def __init__(self, threshold_pct: float = -0.03):
        self.threshold_pct = threshold_pct
        self.tripped = False

    def check(self, starting_equity: float, current_equity: float) -> bool:
        """Updates self.tripped and returns it."""
        if starting_equity <= 0:
            raise ValueError("starting_equity must be > 0")
        pnl_pct = self.daily_pnl_pct(starting_equity, current_equity)
        self.tripped = pnl_pct <= self.threshold_pct
        return self.tripped

    @staticmethod
    def daily_pnl_pct(starting_equity: float, current_equity: float) -> float:
        """Exposed separately so callers (RiskAgent, the dashboard) can log
        the actual number, not just the pass/fail bool."""
        if starting_equity <= 0:
            return 0.0
        return (current_equity - starting_equity) / starting_equity
