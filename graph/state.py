"""
Shared state for the trading graph.

Design rule from the architecture doc: nodes communicate ONLY through this
object, never directly with each other. That's what makes each node
independently inspectable, replayable, and swappable -- the contract is
"what's in the state," not "what this specific agent expects to receive."

Extended for Phase 5/6 wiring (additive only -- every new field is
Optional or defaulted, so nothing that already reads this file breaks):
  - atr, entry_price: sizing inputs RiskAgent needs, computed by
    risk_agent_node itself (see graph/nodes.py docstring for why this
    isn't bolted onto ChartAgent's Signal instead).
  - sector: looked up from config/sectors.py.
  - proposed_shares: RiskAgent's sizing output, carried through to
    ExecutionAgent so it doesn't have to recompute it.
  - execution_notes: ExecutionAgent/HoldNode's outcome, for the dashboard.
"""

from __future__ import annotations

import operator
from datetime import datetime, timezone
from enum import Enum
from typing import Annotated, Optional

from pydantic import BaseModel, Field


class AgentLogEntry(BaseModel):
    node: str
    message: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class Signal(BaseModel):
    direction: Optional[str] = None  # "bullish" | "bearish" | "neutral"
    confidence: Optional[float] = None
    rationale: Optional[str] = None


class RiskDecision(str, Enum):
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"


class TradingState(BaseModel):
    """The one object every node reads from and writes to."""

    ticker: str

    news_signal: Optional[Signal] = None
    chart_signal: Optional[Signal] = None
    merged_signal: Optional[Signal] = None

    risk_decision: RiskDecision = RiskDecision.PENDING
    risk_notes: list[str] = Field(default_factory=list)

    portfolio_snapshot: dict = Field(default_factory=dict)

    # --- Phase 5/6 additions (see module docstring) -----------------
    atr: Optional[float] = None
    entry_price: Optional[float] = None
    sector: Optional[str] = None
    proposed_shares: float = 0.0
    execution_notes: Optional[str] = None

    agent_logs: Annotated[list[AgentLogEntry], operator.add] = Field(
        default_factory=list
    )
