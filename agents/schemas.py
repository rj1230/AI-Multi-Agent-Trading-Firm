"""
Pydantic schemas for every agent's structured output — the guardrails layer.
LLM agent outputs (News, Chart) are validated against these before entering
shared state; a malformed response triggers one repair-prompt retry, then
falls back to a neutral/hold signal.
"""

from pydantic import BaseModel
from typing import Literal


class NewsSignal(BaseModel):
    ticker: str
    sentiment: Literal["bullish", "bearish", "neutral"]
    confidence: float
    key_headlines: list[str]


class ChartSignal(BaseModel):
    ticker: str
    signal: Literal["bullish", "bearish", "neutral"]
    confidence: float
    patterns_detected: list[str]


class RiskDecision(BaseModel):
    ticker: str
    risk_approved: bool
    position_size: float
    stop_price: float | None
    risk_notes: list[str]


class ExecutionResult(BaseModel):
    ticker: str
    order_id: str | None
    status: Literal["filled", "rejected", "held"]
