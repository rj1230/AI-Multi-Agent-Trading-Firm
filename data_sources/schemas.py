"""
Shared data contracts for the data source layer.

Both live.py (real-time) and historical.py (backtest replay) must return
these exact shapes. That's the whole point of Phase 2: ChartAgent and
NewsAgent should never be able to tell, from the data alone, whether
they're running in LIVE or BACKTEST mode.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field, field_validator


class DataSourceMode(str, Enum):
    LIVE = "live"
    BACKTEST = "backtest"


class OHLCVBar(BaseModel):
    """A single price bar."""

    timestamp: datetime
    open: float = Field(gt=0)
    high: float = Field(gt=0)
    low: float = Field(gt=0)
    close: float = Field(gt=0)
    volume: int = Field(ge=0)

    @field_validator("high")
    @classmethod
    def high_is_highest(cls, v, info):
        low = info.data.get("low")
        if low is not None and v < low:
            raise ValueError("high must be >= low")
        return v


class OHLCVSeries(BaseModel):
    """A ticker's price history for one fetch, tagged with provenance so
    you can always tell, from the object alone, where the data came from —
    important once yfinance fallback is in play."""

    ticker: str
    bars: list[OHLCVBar]
    source: str  # e.g. "alpaca", "yfinance", "historical_replay"
    mode: DataSourceMode
    as_of: datetime  # the "current" timestamp this data is valid up to

    @property
    def is_empty(self) -> bool:
        return len(self.bars) == 0

    @property
    def latest(self) -> Optional[OHLCVBar]:
        return self.bars[-1] if self.bars else None


class NewsArticle(BaseModel):
    title: str
    description: Optional[str] = None
    source: str
    published_at: datetime
    url: str


class NewsResult(BaseModel):
    """Wraps the article list so 'zero articles' is a typed, expected
    state (articles=[]) rather than None or a raised exception. NewsAgent
    downstream should branch on `is_empty`, not on truthiness of a raw list."""

    ticker: str
    articles: list[NewsArticle]
    source: str  # e.g. "newsapi", "historical_replay"
    mode: DataSourceMode
    as_of: datetime

    @property
    def is_empty(self) -> bool:
        return len(self.articles) == 0
