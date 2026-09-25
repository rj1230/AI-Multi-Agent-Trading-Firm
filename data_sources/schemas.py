"""
Shared data contracts for the data source layer.

Both live.py (real-time) and historical.py (backtest replay) must return
these exact shapes. That's the whole point of Phase 2: ChartAgent and
NewsAgent should never be able to tell, from the data alone, whether
they're running in LIVE or BACKTEST mode.

NewsResult additionally exposes explicit availability metadata so the
system can distinguish:

    - available + articles
    - available + no articles
    - unavailable historical data

This distinction is important for trustworthy backtesting. Missing
historical news must never be silently interpreted as "there was no news."
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum

from pydantic import BaseModel, Field, field_validator


class DataSourceMode(str, Enum):
    LIVE = "live"
    BACKTEST = "backtest"


class NewsAvailability(str, Enum):
    """
    Describes whether the news data source could provide valid data.

    AVAILABLE:
        The requested news source/cache exists and was successfully
        queried. The article list may still legitimately be empty.

    UNAVAILABLE:
        The requested historical news data does not exist or cannot
        currently be provided. This must not be interpreted as "there
        was no news."

    ERROR:
        The source was expected to be available but failed while being
        queried or parsed.
    """

    AVAILABLE = "available"
    UNAVAILABLE = "unavailable"
    ERROR = "error"


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
    source: str
    mode: DataSourceMode
    as_of: datetime

    @property
    def is_empty(self) -> bool:
        return len(self.bars) == 0

    @property
    def latest(self) -> OHLCVBar | None:
        return self.bars[-1] if self.bars else None


class NewsArticle(BaseModel):
    title: str
    description: str | None = None
    source: str
    published_at: datetime
    url: str


class NewsResult(BaseModel):
    """
    Typed result returned by both live and historical news providers.

    `is_empty` remains available for existing callers, but callers that
    need to distinguish missing data from genuinely empty news should use
    `availability`.

    Examples:

        AVAILABLE + articles
            Historical/live news was successfully obtained.

        AVAILABLE + no articles
            The source was available and there were no matching articles.

        UNAVAILABLE + no articles
            News data could not be provided. This is especially important
            during historical backtests where no archived news exists.
    """

    ticker: str
    articles: list[NewsArticle]
    source: str
    mode: DataSourceMode
    as_of: datetime

    availability: NewsAvailability = NewsAvailability.AVAILABLE

    @property
    def is_empty(self) -> bool:
        """Backward-compatible check for an empty article list."""
        return len(self.articles) == 0

    @property
    def is_available(self) -> bool:
        """True when the news source successfully provided data."""
        return self.availability == NewsAvailability.AVAILABLE

    @property
    def is_unavailable(self) -> bool:
        """True when news data could not be provided."""
        return self.availability == NewsAvailability.UNAVAILABLE

    @property
    def has_articles(self) -> bool:
        """True when at least one usable article is present."""
        return len(self.articles) > 0
