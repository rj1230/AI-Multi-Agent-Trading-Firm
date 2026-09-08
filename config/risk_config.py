"""
Risk configuration, loaded once at startup from mandates/*.yaml.

Integration note: if you already have a `config/` package with its own
Pydantic base settings, drop this into it rather than creating a parallel
one -- the only thing that matters is that `load_risk_config()` returns a
validated RiskConfig before graph construction.
"""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, Field, model_validator


class PositionSizingConfig(BaseModel):
    equity_risk_pct: float = Field(gt=0, le=0.10)
    atr_stop_multiple: float = Field(gt=0)


class CircuitBreakerConfig(BaseModel):
    daily_drawdown_pct: float = Field(lt=0)  # e.g. -0.03


class CorrelationConfig(BaseModel):
    max_correlation: float = Field(gt=0, le=1.0)


class RiskConfig(BaseModel):
    position_sizing: PositionSizingConfig
    per_ticker_cap_pct: float = Field(gt=0, le=1.0)
    per_sector_cap_pct: float = Field(gt=0, le=1.0)
    max_concurrent_positions: int = Field(gt=0)
    circuit_breaker: CircuitBreakerConfig
    correlation: CorrelationConfig
    sector_map: dict[str, str] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _sector_cap_not_below_ticker_cap(self) -> "RiskConfig":
        # A sector cap tighter than the ticker cap would make the ticker
        # cap unreachable and is almost certainly a config typo.
        if self.per_sector_cap_pct < self.per_ticker_cap_pct:
            raise ValueError(
                "per_sector_cap_pct cannot be smaller than per_ticker_cap_pct"
            )
        return self


def load_risk_config(path: str | Path = "mandates/default.yaml") -> RiskConfig:
    """Load and validate a risk mandate YAML file into a RiskConfig."""
    raw = yaml.safe_load(Path(path).read_text())
    return RiskConfig.model_validate(raw)
