"""
harness/experiment.py — Typed experiment configuration and YAML loader.

Phase 1 scope: parse an experiment YAML (name, description, data window,
variants) into typed objects. Variant `overrides` are carried through as
plain dicts for now — actually applying them to a backtest run (toggling
news_enabled, chart_enabled, risk.max_position_pct, etc.) is not wired up
yet; see the note in runner.py.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class Variant:
    name: str
    overrides: dict[str, Any] = field(default_factory=dict)


@dataclass
class ExperimentData:
    tickers: list[str]
    start: str
    end: str


@dataclass
class ExperimentConfig:
    experiment_name: str
    description: str
    data: ExperimentData
    variants: list[Variant]

    @classmethod
    def from_yaml(cls, path: str | Path) -> ExperimentConfig:
        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))

        data = ExperimentData(
            tickers=raw["data"]["tickers"],
            start=str(raw["data"]["start"]),
            end=str(raw["data"]["end"]),
        )
        variants = [
            Variant(name=v["name"], overrides=v.get("overrides") or {})
            for v in raw["variants"]
        ]
        return cls(
            experiment_name=raw["experiment_name"],
            description=raw.get("description", ""),
            data=data,
            variants=variants,
        )
