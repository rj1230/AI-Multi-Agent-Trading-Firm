"""
Integration test for the multi-ticker orchestrator, using the same
patching patterns established in tests/test_nodes.py: fetch_ohlcv and
_ledger patched directly on graph.nodes, _broker_singleton reset +
_price_lookup for a real SimBroker.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone

import pytest

import graph.nodes as nodes_module
import portfolio.ledger as ledger_module
from data_sources.schemas import DataSourceMode, OHLCVBar, OHLCVSeries
from graph.state import Signal
from orchestrator.tick_runner import run_tick

FAKE_ATR = 10.0
FAKE_ENTRY_PRICE = 285.0  # sized so proposed value ~= 19% of a 100k account
FAKE_CONFIDENCE = {"GOOGL": 0.9, "MSFT": 0.8, "AAPL": 0.7}


def _fake_series(ticker: str, lookback_days: int) -> OHLCVSeries:
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    bar = OHLCVBar(
        timestamp=start,
        open=FAKE_ENTRY_PRICE,
        high=FAKE_ENTRY_PRICE + 1,
        low=FAKE_ENTRY_PRICE - 1,
        close=FAKE_ENTRY_PRICE,
        volume=1_000_000,
    )
    return OHLCVSeries(
        ticker=ticker,
        bars=[bar],
        source="test",
        mode=DataSourceMode.LIVE,
        as_of=start + timedelta(days=1),
    )


def _fake_news(ticker: str) -> Signal:
    return Signal(
        direction="bullish", confidence=FAKE_CONFIDENCE[ticker], rationale="mocked news"
    )


def _fake_chart(ticker: str) -> Signal:
    return Signal(
        direction="bullish",
        confidence=FAKE_CONFIDENCE[ticker],
        rationale="mocked chart",
    )


@pytest.fixture
def patched_singletons(tmp_path, monkeypatch):
    monkeypatch.setattr(
        nodes_module,
        "_ledger",
        lambda: ledger_module.PortfolioLedger(
            starting_equity=100_000.0, path=tmp_path / "ledger.json"
        ),
    )
    monkeypatch.setattr(nodes_module, "_broker_singleton", None)
    monkeypatch.setattr(nodes_module, "_price_lookup", lambda t: FAKE_ENTRY_PRICE)
    monkeypatch.setattr(nodes_module, "run_news_agent", _fake_news)
    monkeypatch.setattr(nodes_module, "run_chart_agent", _fake_chart)
    monkeypatch.setattr(nodes_module, "fetch_ohlcv", _fake_series)
    monkeypatch.setattr(nodes_module, "compute_atr", lambda bars: FAKE_ATR)


def test_three_correlated_sector_proposals_highest_conviction_wins_end_to_end(
    patched_singletons,
):
    results = asyncio.run(run_tick(["GOOGL", "MSFT", "AAPL"]))

    # GOOGL/MSFT (Tech, per config/sectors.py) are the two highest-conviction
    # tickers and fit within the 40% sector cap; AAPL (also Tech, lowest
    # conviction) loses its slot once the running sector total crosses 40%.
    assert results["GOOGL"].outcome == "executed"
    assert results["MSFT"].outcome == "executed"
    assert results["AAPL"].outcome == "held_book_reject"

    positions = nodes_module.get_broker().get_positions()
    assert "GOOGL" in positions
    assert "MSFT" in positions
    assert "AAPL" not in positions
