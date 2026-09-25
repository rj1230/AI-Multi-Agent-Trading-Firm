from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from graph.state import RiskDecision, Signal, TradingState
from orchestrator import tick_runner
from portfolio.state import PortfolioSnapshot


def _state(
    ticker: str,
    *,
    direction: str = "bullish",
    confidence: float = 0.90,
    risk: RiskDecision = RiskDecision.APPROVED,
    proposed_shares: float = 10.0,
    entry_price: float = 100.0,
    sector: str = "Technology",
) -> TradingState:
    """
    Build a deterministic TradingState for tick-runner tests.
    """

    return TradingState(
        ticker=ticker,
        news_signal=Signal(
            direction=direction,
            confidence=confidence,
            rationale="test news signal",
        ),
        chart_signal=Signal(
            direction=direction,
            confidence=confidence,
            rationale="test chart signal",
        ),
        merged_signal=Signal(
            direction=direction,
            confidence=confidence,
            rationale="test merged signal",
        ),
        merge_agreement=True,
        news_availability="AVAILABLE",
        news_article_count=0,
        news_source="test",
        news_as_of=None,
        risk_decision=risk,
        risk_notes=[],
        atr=2.0,
        entry_price=entry_price,
        sector=sector,
        proposed_shares=proposed_shares,
    )


def _portfolio_snapshot() -> PortfolioSnapshot:
    """
    Real PortfolioSnapshot model.

    run_tick() calls .model_copy(), so the test must provide the
    actual Pydantic model rather than SimpleNamespace.
    """

    return PortfolioSnapshot(
        equity=100_000.0,
        starting_equity=100_000.0,
        positions={},
        correlation_matrix=None,
    )


def _risk_config():
    """
    The coordinator itself is mocked in these tests, so the actual
    RiskConfig fields are not needed.
    """

    return SimpleNamespace()


def _common_patches(state: TradingState):
    """
    Common isolation patches.

    build_correlation_matrix() is mocked because these tests are
    testing tick-runner orchestration, not historical OHLCV
    correlation calculations.
    """

    return [
        patch.object(
            tick_runner,
            "_run_local_pipeline",
            return_value=state,
        ),
        patch.object(
            tick_runner,
            "build_correlation_matrix",
            return_value={},
        ),
        patch.object(
            tick_runner.nodes,
            "get_ledger",
            return_value=SimpleNamespace(
                snapshot=_portfolio_snapshot,
            ),
        ),
        patch.object(
            tick_runner.nodes,
            "get_risk_config",
            return_value=_risk_config(),
        ),
    ]


def test_run_tick_executes_coordinator_approved_trade():
    """
    Coordinator approves a locally-approved trade.

    Expected:
        execution is called
        execution succeeds
        broker-confirmed execution side is preserved
        TickResult is "executed"
    """

    state = _state(
        "AAPL",
        confidence=0.95,
        proposed_shares=10.0,
    )

    execution_calls = []

    def fake_execution(execution_state):
        execution_calls.append(execution_state)

        return {
            "execution_success": True,
            "execution_notes": "test execution succeeded",
            "execution_side": "buy",
        }

    patches = _common_patches(state)

    with (
        patches[0],
        patches[1],
        patches[2],
        patches[3],
        patch.object(
            tick_runner,
            "run_portfolio_coordinator",
            return_value={
                "AAPL": SimpleNamespace(
                    ticker="AAPL",
                    approved=True,
                    shares=10.0,
                    notes=["Approved at book level."],
                )
            },
        ),
        patch.object(
            tick_runner.nodes,
            "execution_agent_node",
            side_effect=fake_execution,
        ),
        patch.object(
            tick_runner.nodes,
            "hold_node",
        ),
    ):
        results = asyncio.run(tick_runner.run_tick(["AAPL"]))

    result = results["AAPL"]

    assert result.outcome == "executed"
    assert result.execution_attempted is True
    assert result.execution_success is True
    assert result.execution_side == "buy"

    assert result.shares == pytest.approx(10.0)
    assert result.proposed_shares == pytest.approx(10.0)

    assert result.signal_direction == "bullish"
    assert result.signal_confidence == pytest.approx(0.95)
    assert result.merge_agreement is True

    assert len(execution_calls) == 1
    assert execution_calls[0].ticker == "AAPL"
    assert execution_calls[0].proposed_shares == pytest.approx(10.0)


def test_run_tick_execution_failure_is_not_reported_as_executed():
    """
    Coordinator approves the trade, but execution fails.

    Expected:
        execution is attempted
        execution_success=False
        TickResult is "held_execution_reject"
        shares reported as zero
    """

    state = _state(
        "GOOGL",
        confidence=0.95,
        proposed_shares=10.0,
    )

    execution_calls = []

    def fake_execution(execution_state):
        execution_calls.append(execution_state)

        return {
            "execution_success": False,
            "execution_notes": "simulated broker rejection",
        }

    patches = _common_patches(state)

    with (
        patches[0],
        patches[1],
        patches[2],
        patches[3],
        patch.object(
            tick_runner,
            "run_portfolio_coordinator",
            return_value={
                "GOOGL": SimpleNamespace(
                    ticker="GOOGL",
                    approved=True,
                    shares=10.0,
                    notes=["Approved at book level."],
                )
            },
        ),
        patch.object(
            tick_runner.nodes,
            "execution_agent_node",
            side_effect=fake_execution,
        ),
        patch.object(
            tick_runner.nodes,
            "hold_node",
        ),
    ):
        results = asyncio.run(tick_runner.run_tick(["GOOGL"]))

    result = results["GOOGL"]

    assert result.outcome == "held_execution_reject"

    assert result.execution_attempted is True
    assert result.execution_success is False

    assert result.shares == pytest.approx(0.0)

    assert result.notes == "simulated broker rejection"

    assert len(execution_calls) == 1


def test_run_tick_local_risk_rejection_does_not_execute():
    """
    Local RiskAgent rejects the trade.

    Expected:
        ticker is not sent as a coordinator proposal
        execution is never called
        TickResult is "held_local_reject"
    """

    state = _state(
        "AAPL",
        confidence=0.90,
        risk=RiskDecision.REJECTED,
        proposed_shares=10.0,
    )

    execution_called = False

    def fake_execution(execution_state):
        nonlocal execution_called
        execution_called = True

        return {
            "execution_success": True,
            "execution_notes": "should not happen",
        }

    coordinator_calls = []

    def fake_coordinator(
        proposals,
        portfolio,
        config,
    ):
        coordinator_calls.append(proposals)
        return {}

    patches = _common_patches(state)

    with (
        patches[0],
        patches[1],
        patches[2],
        patches[3],
        patch.object(
            tick_runner,
            "run_portfolio_coordinator",
            side_effect=fake_coordinator,
        ),
        patch.object(
            tick_runner.nodes,
            "execution_agent_node",
            side_effect=fake_execution,
        ),
        patch.object(
            tick_runner.nodes,
            "hold_node",
        ),
    ):
        results = asyncio.run(tick_runner.run_tick(["AAPL"]))

    result = results["AAPL"]

    assert result.outcome == "held_local_reject"

    assert result.execution_attempted is False
    assert result.execution_success is None

    assert result.shares == pytest.approx(0.0)

    assert execution_called is False

    assert len(coordinator_calls) == 1


def test_run_tick_book_rejection_does_not_execute():
    """
    PortfolioRiskCoordinator rejects the trade.

    Expected:
        execution is never called
        TickResult is "held_book_reject"
    """

    state = _state(
        "MSFT",
        confidence=0.80,
        proposed_shares=10.0,
    )

    execution_called = False

    def fake_execution(execution_state):
        nonlocal execution_called
        execution_called = True

        return {
            "execution_success": True,
            "execution_notes": "should not execute",
        }

    patches = _common_patches(state)

    with (
        patches[0],
        patches[1],
        patches[2],
        patches[3],
        patch.object(
            tick_runner,
            "run_portfolio_coordinator",
            return_value={
                "MSFT": SimpleNamespace(
                    ticker="MSFT",
                    approved=False,
                    shares=0.0,
                    notes=["Book-level correlation rejection."],
                )
            },
        ),
        patch.object(
            tick_runner.nodes,
            "execution_agent_node",
            side_effect=fake_execution,
        ),
        patch.object(
            tick_runner.nodes,
            "hold_node",
        ),
    ):
        results = asyncio.run(tick_runner.run_tick(["MSFT"]))

    result = results["MSFT"]

    assert result.outcome == "held_book_reject"

    assert result.execution_attempted is False
    assert result.execution_success is None

    assert result.shares == pytest.approx(0.0)

    assert "Book-level correlation rejection." in result.notes

    assert execution_called is False


def test_run_tick_preserves_news_and_signal_metadata():
    """
    Successful execution preserves important telemetry from
    TradingState in TickResult.
    """

    state = _state(
        "JPM",
        confidence=0.91,
        proposed_shares=5.0,
        entry_price=192.50,
        sector="Financials",
    )

    with (
        patch.object(
            tick_runner,
            "_run_local_pipeline",
            return_value=state,
        ),
        patch.object(
            tick_runner,
            "build_correlation_matrix",
            return_value={},
        ),
        patch.object(
            tick_runner.nodes,
            "get_ledger",
            return_value=SimpleNamespace(
                snapshot=_portfolio_snapshot,
            ),
        ),
        patch.object(
            tick_runner.nodes,
            "get_risk_config",
            return_value=_risk_config(),
        ),
        patch.object(
            tick_runner,
            "run_portfolio_coordinator",
            return_value={
                "JPM": SimpleNamespace(
                    ticker="JPM",
                    approved=True,
                    shares=5.0,
                    notes=["Approved at book level."],
                )
            },
        ),
        patch.object(
            tick_runner.nodes,
            "execution_agent_node",
            return_value={
                "execution_success": True,
                "execution_notes": "filled",
                "execution_side": "buy",
            },
        ),
        patch.object(
            tick_runner.nodes,
            "hold_node",
        ),
    ):
        results = asyncio.run(tick_runner.run_tick(["JPM"]))

    result = results["JPM"]

    assert result.outcome == "executed"

    # News provenance.
    assert result.news_availability == "AVAILABLE"
    assert result.news_article_count == 0
    assert result.news_source == "test"
    assert result.news_as_of is None

    # Signal telemetry.
    assert result.signal_direction == "bullish"
    assert result.signal_confidence == pytest.approx(0.91)
    assert result.merge_agreement is True

    # Execution telemetry.
    assert result.execution_attempted is True
    assert result.execution_success is True
    assert result.execution_side == "buy"

    assert result.shares == pytest.approx(5.0)
    assert result.proposed_shares == pytest.approx(5.0)
    assert result.price == pytest.approx(192.50)
    assert result.atr == pytest.approx(2.0)
    assert result.sector == "Financials"
