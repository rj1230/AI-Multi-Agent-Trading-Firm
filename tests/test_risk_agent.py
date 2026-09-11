import pandas as pd
import pytest
from config.risk_config import RiskConfig
from agents.signal_merger import MergedSignal
from portfolio.state import PortfolioSnapshot, Position
from agents.risk_agent import run_risk_agent


@pytest.fixture
def config():
    return RiskConfig.model_validate(
        {
            "position_sizing": {"equity_risk_pct": 0.01, "atr_stop_multiple": 1.5},
            "per_ticker_cap_pct": 0.20,
            "per_sector_cap_pct": 0.40,
            "max_concurrent_positions": 3,
            "circuit_breaker": {"daily_drawdown_pct": -0.03},
            "correlation": {"max_correlation": 0.7},
        }
    )


@pytest.fixture
def bullish_signal():
    return MergedSignal(
        direction="bullish", combined_confidence=0.65, agreement=True, rationale="test"
    )


@pytest.fixture
def clean_portfolio():
    return PortfolioSnapshot(equity=100_000.0, starting_equity=100_000.0, positions={})


def test_neutral_signal_is_never_approved(config, clean_portfolio):
    neutral = MergedSignal(
        direction="neutral", combined_confidence=0.0, agreement=False, rationale="hold"
    )
    decision = run_risk_agent(
        neutral, "AAPL", "Tech", 190.0, 3.0, clean_portfolio, config
    )
    assert decision.approved is False
    assert "neutral" in decision.risk_notes[-1].lower()


def test_clean_trade_is_approved_with_correct_sizing(
    config, clean_portfolio, bullish_signal
):
    decision = run_risk_agent(
        bullish_signal, "AAPL", "Tech", 100.0, 10.0, clean_portfolio, config
    )
    assert decision.approved is True
    # Well within both caps, so raw and capped sizes should match exactly.
    assert decision.proposed_shares == pytest.approx(1000 / 15.0, rel=1e-3)
    assert decision.raw_shares == pytest.approx(decision.proposed_shares, rel=1e-6)


def test_circuit_breaker_blocks_trade_on_bad_day(config, bullish_signal):
    portfolio = PortfolioSnapshot(
        equity=95_000.0, starting_equity=100_000.0, positions={}
    )
    decision = run_risk_agent(
        bullish_signal, "AAPL", "Tech", 100.0, 2.0, portfolio, config
    )
    assert decision.approved is False
    breaker_note = next(c for c in decision.checks if c.rule == "circuit_breaker")
    assert breaker_note.passed is False


def test_concurrent_positions_cap_blocks_fourth_new_ticker(config, bullish_signal):
    positions = {
        t: Position(ticker=t, shares=10, sector="Tech", market_value=1000)
        for t in ["MSFT", "GOOGL", "NVDA"]
    }
    portfolio = PortfolioSnapshot(
        equity=100_000.0, starting_equity=100_000.0, positions=positions
    )
    decision = run_risk_agent(
        bullish_signal, "AAPL", "Tech", 100.0, 2.0, portfolio, config
    )
    assert decision.approved is False
    note = next(c for c in decision.checks if c.rule == "concurrent_positions")
    assert note.passed is False


def test_per_ticker_cap_resizes_instead_of_rejecting_when_room_remains(
    config, bullish_signal
):
    """
    Existing AAPL at $19,000 (19% of $100k) leaves $1,000 of room under the
    20% cap. The raw risk-based size (333 sh) exceeds that room, so the
    trade should be CLAMPED to the room available (10 sh) and approved --
    not rejected outright, since there IS still room, just not much.
    """
    positions = {
        "AAPL": Position(ticker="AAPL", shares=190, sector="Tech", market_value=19_000)
    }
    portfolio = PortfolioSnapshot(
        equity=100_000.0, starting_equity=100_000.0, positions=positions
    )
    decision = run_risk_agent(
        bullish_signal, "AAPL", "Tech", 100.0, 2.0, portfolio, config
    )

    assert decision.raw_shares == pytest.approx(1000 / 3.0, rel=1e-3)  # pre-cap size
    assert decision.approved is True
    assert decision.proposed_shares == pytest.approx(
        10.0, rel=1e-3
    )  # clamped to remaining room
    ticker_note = next(c for c in decision.checks if c.rule == "per_ticker_cap")
    assert ticker_note.passed is True
    sizing_note = next(c for c in decision.checks if c.rule == "position_sizing")
    assert "resized" in sizing_note.note.lower()


def test_per_ticker_cap_rejects_when_already_at_cap(config, bullish_signal):
    """
    Existing AAPL already sits exactly at the 20% cap ($20,000 of $100k) --
    zero room left, so the trade should be rejected outright with
    proposed_shares == 0.0, not resized to a fractional amount.
    """
    positions = {
        "AAPL": Position(ticker="AAPL", shares=200, sector="Tech", market_value=20_000)
    }
    portfolio = PortfolioSnapshot(
        equity=100_000.0, starting_equity=100_000.0, positions=positions
    )
    decision = run_risk_agent(
        bullish_signal, "AAPL", "Tech", 100.0, 2.0, portfolio, config
    )

    assert decision.approved is False
    assert decision.proposed_shares == 0.0
    assert decision.raw_shares > 0.0  # raw size still recorded for debugging
    rejection_note = next(c for c in decision.checks if c.rule == "sizing_rejected")
    assert "no cap room" in rejection_note.note.lower()


def test_per_sector_cap_resizes_instead_of_rejecting_when_room_remains(
    config, bullish_signal
):
    """
    MSFT ($20,000) + GOOGL ($19,500) = $39,500 of the $40,000 (40%) sector
    cap, leaving $500 of room. The raw risk-based NVDA size exceeds that,
    so it should be clamped to 1 sh (the room available) and approved.
    """
    positions = {
        "MSFT": Position(ticker="MSFT", shares=100, sector="Tech", market_value=20_000),
        "GOOGL": Position(
            ticker="GOOGL", shares=100, sector="Tech", market_value=19_500
        ),
    }
    portfolio = PortfolioSnapshot(
        equity=100_000.0, starting_equity=100_000.0, positions=positions
    )
    decision = run_risk_agent(
        bullish_signal, "NVDA", "Tech", 500.0, 5.0, portfolio, config
    )

    assert decision.approved is True
    assert decision.proposed_shares == pytest.approx(
        1.0, rel=1e-3
    )  # clamped to remaining sector room
    sector_note = next(c for c in decision.checks if c.rule == "per_sector_cap")
    assert sector_note.passed is True


def test_per_sector_cap_rejects_when_already_at_cap(config, bullish_signal):
    """
    MSFT + GOOGL already sum to exactly $40,000 (the 40% sector cap) --
    zero sector room left, so a new NVDA trade in the same sector should
    be rejected outright, not resized.
    """
    positions = {
        "MSFT": Position(ticker="MSFT", shares=100, sector="Tech", market_value=20_000),
        "GOOGL": Position(
            ticker="GOOGL", shares=100, sector="Tech", market_value=20_000
        ),
    }
    portfolio = PortfolioSnapshot(
        equity=100_000.0, starting_equity=100_000.0, positions=positions
    )
    decision = run_risk_agent(
        bullish_signal, "NVDA", "Tech", 500.0, 5.0, portfolio, config
    )

    assert decision.approved is False
    assert decision.proposed_shares == 0.0
    rejection_note = next(c for c in decision.checks if c.rule == "sizing_rejected")
    assert "no cap room" in rejection_note.note.lower()


def test_correlation_check_blocks_highly_correlated_new_position(
    config, bullish_signal
):
    positions = {
        "MSFT": Position(ticker="MSFT", shares=50, sector="Tech", market_value=10_000)
    }
    matrix = pd.DataFrame(
        {"AAPL": [1.0, 0.85], "MSFT": [0.85, 1.0]}, index=["AAPL", "MSFT"]
    )
    portfolio = PortfolioSnapshot(
        equity=100_000.0,
        starting_equity=100_000.0,
        positions=positions,
        correlation_matrix=matrix,
    )
    decision = run_risk_agent(
        bullish_signal, "AAPL", "Tech", 100.0, 2.0, portfolio, config
    )
    note = next(c for c in decision.checks if c.rule == "correlation")
    assert note.passed is False
    assert decision.approved is False


def test_no_correlation_matrix_yet_does_not_block_trade(
    config, clean_portfolio, bullish_signal
):
    """Brand-new session, correlation_matrix is still None -- should fail
    open (not block every trade) rather than crash."""
    decision = run_risk_agent(
        bullish_signal, "AAPL", "Tech", 100.0, 10.0, clean_portfolio, config
    )
    assert decision.approved is True


def test_each_failing_rule_produces_its_own_specific_note(config, bullish_signal):
    portfolio = PortfolioSnapshot(
        equity=90_000.0, starting_equity=100_000.0, positions={}
    )
    decision = run_risk_agent(
        bullish_signal, "AAPL", "Tech", 100.0, 2.0, portfolio, config
    )
    breaker = next(c for c in decision.checks if c.rule == "circuit_breaker")
    assert "-10.00%" in breaker.note
