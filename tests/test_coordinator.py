import pandas as pd
import pytest

from config.risk_config import RiskConfig
from portfolio.coordinator import TickerProposal, run_portfolio_coordinator
from portfolio.state import PortfolioSnapshot, Position


@pytest.fixture
def config():
    return RiskConfig.model_validate({
        "position_sizing": {"equity_risk_pct": 0.01, "atr_stop_multiple": 1.5},
        "per_ticker_cap_pct": 0.20,
        "per_sector_cap_pct": 0.40,
        "max_concurrent_positions": 3,
        "circuit_breaker": {"daily_drawdown_pct": -0.03},
        "correlation": {"max_correlation": 0.7},
    })


@pytest.fixture
def clean_portfolio():
    return PortfolioSnapshot(equity=100_000.0, starting_equity=100_000.0, positions={})


def test_circuit_breaker_rejects_every_proposal(config):
    portfolio = PortfolioSnapshot(equity=95_000.0, starting_equity=100_000.0, positions={})
    proposals = [
        TickerProposal(ticker="AAPL", sector="Tech", entry_price=100, proposed_shares=10, combined_confidence=0.9),
    ]
    decisions = run_portfolio_coordinator(proposals, portfolio, config)
    assert decisions["AAPL"].approved is False
    assert "circuit breaker" in decisions["AAPL"].notes[0].lower()


def test_three_correlated_sector_proposals_highest_conviction_wins(config, clean_portfolio):
    """The architecture doc's Phase 7 self-check: three tickers in the same
    sector each get an independent buy signal at the same tick. Each is
    individually within the 20% per-ticker cap (19% each), but all three
    together would be 57% of equity -- well over the 40% sector cap.
    Highest-confidence proposals should win the shared budget."""
    proposals = [
        TickerProposal(ticker="GOOGL", sector="Tech", entry_price=100, proposed_shares=190, combined_confidence=0.9),
        TickerProposal(ticker="MSFT", sector="Tech", entry_price=100, proposed_shares=190, combined_confidence=0.8),
        TickerProposal(ticker="AAPL", sector="Tech", entry_price=100, proposed_shares=190, combined_confidence=0.7),
    ]
    decisions = run_portfolio_coordinator(proposals, clean_portfolio, config)

    assert decisions["GOOGL"].approved is True   # 19% -- fits
    assert decisions["MSFT"].approved is True    # 19% + 19% = 38% -- still fits
    assert decisions["AAPL"].approved is False   # 38% + 19% = 57% -- breaches 40% cap
    assert "sector cap" in decisions["AAPL"].notes[0].lower()
    assert "budget" in decisions["AAPL"].notes[-1].lower()


def test_per_ticker_cap_still_enforced_at_book_level(config, clean_portfolio):
    proposals = [
        TickerProposal(ticker="AAPL", sector="Tech", entry_price=100, proposed_shares=250, combined_confidence=0.9),
    ]
    decisions = run_portfolio_coordinator(proposals, clean_portfolio, config)
    assert decisions["AAPL"].approved is False
    assert "per-ticker cap" in decisions["AAPL"].notes[0].lower()


def test_concurrent_positions_cap_blocks_lower_conviction_new_ticker(config):
    existing = {
        "MSFT": Position(ticker="MSFT", shares=10, sector="Tech", market_value=1000),
        "GOOGL": Position(ticker="GOOGL", shares=10, sector="Tech", market_value=1000),
    }
    portfolio = PortfolioSnapshot(equity=100_000.0, starting_equity=100_000.0, positions=existing)
    proposals = [
        TickerProposal(ticker="AAPL", sector="Tech", entry_price=50, proposed_shares=10, combined_confidence=0.9),
        TickerProposal(ticker="NVDA", sector="Tech", entry_price=50, proposed_shares=10, combined_confidence=0.5),
    ]
    decisions = run_portfolio_coordinator(proposals, portfolio, config)
    # 2 already open + AAPL = 3 (at max, ok); + NVDA would be 4 (breach)
    assert decisions["AAPL"].approved is True
    assert decisions["NVDA"].approved is False
    assert "concurrent positions" in decisions["NVDA"].notes[0].lower()


def test_correlation_check_flags_two_new_correlated_tickers_same_tick(config, clean_portfolio):
    matrix = pd.DataFrame(
        {"AAPL": [1.0, 0.85], "MSFT": [0.85, 1.0]}, index=["AAPL", "MSFT"],
    )
    portfolio = PortfolioSnapshot(
        equity=100_000.0, starting_equity=100_000.0, positions={}, correlation_matrix=matrix,
    )
    proposals = [
        TickerProposal(ticker="AAPL", sector="Tech", entry_price=50, proposed_shares=10, combined_confidence=0.9),
        TickerProposal(ticker="MSFT", sector="Tech", entry_price=50, proposed_shares=10, combined_confidence=0.8),
    ]
    decisions = run_portfolio_coordinator(proposals, portfolio, config)
    # AAPL goes first (higher confidence) and gets in since nothing is held
    # yet; MSFT is then checked against a portfolio that now includes AAPL.
    assert decisions["AAPL"].approved is True
    assert decisions["MSFT"].approved is False
    assert "correlation" in decisions["MSFT"].notes[0].lower()


def test_equal_confidence_breaks_tie_by_ticker_name(config, clean_portfolio):
    proposals = [
        TickerProposal(ticker="ZETA", sector="Tech", entry_price=100, proposed_shares=190, combined_confidence=0.8),
        TickerProposal(ticker="ALPHA", sector="Tech", entry_price=100, proposed_shares=190, combined_confidence=0.8),
        TickerProposal(ticker="MID", sector="Tech", entry_price=100, proposed_shares=190, combined_confidence=0.8),
    ]
    decisions = run_portfolio_coordinator(proposals, clean_portfolio, config)
    # Alphabetical tiebreak: ALPHA, MID processed before ZETA
    assert decisions["ALPHA"].approved is True
    assert decisions["MID"].approved is True
    assert decisions["ZETA"].approved is False


def test_all_proposals_pass_when_comfortably_under_caps(config, clean_portfolio):
    proposals = [
        TickerProposal(ticker="AAPL", sector="Tech", entry_price=50, proposed_shares=10, combined_confidence=0.9),
        TickerProposal(ticker="XOM", sector="Energy", entry_price=50, proposed_shares=10, combined_confidence=0.8),
    ]
    decisions = run_portfolio_coordinator(proposals, clean_portfolio, config)
    assert all(d.approved for d in decisions.values())
