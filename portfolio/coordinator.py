"""
PortfolioRiskCoordinator -- runs after ALL per-ticker subgraphs complete.
Deterministic, no LLM. Gates trades at the book level:
- total exposure vs. equity
- sector concentration (max 40%)
- max concurrent positions (3, tunable)
Calls into correlation.py and circuit_breaker.py for their respective checks.

Why this exists (architecture doc, section 4.2): a per-ticker RiskAgent
only ever sees the current, already-settled portfolio -- it has no idea
another ticker in the same sector was ALSO approved this same tick. Five
tickers can each individually pass their local per-sector-cap check and
still collectively blow through 40% once you add them all up. This module
is what catches that.

Controlling policy, in one sentence: process locally-approved proposals in
descending order of signal confidence, admitting each against a running
simulated portfolio, so that when the shared risk budget runs out, the
trades that lose their slot are the LOWEST-conviction ones, not just
whichever ticker happened to be processed first.

This is a gate, not a report (the doc's Phase 7 common pitfall): every
proposal is checked against the *running* state as trades ahead of it get
admitted, not filtered against the original static portfolio after the
fact.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from config.risk_config import RiskConfig
from portfolio.circuit_breaker import CircuitBreaker
from portfolio.correlation import check_correlation
from portfolio.state import PortfolioSnapshot, Position


@dataclass
class TickerProposal:
    """One ticker's locally-approved trade, going into book-level review.
    Comes from that ticker's local RiskAgent (agents/risk_agent.py) --
    only locally-approved proposals should be passed in here."""
    ticker: str
    sector: str
    entry_price: float
    proposed_shares: float
    combined_confidence: float  # from SignalMerger -- used to prioritize


@dataclass
class CoordinatorDecision:
    ticker: str
    approved: bool
    shares: float
    notes: list[str] = field(default_factory=list)


def run_portfolio_coordinator(
    proposals: list[TickerProposal],
    portfolio: PortfolioSnapshot,
    config: RiskConfig,
) -> dict[str, CoordinatorDecision]:
    """
    Returns {ticker: CoordinatorDecision} for every proposal passed in.
    """
    decisions: dict[str, CoordinatorDecision] = {}

    # Circuit breaker is a book-wide halt: if it's tripped, nothing this
    # tick gets through, regardless of what any local RiskAgent already
    # approved.
    breaker = CircuitBreaker(threshold_pct=config.circuit_breaker.daily_drawdown_pct)
    if breaker.check(portfolio.starting_equity, portfolio.equity):
        pnl = CircuitBreaker.daily_pnl_pct(portfolio.starting_equity, portfolio.equity)
        for p in proposals:
            decisions[p.ticker] = CoordinatorDecision(
                ticker=p.ticker, approved=False, shares=0.0,
                notes=[f"Circuit breaker halted the book (today's P&L {pnl:.2%})."],
            )
        return decisions

    # Highest-conviction trades get first claim on the shared risk budget.
    # Ticker name is a deterministic tiebreak so equal-confidence proposals
    # don't resolve differently between runs.
    ordered = sorted(proposals, key=lambda p: (-p.combined_confidence, p.ticker))

    # Running simulated state, seeded from the real portfolio and mutated
    # as each proposal is admitted -- this is what makes the check
    # cumulative instead of a static post-hoc filter.
    running_positions: dict[str, Position] = dict(portfolio.positions)
    running_held_tickers: list[str] = list(portfolio.positions.keys())

    for proposal in ordered:
        notes: list[str] = []

        existing = running_positions.get(proposal.ticker)
        existing_value = existing.market_value if existing else 0.0
        trade_value = proposal.proposed_shares * proposal.entry_price
        new_ticker_value = existing_value + trade_value
        ticker_pct = new_ticker_value / portfolio.equity if portfolio.equity else 1.0

        sector_value = sum(
            pos.market_value for t, pos in running_positions.items()
            if pos.sector == proposal.sector and t != proposal.ticker
        )
        new_sector_value = sector_value + new_ticker_value
        sector_pct = new_sector_value / portfolio.equity if portfolio.equity else 1.0

        would_open_new_ticker = proposal.ticker not in running_positions
        new_open_count = len(running_positions) + (1 if would_open_new_ticker else 0)

        corr_result = check_correlation(
            proposal.ticker, running_held_tickers, portfolio.correlation_matrix,
            config.correlation.max_correlation,
        )

        checks_passed = True
        if ticker_pct > config.per_ticker_cap_pct:
            checks_passed = False
            notes.append(
                f"Book-level per-ticker cap: would be {ticker_pct:.1%} of equity, "
                f"cap {config.per_ticker_cap_pct:.0%}."
            )
        if sector_pct > config.per_sector_cap_pct:
            checks_passed = False
            notes.append(
                f"Book-level per-sector cap ({proposal.sector}): would be {sector_pct:.1%} "
                f"of equity (including other trades approved this tick), "
                f"cap {config.per_sector_cap_pct:.0%}."
            )
        if new_open_count > config.max_concurrent_positions:
            checks_passed = False
            notes.append(
                f"Book-level concurrent positions: would be {new_open_count} open, "
                f"max {config.max_concurrent_positions}."
            )
        if corr_result["flagged"]:
            checks_passed = False
            notes.append(
                f"Book-level correlation: {proposal.ticker} at "
                f"{corr_result['max_correlation']} vs {corr_result['against']} "
                f"exceeds {config.correlation.max_correlation}."
            )

        if checks_passed:
            running_positions[proposal.ticker] = Position(
                ticker=proposal.ticker, sector=proposal.sector,
                shares=(existing.shares if existing else 0.0) + proposal.proposed_shares,
                market_value=new_ticker_value,
            )
            if would_open_new_ticker:
                running_held_tickers.append(proposal.ticker)
            decisions[proposal.ticker] = CoordinatorDecision(
                ticker=proposal.ticker, approved=True, shares=proposal.proposed_shares,
                notes=["Approved at book level."],
            )
        else:
            notes.append(
                "Rejected at book level -- higher-conviction trades approved "
                "this tick already used the available risk budget."
            )
            decisions[proposal.ticker] = CoordinatorDecision(
                ticker=proposal.ticker, approved=False, shares=0.0, notes=notes,
            )

    return decisions
