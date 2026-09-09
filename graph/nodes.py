"""
Real node implementations, replacing the Phase 3 stubs. news_agent_node
and chart_agent_node were already wired in Phase 4 and are untouched here.

Two design notes worth being able to defend:

1. ATR/entry price are computed HERE (via a dedicated fetch_ohlcv call),
   not inside ChartAgent. ChartAgent's Signal is about DIRECTION
   (bullish/bearish/neutral) and graph.state.Signal only carries
   {direction, confidence, rationale} -- no numeric sizing fields. Adding
   ATR to ChartAgent's output would conflate "what does the chart say"
   with "how big should the trade be", two different concerns the
   architecture doc keeps separate (ChartAgent vs RiskAgent). The cost is
   a second fetch_ohlcv call per tick for the same ticker; that's a real,
   known inefficiency (worth caching per-tick later), not an oversight.

2. Risk config comes from config/risk_config.py + mandates/*.yaml, NOT
   config/settings.py's RISK_* constants. Those two are currently
   DUPLICATED in the repo -- see config/settings.py's docstring note. This
   file uses the YAML-driven, Pydantic-validated one since that's what
   RiskAgent/PortfolioRiskCoordinator were built and tested against.
"""

from __future__ import annotations

from agents.chart_agent import compute_atr, run_chart_agent
from agents.execution_agent import run_execution_agent
from agents.news_agent import run_news_agent
from agents.risk_agent import RiskDecision as RiskAgentDecision
from agents.risk_agent import run_risk_agent
from agents.signal_merger import MergedSignal, Signal as MergerSignal, merge_signals
from broker.alpaca_broker import AlpacaBroker
from broker.protocol import Broker
from broker.sim_broker import SimBroker
from config.risk_config import load_risk_config
from config.sectors import get_sector
from config.settings import MODE, PAPER_STARTING_EQUITY
from data_sources import fetch_ohlcv
from graph.state import AgentLogEntry, RiskDecision, Signal, TradingState
from portfolio.ledger import PortfolioLedger

ATR_LOOKBACK_DAYS = 30

# --- module-level singletons, built lazily on first use ------------------
# LangGraph nodes only receive `state`, so config/ledger/broker are held
# here rather than threaded through every node's arguments. All three are
# process-lifetime singletons: one risk mandate, one ledger file, one
# broker connection per running process.

_risk_config_singleton = None
_ledger_singleton = None
_broker_singleton = None


def _risk_config():
    global _risk_config_singleton
    if _risk_config_singleton is None:
        _risk_config_singleton = load_risk_config()
    return _risk_config_singleton


def _ledger() -> PortfolioLedger:
    global _ledger_singleton
    if _ledger_singleton is None:
        _ledger_singleton = PortfolioLedger(starting_equity=PAPER_STARTING_EQUITY)
    return _ledger_singleton


def _price_lookup(ticker: str) -> float:
    series = fetch_ohlcv(ticker, lookback_days=1)
    if series.is_empty or series.latest is None:
        raise ValueError(f"No price data available for {ticker}")
    return series.latest.close


def _broker() -> Broker:
    """SimBroker (backtest) is seeded from the ledger's current cash at
    first use, then updates in lockstep with the ledger via the same
    execution_agent_node call path below -- both get updated from the same
    order fill, so they can only drift from a bug in this file, not from
    independently-derived numbers. AlpacaBroker (live) is Alpaca's own
    account; the ledger still mirrors fills locally per the "positions in
    a local file" choice."""
    global _broker_singleton
    if _broker_singleton is None:
        if MODE == "live":
            _broker_singleton = AlpacaBroker()
        else:
            starting_cash = _ledger().snapshot().equity
            _broker_singleton = SimBroker(
                starting_cash=starting_cash, price_lookup=_price_lookup
            )
    return _broker_singleton


def _log(node: str, message: str) -> list[AgentLogEntry]:
    return [AgentLogEntry(node=node, message=message)]


def _to_merger_signal(signal: Signal | None, label: str) -> MergerSignal:
    if signal is None or signal.direction is None or signal.confidence is None:
        # Should not happen once News/ChartAgent have run (graph edges
        # guarantee they run first) -- fail safe to neutral rather than
        # crash the graph on a genuinely unexpected missing signal.
        return MergerSignal(
            direction="neutral", confidence=0.0, rationale=f"{label} signal missing"
        )
    return MergerSignal(
        direction=signal.direction,
        confidence=signal.confidence,
        rationale=signal.rationale or "",
    )


def _merged_signal_for_downstream(state: TradingState) -> MergedSignal:
    """Reconstructs an agents.signal_merger.MergedSignal from
    state.merged_signal (graph.state.Signal). `agreement` isn't stored in
    graph state -- it's folded into the rationale text by signal_merger_node
    below -- and isn't read by run_risk_agent/run_execution_agent, so its
    value here is a structural placeholder, not a real signal."""
    m = state.merged_signal
    if m is None or m.direction is None:
        return MergedSignal(
            direction="neutral",
            combined_confidence=0.0,
            agreement=False,
            rationale="missing",
        )
    return MergedSignal(
        direction=m.direction,
        combined_confidence=m.confidence or 0.0,
        agreement=True,
        rationale=m.rationale or "",
    )


# --- real nodes ------------------------------------------------------------


def news_agent_node(state: TradingState) -> dict:
    signal = run_news_agent(state.ticker)
    return {
        "news_signal": signal,
        "agent_logs": _log(
            "NewsAgent",
            f"{signal.direction} (conf={signal.confidence:.2f}): {signal.rationale}",
        ),
    }


def chart_agent_node(state: TradingState) -> dict:
    signal = run_chart_agent(state.ticker)
    return {
        "chart_signal": signal,
        "agent_logs": _log(
            "ChartAgent",
            f"{signal.direction} (conf={signal.confidence:.2f}): {signal.rationale}",
        ),
    }


def signal_merger_node(state: TradingState) -> dict:
    news = _to_merger_signal(state.news_signal, "News")
    chart = _to_merger_signal(state.chart_signal, "Chart")
    merged = merge_signals(news, chart)

    return {
        "merged_signal": Signal(
            direction=merged.direction,
            confidence=merged.combined_confidence,
            rationale=merged.rationale,
        ),
        "agent_logs": _log(
            "SignalMerger",
            f"{merged.direction} (conf={merged.combined_confidence:.2f}, "
            f"agreement={merged.agreement}): {merged.rationale}",
        ),
    }


def risk_agent_node(state: TradingState) -> dict:
    config = _risk_config()
    portfolio = _ledger().snapshot()
    sector = get_sector(state.ticker)

    series = fetch_ohlcv(state.ticker, lookback_days=ATR_LOOKBACK_DAYS)
    atr = compute_atr(series.bars) if not series.is_empty else None
    entry_price = (
        series.latest.close if (not series.is_empty and series.latest) else None
    )

    if atr is None or entry_price is None:
        return {
            "risk_decision": RiskDecision.REJECTED,
            "risk_notes": [
                "Cannot size trade: insufficient OHLCV history for ATR/entry price."
            ],
            "sector": sector,
            "agent_logs": _log(
                "RiskAgent", "rejected: insufficient price history for sizing"
            ),
        }

    merged = _merged_signal_for_downstream(state)
    decision = run_risk_agent(
        merged, state.ticker, sector, entry_price, atr, portfolio, config
    )

    return {
        "risk_decision": RiskDecision.APPROVED
        if decision.approved
        else RiskDecision.REJECTED,
        "risk_notes": decision.risk_notes,
        "atr": atr,
        "entry_price": entry_price,
        "sector": sector,
        "proposed_shares": decision.proposed_shares,
        "agent_logs": _log(
            "RiskAgent",
            f"{'approved' if decision.approved else 'rejected'}: {'; '.join(decision.risk_notes)}",
        ),
    }


def execution_agent_node(state: TradingState) -> dict:
    broker = _broker()
    merged = _merged_signal_for_downstream(state)
    risk_decision = RiskAgentDecision(
        approved=True,  # only reachable via route_after_risk when APPROVED
        ticker=state.ticker,
        proposed_shares=state.proposed_shares,
        checks=[],  # detail already logged by risk_agent_node; not needed here
    )

    result = run_execution_agent(risk_decision, merged, broker)

    if result.executed and result.order is not None:
        _ledger().record_fill(
            ticker=state.ticker,
            side=result.order.side,
            qty=result.order.qty,
            price=result.order.filled_avg_price or state.entry_price or 0.0,
            sector=state.sector or get_sector(state.ticker),
        )

    return {
        "execution_notes": result.notes,
        "agent_logs": _log("ExecutionAgent", result.notes),
    }


def hold_node(state: TradingState) -> dict:
    return {
        "execution_notes": "; ".join(state.risk_notes) or "held",
        "agent_logs": _log(
            "HoldNode", f"holding {state.ticker}, reason={state.risk_notes}"
        ),
    }


def route_after_risk(state: TradingState) -> str:
    if state.risk_decision == RiskDecision.APPROVED:
        return "execution_agent"
    return "hold_node"


# --- public accessors, for callers outside this module (e.g. the
# multi-ticker orchestrator) that need the same singletons every node
# here already uses -- keeps local (per-ticker) and book-level checks
# looking at identical state instead of two independently-built copies.


def get_risk_config():
    return _risk_config()


def get_ledger() -> PortfolioLedger:
    return _ledger()


def get_broker() -> Broker:
    return _broker()
