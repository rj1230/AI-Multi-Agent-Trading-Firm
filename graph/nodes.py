"""
Real node implementations for the trading graph.

Graph flow:

    NewsAgent ───────┐
                     │
                     ▼
                SignalMerger
                     │
    ChartAgent ──────┘
                     │
                     ▼
                 RiskAgent
                  /      \
                 /        \
           APPROVED      REJECTED
              |             |
              v             v
       ExecutionAgent    HoldNode

Design principles:

1. Nodes communicate through TradingState.
2. SignalMerger is deterministic and contains no LLM call.
3. ChartAgent produces directional information only.
4. RiskAgent owns ATR, entry-price and position sizing logic.
5. ExecutionAgent only executes trades that passed RiskAgent.
6. HoldNode records blocked trades.
7. Broker implementation is hidden behind the Broker protocol.
8. PortfolioLedger is the local portfolio source of truth.
9. NewsAgent also propagates data provenance into TradingState.
"""

from __future__ import annotations

from typing import Optional

from agents.chart_agent import compute_atr, run_chart_agent
from agents.execution_agent import run_execution_agent
from agents.news_agent import (
    NewsAgentResult,
    run_news_agent,
    run_news_agent_with_metadata,
)
from agents.risk_agent import RiskDecision as RiskAgentDecision
from agents.risk_agent import run_risk_agent
from agents.signal_merger import (
    MergedSignal,
    Signal as MergerSignal,
    merge_signals,
)

from broker.alpaca_broker import AlpacaBroker
from broker.protocol import Broker
from broker.sim_broker import SimBroker

from config.risk_config import load_risk_config
from config.sectors import get_sector
from config.settings import MODE, PAPER_STARTING_EQUITY

from data_sources import fetch_ohlcv

from graph.state import (
    AgentLogEntry,
    RiskDecision,
    Signal,
    TradingState,
)

from portfolio.ledger import PortfolioLedger


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

ATR_LOOKBACK_DAYS = 30


# ---------------------------------------------------------------------------
# Lazy process-level singletons
# ---------------------------------------------------------------------------

_risk_config_singleton = None
_ledger_singleton: Optional[PortfolioLedger] = None
_broker_singleton: Optional[Broker] = None


def _risk_config():
    """
    Load the validated YAML/Pydantic risk configuration once per process.
    """

    global _risk_config_singleton

    if _risk_config_singleton is None:
        _risk_config_singleton = load_risk_config()

    return _risk_config_singleton


def _ledger() -> PortfolioLedger:
    """
    Return the process-level portfolio ledger.
    """

    global _ledger_singleton

    if _ledger_singleton is None:
        _ledger_singleton = PortfolioLedger(starting_equity=PAPER_STARTING_EQUITY)

    return _ledger_singleton


def _price_lookup(
    ticker: str,
) -> float:
    """
    Return the latest available close for a ticker.

    Historical backtesting should use the simulated-date-aware data facade.
    """

    series = fetch_ohlcv(
        ticker,
        lookback_days=1,
    )

    if series.is_empty or series.latest is None:
        raise ValueError(f"No price data available for {ticker}")

    return series.latest.close


def _broker() -> Broker:
    """
    Return the process-level broker.
    """

    global _broker_singleton

    if _broker_singleton is None:
        if MODE == "live":
            _broker_singleton = AlpacaBroker()

        else:
            starting_cash = _ledger().snapshot().equity

            _broker_singleton = SimBroker(
                starting_cash=starting_cash,
                price_lookup=_price_lookup,
            )

    return _broker_singleton


# ---------------------------------------------------------------------------
# Logging helper
# ---------------------------------------------------------------------------


def _log(
    node: str,
    message: str,
) -> list[AgentLogEntry]:
    """
    Create one structured graph log entry.
    """

    return [
        AgentLogEntry(
            node=node,
            message=message,
        )
    ]


# ---------------------------------------------------------------------------
# Signal conversion helpers
# ---------------------------------------------------------------------------


def _to_merger_signal(
    signal: Signal | None,
    label: str,
) -> MergerSignal:
    """
    Convert graph.state.Signal into signal_merger.Signal.
    """

    if signal is None:
        return MergerSignal(
            direction="neutral",
            confidence=0.0,
            rationale=f"{label} signal missing",
        )

    return MergerSignal(
        direction=signal.direction,
        confidence=signal.confidence,
        rationale=signal.rationale,
    )


def _merged_signal_for_downstream(
    state: TradingState,
) -> MergedSignal:
    """
    Reconstruct the richer SignalMerger MergedSignal from TradingState.
    """

    merged = state.merged_signal

    if merged is None:
        return MergedSignal(
            direction="neutral",
            combined_confidence=0.0,
            agreement=False,
            rationale="Missing merged signal.",
        )

    return MergedSignal(
        direction=merged.direction,
        combined_confidence=merged.confidence,
        agreement=state.merge_agreement is True,
        rationale=merged.rationale,
    )


# ---------------------------------------------------------------------------
# NewsAgent node
# ---------------------------------------------------------------------------


def news_agent_node(
    state: TradingState,
) -> dict:
    """
    Run NewsAgent and propagate signal + provenance metadata.

    Production execution uses the metadata-aware API.

    The legacy run_news_agent symbol remains imported at module scope
    intentionally because the existing test suite monkeypatches it.
    """

    result: NewsAgentResult = run_news_agent_with_metadata(state.ticker)

    signal = result.signal

    return {
        "news_signal": signal,
        # ------------------------------------------------------
        # News provenance.
        # ------------------------------------------------------
        "news_availability": result.availability.value,
        "news_article_count": result.article_count,
        "news_source": result.source,
        "news_as_of": result.as_of,
        # ------------------------------------------------------
        # Agent log.
        # ------------------------------------------------------
        "agent_logs": _log(
            "NewsAgent",
            (
                f"{signal.direction} "
                f"(conf={signal.confidence:.2f}); "
                f"availability={result.availability.value}; "
                f"articles={result.article_count}; "
                f"source={result.source}; "
                f"as_of={result.as_of.isoformat()}: "
                f"{signal.rationale}"
            ),
        ),
    }


# ---------------------------------------------------------------------------
# ChartAgent node
# ---------------------------------------------------------------------------


def chart_agent_node(
    state: TradingState,
) -> dict:
    """
    Run the deterministic ChartAgent.
    """

    signal = run_chart_agent(state.ticker)

    return {
        "chart_signal": signal,
        "agent_logs": _log(
            "ChartAgent",
            (f"{signal.direction} (conf={signal.confidence:.2f}): {signal.rationale}"),
        ),
    }


# ---------------------------------------------------------------------------
# SignalMerger node
# ---------------------------------------------------------------------------


def signal_merger_node(
    state: TradingState,
) -> dict:
    """
    Deterministically combine NewsAgent and ChartAgent signals.
    """

    news = _to_merger_signal(
        state.news_signal,
        "News",
    )

    chart = _to_merger_signal(
        state.chart_signal,
        "Chart",
    )

    merged = merge_signals(
        news,
        chart,
    )

    merged_signal = Signal(
        direction=merged.direction,
        confidence=merged.combined_confidence,
        rationale=merged.rationale,
    )

    return {
        "merged_signal": merged_signal,
        "merge_agreement": merged.agreement,
        "agent_logs": _log(
            "SignalMerger",
            (
                f"news={news.direction} "
                f"(conf={news.confidence:.2f}); "
                f"chart={chart.direction} "
                f"(conf={chart.confidence:.2f}); "
                f"merged={merged.direction} "
                f"(conf={merged.combined_confidence:.2f}, "
                f"agreement={merged.agreement}): "
                f"{merged.rationale}"
            ),
        ),
    }


# ---------------------------------------------------------------------------
# RiskAgent node
# ---------------------------------------------------------------------------


def risk_agent_node(
    state: TradingState,
) -> dict:
    """
    Run deterministic portfolio/risk checks.
    """

    config = _risk_config()

    portfolio = _ledger().snapshot()

    sector = get_sector(state.ticker)

    series = fetch_ohlcv(
        state.ticker,
        lookback_days=ATR_LOOKBACK_DAYS,
    )

    atr = compute_atr(series.bars) if not series.is_empty else None

    entry_price = (
        series.latest.close
        if (not series.is_empty and series.latest is not None)
        else None
    )

    if atr is None or entry_price is None:
        return {
            "risk_decision": RiskDecision.REJECTED,
            "risk_notes": [
                ("Cannot size trade: insufficient OHLCV history for ATR/entry price.")
            ],
            "atr": atr,
            "entry_price": entry_price,
            "sector": sector,
            "proposed_shares": 0.0,
            "agent_logs": _log(
                "RiskAgent",
                ("rejected: insufficient price history for sizing"),
            ),
        }

    merged = _merged_signal_for_downstream(state)

    decision = run_risk_agent(
        merged,
        state.ticker,
        sector,
        entry_price,
        atr,
        portfolio,
        config,
    )

    risk_status = "approved" if decision.approved else "rejected"

    risk_message = (
        f"merged={merged.direction} "
        f"conf={merged.combined_confidence:.3f} "
        f"agreement={merged.agreement}; "
        f"entry={entry_price:.2f}; "
        f"ATR={atr:.4f}; "
        f"{risk_status}: "
        f"{'; '.join(decision.risk_notes)}"
    )

    return {
        "risk_decision": (
            RiskDecision.APPROVED if decision.approved else RiskDecision.REJECTED
        ),
        "risk_notes": decision.risk_notes,
        "atr": atr,
        "entry_price": entry_price,
        "sector": sector,
        "proposed_shares": decision.proposed_shares,
        "agent_logs": _log(
            "RiskAgent",
            risk_message,
        ),
    }


# ---------------------------------------------------------------------------
# ExecutionAgent node
# ---------------------------------------------------------------------------


def execution_agent_node(
    state: TradingState,
) -> dict:
    """
    Execute a trade that passed RiskAgent.
    """

    broker = _broker()

    merged = _merged_signal_for_downstream(state)

    risk_decision = RiskAgentDecision(
        approved=True,
        ticker=state.ticker,
        proposed_shares=state.proposed_shares,
        checks=[],
    )

    result = run_execution_agent(
        risk_decision,
        merged,
        broker,
    )

    if result.executed and result.order is not None:
        fill_price = result.order.filled_avg_price or state.entry_price or 0.0

        fill_sector = state.sector or get_sector(state.ticker)

        _ledger().record_fill(
            ticker=state.ticker,
            side=result.order.side,
            qty=result.order.qty,
            price=fill_price,
            sector=fill_sector,
        )

    return {
        "execution_notes": result.notes,
        "execution_success": result.executed,
        "agent_logs": _log(
            "ExecutionAgent",
            result.notes,
        ),
    }


# ---------------------------------------------------------------------------
# HoldNode
# ---------------------------------------------------------------------------


def hold_node(
    state: TradingState,
) -> dict:
    """
    Record a trade that did not pass the risk gate.
    """

    reason = "; ".join(state.risk_notes) if state.risk_notes else "held"

    return {
        "execution_notes": reason,
        "execution_success": False,
        "agent_logs": _log(
            "HoldNode",
            (f"holding {state.ticker}, reason={state.risk_notes}"),
        ),
    }


# ---------------------------------------------------------------------------
# Conditional routing
# ---------------------------------------------------------------------------


def route_after_risk(
    state: TradingState,
) -> str:
    """
    Route approved trades to execution.

    Everything else goes to HoldNode.
    """

    if state.risk_decision == RiskDecision.APPROVED:
        return "execution_agent"

    return "hold_node"


# ---------------------------------------------------------------------------
# Public infrastructure accessors
# ---------------------------------------------------------------------------


def get_risk_config():
    """
    Return the shared validated risk configuration.
    """

    return _risk_config()


def get_ledger() -> PortfolioLedger:
    """
    Return the shared portfolio ledger.
    """

    return _ledger()


def get_broker() -> Broker:
    """
    Return the shared broker.
    """

    return _broker()


def reset_singletons() -> None:
    """
    Reset process-level infrastructure.

    Used by tests/backtests so separate runs do not leak portfolio,
    broker, or configuration state into each other.
    """

    global _risk_config_singleton
    global _ledger_singleton
    global _broker_singleton

    _risk_config_singleton = None
    _ledger_singleton = None
    _broker_singleton = None
