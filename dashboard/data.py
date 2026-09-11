"""
Data access layer for the dashboard.
"""

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import streamlit as st

from portfolio.ledger import PortfolioLedger, DEFAULT_LEDGER_PATH

PROJECT_ROOT = Path(__file__).resolve().parent.parent
AGENT_LOGS_DIR = PROJECT_ROOT / "agent_logs"
BACKTEST_RESULTS_FILE = PROJECT_ROOT / "data" / "backtest_results.csv"
BACKTEST_TRADES_FILE = PROJECT_ROOT / "data" / "backtest_trades.csv"
BACKTEST_TICK_LOG_FILE = PROJECT_ROOT / "data" / "backtest_tick_log.csv"

REFRESH_INTERVAL_SECONDS = 10


def has_real_data() -> bool:
    return AGENT_LOGS_DIR.exists() or DEFAULT_LEDGER_PATH.exists() or BACKTEST_RESULTS_FILE.exists()


@st.cache_data(ttl=REFRESH_INTERVAL_SECONDS)
def load_agent_logs(max_rows: int = 200) -> pd.DataFrame:
    rows = []
    if AGENT_LOGS_DIR.exists():
        for f in sorted(AGENT_LOGS_DIR.glob("*.jsonl"), reverse=True):
            with open(f, "r", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rows.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
            if len(rows) >= max_rows:
                break

    if not rows:
        now = datetime.now(timezone.utc)
        rows = [
            {"timestamp": (now - timedelta(minutes=1)).isoformat(), "agent": "NewsAgent",
             "ticker": "AAPL", "message": "Sentiment: bullish (earnings beat). [sample data]"},
            {"timestamp": (now - timedelta(minutes=2)).isoformat(), "agent": "RiskAgent",
             "ticker": "MSFT", "message": "Signal rejected: exceeds sector cap. [sample data]"},
        ]

    df = pd.DataFrame(rows)
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)
        df = df.sort_values("timestamp", ascending=False)
    return df.head(max_rows)


@st.cache_data(ttl=REFRESH_INTERVAL_SECONDS)
def load_positions() -> pd.DataFrame:
    if DEFAULT_LEDGER_PATH.exists():
        try:
            ledger = PortfolioLedger(starting_equity=100_000.0, path=DEFAULT_LEDGER_PATH)
            snapshot = ledger.snapshot()
            if snapshot.positions:
                rows = [
                    {
                        "ticker": p.ticker,
                        "shares": round(p.shares, 4),
                        "sector": p.sector,
                        "market_value": round(p.market_value, 2),
                        "pct_of_equity": round(p.market_value / snapshot.equity * 100, 2) if snapshot.equity else 0,
                    }
                    for p in snapshot.positions.values()
                ]
                return pd.DataFrame(rows)
        except Exception:
            pass

    return pd.DataFrame([
        {"ticker": "AAPL", "shares": 12, "sector": "Tech", "market_value": 2773.20, "pct_of_equity": 2.77},
        {"ticker": "JPM", "shares": 15, "sector": "Financials", "market_value": 3036.00, "pct_of_equity": 3.04},
    ])


@st.cache_data(ttl=REFRESH_INTERVAL_SECONDS)
def load_account_summary() -> dict:
    if DEFAULT_LEDGER_PATH.exists():
        try:
            ledger = PortfolioLedger(starting_equity=100_000.0, path=DEFAULT_LEDGER_PATH)
            snapshot = ledger.snapshot()
            daily_pnl = snapshot.equity - snapshot.starting_equity
            daily_pnl_pct = (daily_pnl / snapshot.starting_equity * 100) if snapshot.starting_equity else 0
            return {
                "equity": snapshot.equity,
                "cash": ledger._data.get("cash", 0),
                "daily_pnl": daily_pnl,
                "daily_pnl_pct": daily_pnl_pct,
                "open_positions": len(snapshot.positions),
                "circuit_breaker_ok": daily_pnl_pct > -3.0,
                "is_sample": False,
            }
        except Exception:
            pass

    return {
        "equity": 100_588.94, "cash": 90_000.00, "daily_pnl": 588.94, "daily_pnl_pct": 0.59,
        "open_positions": 1, "circuit_breaker_ok": True, "is_sample": True,
    }


@st.cache_data(ttl=REFRESH_INTERVAL_SECONDS)
def load_backtest_results():
    equity_df, trades_df, tick_log_df = None, None, None

    if BACKTEST_RESULTS_FILE.exists():
        try:
            equity_df = pd.read_csv(BACKTEST_RESULTS_FILE, parse_dates=["date"])
        except Exception:
            equity_df = None
    if BACKTEST_TRADES_FILE.exists():
        try:
            trades_df = pd.read_csv(BACKTEST_TRADES_FILE, parse_dates=["date"])
        except Exception:
            trades_df = None
    if BACKTEST_TICK_LOG_FILE.exists():
        try:
            tick_log_df = pd.read_csv(BACKTEST_TICK_LOG_FILE, parse_dates=["date"])
        except Exception:
            tick_log_df = None

    is_sample = equity_df is None
    if equity_df is None:
        dates = pd.date_range(end=datetime.now(timezone.utc), periods=7, freq="D")
        equity_df = pd.DataFrame({"date": dates, "equity": [100000, 100000, 100000, 100000, 100390, 100497, 100589]})
    if trades_df is None:
        trades_df = pd.DataFrame([{"date": datetime.now(timezone.utc), "ticker": "AAPL", "notes": "[sample data]"}])
    if tick_log_df is None:
        tick_log_df = pd.DataFrame([{"date": datetime.now(timezone.utc), "ticker": "AAPL", "outcome": "executed", "notes": "[sample data]"}])

    return equity_df, trades_df, tick_log_df, is_sample


# ---------------------------------------------------------------------------
# Adapter layer: builds the dataclasses ui_components.py expects on top of
# the DataFrame/dict loaders above.
# ---------------------------------------------------------------------------

from dataclasses import dataclass, field


@dataclass
class Position:
    ticker: str
    shares: float
    sector: str
    market_value: float
    pct_of_portfolio: float
    pct_of_ticker_cap: float


@dataclass
class PortfolioSnapshot:
    equity: float
    daily_pnl: float
    daily_pnl_pct: float
    circuit_breaker_halted: bool
    positions: list = field(default_factory=list)


@dataclass
class ReasoningStep:
    icon: str
    label: str
    detail: str
    passed: object


@dataclass
class TickCard:
    ticker: str
    tick_date: str
    outcome: str
    headline_reason: str
    steps: list = field(default_factory=list)


@dataclass
class BacktestSummary:
    sharpe_ratio: float
    win_rate: float
    max_drawdown: float
    total_return: float
    buy_hold_return: float


PER_TICKER_CAP_PCT = 0.20


def _parse_notes_to_steps(notes):
    if not notes or not isinstance(notes, str):
        return []
    if ":" not in notes:
        return []
    steps = []
    for part in notes.split(";"):
        part = part.strip()
        if not part:
            continue
        passed = None
        if "BREACH" in part:
            passed = False
        elif "-> ok" in part or part.endswith("ok."):
            passed = True
        label = part.split(":")[0].strip() if ":" in part else part
        steps.append(ReasoningStep(icon="", label=label, detail=part, passed=passed))
    return steps


def load_portfolio_snapshot():
    summary = load_account_summary()
    positions_df = load_positions()

    positions = [
        Position(
            ticker=row["ticker"],
            shares=row["shares"],
            sector=row["sector"],
            market_value=row["market_value"],
            pct_of_portfolio=row["pct_of_equity"] / 100,
            pct_of_ticker_cap=min((row["pct_of_equity"] / 100) / PER_TICKER_CAP_PCT, 1.0),
        )
        for _, row in positions_df.iterrows()
    ]

    return PortfolioSnapshot(
        equity=summary["equity"],
        daily_pnl=summary["daily_pnl"],
        daily_pnl_pct=summary["daily_pnl_pct"] / 100,
        circuit_breaker_halted=not summary["circuit_breaker_ok"],
        positions=positions,
    )


def load_sector_exposure():
    positions_df = load_positions()
    if positions_df.empty:
        return {}
    grouped = positions_df.groupby("sector")["pct_of_equity"].sum() / 100
    return grouped.to_dict()


def load_recent_ticks(max_rows: int = 10):
    _, _, tick_log_df, _ = load_backtest_results()
    cards = []
    for _, row in tick_log_df.sort_values("date", ascending=False).head(max_rows).iterrows():
        notes = str(row.get("notes", ""))
        cards.append(TickCard(
            ticker=row["ticker"],
            tick_date=str(row["date"]),
            outcome=row.get("outcome", "held"),
            headline_reason=notes[:120],
            steps=_parse_notes_to_steps(notes),
        ))
    return cards


def load_reasoning_cards():
    return load_recent_ticks(max_rows=500)


def load_equity_curve():
    equity_df, _, _, _ = load_backtest_results()
    return equity_df


def load_backtest_trades():
    _, trades_df, _, _ = load_backtest_results()
    return trades_df


def load_backtest_summary():
    from backtest.metrics import summarize
    equity_df, trades_df, _, _ = load_backtest_results()
    equity_values = equity_df["equity"].tolist()
    metrics = summarize(equity_values, trades_df.to_dict("records"), equity_values)
    return BacktestSummary(
        sharpe_ratio=metrics["sharpe_ratio"],
        win_rate=metrics["win_rate"],
        max_drawdown=metrics["max_drawdown"],
        total_return=metrics["total_return"],
        buy_hold_return=metrics["buy_and_hold_return"],
    )


def is_backtest_sample() -> bool:
    return load_backtest_results()[3]
