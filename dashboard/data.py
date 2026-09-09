"""
Data access layer for the dashboard.

Isolated from app.py and components.py on purpose: this is the only file
that should know about file paths, JSON/CSV schemas, or where real data
will eventually come from (a DB, an API, wherever). Swap the internals here
and nothing else in the dashboard needs to change.
"""

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent.parent
AGENT_LOGS_DIR = PROJECT_ROOT / "agent_logs"
POSITIONS_FILE = PROJECT_ROOT / "data" / "positions_snapshot.json"
BACKTEST_RESULTS_FILE = PROJECT_ROOT / "data" / "backtest_results.csv"
BACKTEST_TRADES_FILE = PROJECT_ROOT / "data" / "backtest_trades.csv"

REFRESH_INTERVAL_SECONDS = 10


def has_real_data() -> bool:
    return AGENT_LOGS_DIR.exists() or POSITIONS_FILE.exists()


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
            {
                "timestamp": (now - timedelta(minutes=1)).isoformat(),
                "agent": "NewsAgent",
                "ticker": "AAPL",
                "message": "Sentiment: bullish (earnings beat). [sample data]",
            },
            {
                "timestamp": (now - timedelta(minutes=2)).isoformat(),
                "agent": "ChartAgent",
                "ticker": "AAPL",
                "message": "RSI 62, trend up, breakout above 20MA. [sample data]",
            },
            {
                "timestamp": (now - timedelta(minutes=2)).isoformat(),
                "agent": "SignalMerger",
                "ticker": "AAPL",
                "message": "News + chart agree: BUY signal. [sample data]",
            },
            {
                "timestamp": (now - timedelta(minutes=3)).isoformat(),
                "agent": "RiskAgent",
                "ticker": "AAPL",
                "message": "Position sized at 2% of portfolio, within sector cap. [sample data]",
            },
            {
                "timestamp": (now - timedelta(minutes=3)).isoformat(),
                "agent": "ExecutionAgent",
                "ticker": "AAPL",
                "message": "Order submitted via SimBroker. [sample data]",
            },
            {
                "timestamp": (now - timedelta(minutes=8)).isoformat(),
                "agent": "RiskAgent",
                "ticker": "MSFT",
                "message": "Signal rejected: exceeds sector concentration cap. [sample data]",
            },
        ]

    df = pd.DataFrame(rows)
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)
        df = df.sort_values("timestamp", ascending=False)
    return df.head(max_rows)


@st.cache_data(ttl=REFRESH_INTERVAL_SECONDS)
def load_positions() -> pd.DataFrame:
    if POSITIONS_FILE.exists():
        try:
            with open(POSITIONS_FILE, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            return pd.DataFrame(data)
        except Exception:
            pass

    return pd.DataFrame(
        [
            {
                "ticker": "AAPL",
                "qty": 12,
                "entry_price": 227.50,
                "current_price": 231.10,
                "unrealized_pnl": 43.20,
                "sector": "Tech",
            },
            {
                "ticker": "MSFT",
                "qty": 8,
                "entry_price": 415.20,
                "current_price": 411.00,
                "unrealized_pnl": -33.60,
                "sector": "Tech",
            },
            {
                "ticker": "JPM",
                "qty": 15,
                "entry_price": 198.10,
                "current_price": 202.40,
                "unrealized_pnl": 64.50,
                "sector": "Financials",
            },
        ]
    )


@st.cache_data(ttl=REFRESH_INTERVAL_SECONDS)
def load_backtest_results():
    equity_df, trades_df = None, None

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

    if equity_df is None:
        dates = pd.date_range(end=datetime.now(timezone.utc), periods=90, freq="D")
        equity = 100000 + pd.Series(range(90)).cumsum() * 15
        equity_df = pd.DataFrame({"date": dates, "equity": equity})

    if trades_df is None:
        now = datetime.now(timezone.utc)
        trades_df = pd.DataFrame(
            [
                {
                    "date": now - timedelta(days=5),
                    "ticker": "AAPL",
                    "side": "BUY",
                    "qty": 10,
                    "price": 225.00,
                    "pnl": None,
                },
                {
                    "date": now - timedelta(days=2),
                    "ticker": "AAPL",
                    "side": "SELL",
                    "qty": 10,
                    "price": 231.10,
                    "pnl": 61.00,
                },
            ]
        )

    return equity_df, trades_df
