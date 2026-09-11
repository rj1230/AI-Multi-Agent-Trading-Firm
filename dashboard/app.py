"""
Main Streamlit entrypoint. Wires dashboard/data.py's adapter loaders into
dashboard/ui_components.py's presentation functions across three tabs.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st

from dashboard import ui_components as ui
from dashboard.data import (
    has_real_data,
    is_backtest_sample,
    load_account_summary,
    load_backtest_summary,
    load_backtest_trades,
    load_equity_curve,
    load_portfolio_snapshot,
    load_reasoning_cards,
    load_recent_ticks,
    load_sector_exposure,
)
from dashboard.styles import inject_base_css


def _sample_badge(is_sample: bool):
    if is_sample:
        st.caption("[sample] No real result recorded for this section yet.")


st.set_page_config(page_title="AI Multi-Agent Trading Firm", page_icon="\U0001f4c8", layout="wide")
inject_base_css()

st.title("AI Multi-Agent Trading Firm")
st.caption("Paper trading only — no real capital at risk.")

if not has_real_data():
    st.info(
        "No live portfolio or backtest data found yet. Showing sample data "
        "throughout — run `python run_one_tick.py` or export a backtest via "
        "`backtest/export_for_dashboard.py` to see real results."
    )

tab_overview, tab_reasoning, tab_backtest = st.tabs(["Overview", "Reasoning Feed", "Backtest Report"])

with tab_overview:
    summary = load_account_summary()
    _sample_badge(summary["is_sample"])
    snapshot = load_portfolio_snapshot()
    ui.metric_row(snapshot)
    st.subheader("Sector Exposure")
    ui.sector_exposure_bars(load_sector_exposure())
    st.subheader("Positions")
    ui.positions_table(snapshot.positions)
    st.subheader("Recent Activity")
    ui.recent_activity_list(load_recent_ticks(max_rows=10))

with tab_reasoning:
    st.caption("Every trade or hold decision, with the specific rule that approved or blocked it — not just the final verdict.")
    _sample_badge(is_backtest_sample())
    filtered_cards = ui.outcome_filter(load_reasoning_cards())
    if not filtered_cards:
        st.markdown("_No decisions match the current filter._")
    for card in filtered_cards:
        ui.reasoning_card(card)

with tab_backtest:
    _sample_badge(is_backtest_sample())
    summary = load_backtest_summary()
    ui.backtest_headline_metrics(summary)
    st.subheader("Equity Curve")
    equity_df = load_equity_curve()
    if equity_df is not None and not equity_df.empty:
        st.line_chart(equity_df.set_index("date"))
    else:
        st.markdown("_No backtest equity curve available yet._")
    st.subheader("Trade Log")
    trades_df = load_backtest_trades()
    if trades_df is not None and not trades_df.empty:
        st.dataframe(trades_df, width="stretch")
    else:
        st.markdown("_No trades recorded in this backtest window._")
    ui.limitations_caption(
        "News data limited to hand-seeded headlines for demo tickers; "
        "buy-and-hold comparison not yet computed separately from the "
        "strategy curve; win rate requires per-trade P&L not yet tracked — "
        "NewsAPI's free tier does not serve historical headlines old enough "
        "for a real backtest window."
    )
