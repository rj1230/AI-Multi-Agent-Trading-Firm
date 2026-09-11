"""
Reasoning Feed tab. Groups tick_log entries into one card per ticker per
day, color-coded by outcome. Built from real backtest tick_log data,
requiring zero new persistence hooks.
"""

import streamlit as st

from dashboard.data import load_backtest_results

OUTCOME_STYLE = {
    "executed": ("✅", "green"),
    "held_local_reject": ("⏸️", "gray"),
    "held_book_reject": ("🟠", "orange"),
    "held_execution_failed": ("🔴", "red"),
}


def render():
    _, _, tick_log_df, is_sample = load_backtest_results()

    if is_sample:
        st.info("Showing sample data.")

    ticker_filter = st.selectbox(
        "Ticker", ["All"] + sorted(tick_log_df["ticker"].unique().tolist())
    )
    if ticker_filter != "All":
        tick_log_df = tick_log_df[tick_log_df["ticker"] == ticker_filter]

    for _, row in tick_log_df.sort_values("date", ascending=False).iterrows():
        icon, color = OUTCOME_STYLE.get(row["outcome"], ("❔", "gray"))
        with st.container(border=True):
            st.markdown(
                f"**{row['ticker']}** · {row['date']} · :{color}[{icon} {row['outcome'].upper()}]"
            )
            st.caption(row["notes"])
