"""
Overview tab. Live account summary + positions table.
"""

import streamlit as st

from dashboard.data import load_account_summary, load_positions


def render():
    summary = load_account_summary()

    if summary["is_sample"]:
        st.info("Showing sample data — no portfolio/ledger.json found yet.")

    col1, col2, col3 = st.columns(3)
    col1.metric(
        "Equity", f"${summary['equity']:,.2f}", f"{summary['daily_pnl_pct']:+.2f}%"
    )
    col2.metric("Open Positions", summary["open_positions"])
    col3.metric("Circuit Breaker", "OK" if summary["circuit_breaker_ok"] else "HALTED")

    st.subheader("Positions")
    positions_df = load_positions()
    st.dataframe(positions_df, use_container_width=True)
