"""
Backtest Report tab. Renders headline metrics, equity curve, and trade log
from load_backtest_results(). Call render() from app.py inside whichever
tab/container holds this view.
"""

import streamlit as st

from backtest.metrics import summarize
from dashboard.data import load_backtest_results


def render():
    equity_df, trades_df, tick_log_df, is_sample = load_backtest_results()

    if is_sample:
        st.info(
            "Showing sample data — run a real backtest and export_result() to see live results."
        )

    equity_values = equity_df["equity"].tolist()
    metrics = summarize(
        equity_values, trades_df.to_dict("records"), equity_values
    )  # buy-and-hold placeholder if not tracked separately

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Sharpe Ratio", f"{metrics['sharpe_ratio']:.2f}")
    col2.metric("Win Rate", f"{metrics['win_rate'] * 100:.1f}%")
    col3.metric("Max Drawdown", f"{metrics['max_drawdown'] * 100:.1f}%")
    col4.metric("Total Return", f"{metrics['total_return'] * 100:.2f}%")

    st.subheader("Equity Curve")
    st.line_chart(equity_df.set_index("date")["equity"])

    st.subheader("Trade Log")
    st.dataframe(trades_df, use_container_width=True)

    st.caption(
        "News data limited to hand-seeded headlines for demo tickers — "
        "NewsAPI's free tier does not serve historical headlines old enough for backtesting."
    )
