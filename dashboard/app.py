"""
Streamlit entry point. Tabs: Live agent reasoning feed, Trade log + P&L,
Backtest results. Run with: streamlit run dashboard/app.py
"""
import streamlit as st


def main():
    st.set_page_config(page_title="AI Multi-Agent Trading Firm", layout="wide")
    st.title("AI Multi-Agent Trading Firm")
    # TODO: tabs -> components.agent_feed, components.trade_log, components.pnl_chart
    raise NotImplementedError


if __name__ == "__main__":
    main()
