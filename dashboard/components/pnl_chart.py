"""
Live portfolio performance chart (equity curve) + a backtest-results tab
showing Sharpe / win rate / max drawdown from backtest/metrics.py.
"""


def render_pnl_chart(equity_curve: list[dict]):
    raise NotImplementedError


def render_backtest_results(metrics: dict):
    raise NotImplementedError
