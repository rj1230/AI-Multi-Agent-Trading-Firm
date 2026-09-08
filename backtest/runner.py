"""
Walks HistoricalDataSource's as_of clock forward step by step, invoking the
graph at each step and applying simulated fills — the same graph used live.
"""


def run_backtest(tickers: list[str], start_date: str, end_date: str, step: str = "1d") -> dict:
    """Returns a ledger of all trades + a per-step equity curve."""
    raise NotImplementedError
