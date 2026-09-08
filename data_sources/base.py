"""
Abstract DataSource interface. Every agent and graph node depends only on
this — never on yFinance/NewsAPI/Alpaca directly — which is what makes the
LIVE/BACKTEST mode swap invisible to agent logic.
"""
from abc import ABC, abstractmethod


class DataSource(ABC):
    @abstractmethod
    def get_ohlcv(self, ticker: str, lookback: int = 100):
        """Return recent OHLCV bars for ticker."""
        raise NotImplementedError

    @abstractmethod
    def get_news(self, ticker: str, limit: int = 10):
        """Return recent news articles for ticker."""
        raise NotImplementedError

    @abstractmethod
    def get_account_state(self):
        """Return current equity, positions, and buying power."""
        raise NotImplementedError
