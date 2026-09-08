"""
LIVE-mode entry point. Polls on an interval, checks market hours, and
triggers a graph run (asyncio.gather across the watchlist, then
PortfolioRiskCoordinator) each cycle.
"""
import asyncio


async def run_loop(tickers: list[str], interval_seconds: int = 300):
    raise NotImplementedError


if __name__ == "__main__":
    asyncio.run(run_loop(tickers=[]))
