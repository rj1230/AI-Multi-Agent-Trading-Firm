import asyncio
from orchestrator.tick_runner import run_tick

result = asyncio.run(run_tick(["AAPL"]))
for ticker, tick_result in result.items():
    print(f"--- {ticker} ---")
    print(tick_result)
