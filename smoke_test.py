"""
Phase 1 smoke test — AI Multi-Agent Trading Firm

Confirms both external services are reachable and authenticated before any
agent logic is written:
  1. Alpaca paper-trading account endpoint (via alpaca-py)
  2. NewsAPI top-headlines endpoint

Usage:
    pip install alpaca-py requests python-dotenv
    python smoke_test.py

Expects a .env file (or exported env vars) with:
    ALPACA_API_KEY=...
    ALPACA_SECRET_KEY=...
    NEWSAPI_KEY=...
"""

import os
import sys

from dotenv import load_dotenv

load_dotenv()


def check_env_vars() -> dict:
    """Fail loudly and immediately if any key is missing, rather than
    letting a downstream SDK raise an opaque auth error."""
    required = ["ALPACA_API_KEY", "ALPACA_SECRET_KEY", "NEWSAPI_KEY"]
    values = {name: os.getenv(name) for name in required}
    missing = [name for name, val in values.items() if not val]

    if missing:
        print(f"[FAIL] Missing environment variables: {', '.join(missing)}")
        print("       Check your .env file is present and loaded.")
        sys.exit(1)

    print("[OK]   All required environment variables are set.")
    return values


def check_alpaca(api_key: str, secret_key: str) -> None:
    try:
        from alpaca.trading.client import TradingClient
    except ImportError:
        print("[FAIL] alpaca-py not installed. Run: pip install alpaca-py")
        sys.exit(1)

    try:
        # paper=True forces the paper-trading endpoint — never point this
        # at live trading during development.
        client = TradingClient(api_key, secret_key, paper=True)
        account = client.get_account()
    except Exception as e:
        print(f"[FAIL] Alpaca auth/connection failed: {e}")
        sys.exit(1)

    print("[OK]   Alpaca paper account reachable.")
    print(f"       Account status : {account.status}")
    print(f"       Equity         : ${account.equity}")
    print(f"       Buying power   : ${account.buying_power}")


def check_newsapi(api_key: str, query: str = "AAPL") -> None:
    import requests

    url = "https://newsapi.org/v2/top-headlines"
    params = {"q": query, "language": "en", "pageSize": 1, "apiKey": api_key}

    try:
        resp = requests.get(url, params=params, timeout=10)
        resp.raise_for_status()
    except Exception as e:
        print(f"[FAIL] NewsAPI request failed: {e}")
        sys.exit(1)

    data = resp.json()
    articles = data.get("articles", [])

    print("[OK]   NewsAPI reachable.")
    if articles:
        print(f"       Sample headline: {articles[0]['title']}")
    else:
        # This is the exact "zero articles" case NewsAgent must handle later
        # (see Phase 4 self-check) — good to see it here first.
        print("       No articles returned for this query (this is a valid")
        print("       case your NewsAgent will need to handle explicitly).")


def main():
    print("=== AI Multi-Agent Trading Firm — Phase 1 Smoke Test ===\n")

    env = check_env_vars()
    print()
    check_alpaca(env["ALPACA_API_KEY"], env["ALPACA_SECRET_KEY"])
    print()
    check_newsapi(env["NEWSAPI_KEY"])

    print("\n=== All checks passed. Safe to proceed to Phase 2. ===")


if __name__ == "__main__":
    main()
