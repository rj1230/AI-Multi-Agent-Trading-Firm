"""
Thin wrapper around the Alpaca paper-trading API.
"""


class AlpacaClient:
    def __init__(self, api_key: str, api_secret: str, paper: bool = True):
        raise NotImplementedError

    def get_account(self):
        raise NotImplementedError

    def get_positions(self):
        raise NotImplementedError

    def place_order(self, ticker: str, qty: float, side: str, order_type: str = "market"):
        raise NotImplementedError
