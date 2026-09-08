"""
DB connection/ORM layer for persisting trades, agent_logs, traces, and
portfolio snapshots. SQLite to start; interface stays swappable to Postgres.
"""
import sqlite3


def get_connection(db_path: str = "storage/trading_firm.db"):
    raise NotImplementedError


def init_db(conn):
    """Runs storage/schema.sql against the connection."""
    raise NotImplementedError


def insert_trade(conn, trade: dict):
    raise NotImplementedError


def insert_agent_log(conn, log_entry: dict):
    raise NotImplementedError


def insert_portfolio_snapshot(conn, snapshot: dict):
    raise NotImplementedError
