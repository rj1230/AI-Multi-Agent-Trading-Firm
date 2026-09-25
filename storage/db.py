"""
SQLite connection and schema initialization for the trading firm.

The storage layer is persistence-only. It must not make trading decisions
or modify portfolio state.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Mapping
from pathlib import Path
from typing import Any

DEFAULT_DB_PATH = "storage/trading_firm.db"


def get_connection(db_path: str | Path = DEFAULT_DB_PATH) -> sqlite3.Connection:
    """Create a SQLite connection with foreign-key enforcement enabled."""
    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")

    return conn


def init_db(conn: sqlite3.Connection) -> None:
    """Create all storage tables and indexes if they do not exist."""
    schema_path = Path(__file__).with_name("schema.sql")
    schema = schema_path.read_text(encoding="utf-8")

    conn.executescript(schema)
    conn.commit()


def insert_trade(conn: sqlite3.Connection, trade: Mapping[str, Any]) -> int:
    """Persist an executed trade and return its database ID."""
    cursor = conn.execute(
        """
        INSERT INTO trades (
            trace_id,
            ticker,
            side,
            qty,
            price,
            order_id,
            status,
            mode,
            created_at
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            trade.get("trace_id"),
            trade["ticker"],
            trade["side"],
            trade["qty"],
            trade.get("price"),
            trade.get("order_id"),
            trade.get("status"),
            trade["mode"],
            trade["created_at"],
        ),
    )
    conn.commit()

    return int(cursor.lastrowid)


def insert_agent_log(
    conn: sqlite3.Connection,
    log_entry: Mapping[str, Any],
) -> int:
    """Persist a graph-node execution log and return its database ID."""
    cursor = conn.execute(
        """
        INSERT INTO agent_logs (
            trace_id,
            node_name,
            ticker,
            input_summary,
            output_summary,
            latency_ms,
            token_usage,
            created_at
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            log_entry["trace_id"],
            log_entry["node_name"],
            log_entry.get("ticker"),
            log_entry.get("input_summary"),
            log_entry.get("output_summary"),
            log_entry.get("latency_ms"),
            log_entry.get("token_usage"),
            log_entry["created_at"],
        ),
    )
    conn.commit()

    return int(cursor.lastrowid)


def insert_portfolio_snapshot(
    conn: sqlite3.Connection,
    snapshot: Mapping[str, Any],
) -> int:
    """Persist a portfolio snapshot and return its database ID."""
    cursor = conn.execute(
        """
        INSERT INTO portfolio_snapshots (
            equity,
            total_exposure_pct,
            sector_concentration,
            circuit_breaker_tripped,
            created_at
        )
        VALUES (?, ?, ?, ?, ?)
        """,
        (
            snapshot["equity"],
            snapshot.get("total_exposure_pct"),
            snapshot.get("sector_concentration"),
            snapshot.get("circuit_breaker_tripped"),
            snapshot["created_at"],
        ),
    )
    conn.commit()

    return int(cursor.lastrowid)
