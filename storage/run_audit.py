"""
Persistent run-level audit storage.

RunAuditStore persists backtest/live run metadata and deterministic
TradeTrace records into SQLite.

This module is observational only. It must never participate in trading
decisions, risk sizing, execution, or portfolio accounting.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any

from storage.db import get_connection, init_db, insert_trade
from telemetry.trade_trace import TradeTrace


class RunAuditStore:
    """SQLite persistence layer for trading runs and TradeTrace records."""

    def __init__(self, db_path: str = "storage/trading_firm.db") -> None:
        self.db_path = db_path
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)

        with self._connection() as conn:
            init_db(conn)

    # ---------- Public API ----------

    def start_run(
        self,
        run_id: str,
        *,
        started_at: datetime | str,
        start_date: datetime | str | None = None,
        end_date: datetime | str | None = None,
        tickers: Iterable[str] = (),
        starting_equity: float | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Create a new audit run."""
        if not run_id.strip():
            raise ValueError("run_id must not be empty")

        tickers_json = json.dumps(list(tickers), separators=(",", ":"))

        with self._connection() as conn:
            conn.execute(
                """
                INSERT INTO runs (
                    run_id,
                    started_at,
                    status,
                    start_date,
                    end_date,
                    tickers_json,
                    starting_equity,
                    metadata_json
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    run_id,
                    _isoformat(started_at),
                    "running",
                    _isoformat(start_date),
                    _isoformat(end_date),
                    tickers_json,
                    starting_equity,
                    _json_dumps(metadata or {}),
                ),
            )

    def record_trade_trace(self, trace: TradeTrace) -> None:
        """Persist one complete TradeTrace, upserting by trace_key."""
        if not isinstance(trace, TradeTrace):
            raise TypeError("trace must be a TradeTrace instance")

        payload = trace.to_dict()

        execution = payload.get("execution") or {}
        risk = payload.get("risk") or {}
        signal = payload.get("signal") or {}
        coordinator = payload.get("coordinator") or {}
        accounting = payload.get("accounting") or {}

        metadata = payload.get("metadata") or {}
        tick_id = metadata.get("tick_id")
        outcome = metadata.get("outcome")

        trace_key = _trace_key(
            run_id=trace.run_id,
            ticker=trace.ticker,
            simulated_date=trace.simulated_date,
            tick_id=tick_id,
        )

        with self._connection() as conn:
            conn.execute(
                """
                INSERT INTO trade_traces (
                    run_id,
                    trace_key,
                    tick_id,
                    ticker,
                    simulated_date,
                    outcome,
                    signal_direction,
                    signal_confidence,
                    risk_approved,
                    risk_raw_shares,
                    risk_proposed_shares,
                    risk_final_shares,
                    coordinator_approved,
                    coordinator_shares,
                    execution_attempted,
                    execution_success,
                    execution_side,
                    execution_quantity,
                    execution_price,
                    execution_order_id,
                    realized_pnl,
                    remaining_shares,
                    position_closed,
                    trace_json,
                    created_at
                )
                VALUES (
                    ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                    ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                    ?, ?, ?
                )
                ON CONFLICT(trace_key) DO UPDATE SET
                    tick_id = excluded.tick_id,
                    ticker = excluded.ticker,
                    simulated_date = excluded.simulated_date,
                    outcome = excluded.outcome,
                    signal_direction = excluded.signal_direction,
                    signal_confidence = excluded.signal_confidence,
                    risk_approved = excluded.risk_approved,
                    risk_raw_shares = excluded.risk_raw_shares,
                    risk_proposed_shares = excluded.risk_proposed_shares,
                    risk_final_shares = excluded.risk_final_shares,
                    coordinator_approved = excluded.coordinator_approved,
                    coordinator_shares = excluded.coordinator_shares,
                    execution_attempted = excluded.execution_attempted,
                    execution_success = excluded.execution_success,
                    execution_side = excluded.execution_side,
                    execution_quantity = excluded.execution_quantity,
                    execution_price = excluded.execution_price,
                    execution_order_id = excluded.execution_order_id,
                    realized_pnl = excluded.realized_pnl,
                    remaining_shares = excluded.remaining_shares,
                    position_closed = excluded.position_closed,
                    trace_json = excluded.trace_json,
                    created_at = excluded.created_at
                """,
                (
                    trace.run_id,
                    trace_key,
                    tick_id,
                    trace.ticker,
                    _isoformat(trace.simulated_date),
                    outcome,
                    _normalize_signal_direction(signal.get("direction")),
                    signal.get("confidence"),
                    _bool_to_int(risk.get("approved")),
                    risk.get("raw_shares"),
                    risk.get("proposed_shares"),
                    risk.get("final_shares"),
                    _bool_to_int(coordinator.get("approved")),
                    coordinator.get("shares"),
                    _bool_to_int(execution.get("attempted")),
                    _bool_to_int(execution.get("success")),
                    _normalize_execution_side(execution.get("side")),
                    execution.get("quantity"),
                    execution.get("price"),
                    execution.get("order_id"),
                    accounting.get("realized_pnl"),
                    accounting.get("remaining_shares"),
                    _bool_to_int(accounting.get("position_closed")),
                    _json_dumps(payload),
                    _isoformat(trace.timestamp),
                ),
            )

            if execution.get("success"):
                existing_trade = conn.execute(
                    """
                    SELECT id
                    FROM trades
                    WHERE trace_id = ?
                    LIMIT 1
                    """,
                    (trace_key,),
                ).fetchone()

                if existing_trade is None:
                    insert_trade(
                        conn,
                        {
                            "trace_id": trace_key,
                            "ticker": trace.ticker,
                            "side": _normalize_execution_side(execution.get("side")),
                            "qty": execution.get("quantity"),
                            "price": execution.get("price"),
                            "order_id": execution.get("order_id"),
                            "status": "filled",
                            "mode": "backtest",
                            "created_at": _isoformat(trace.timestamp),
                        },
                    )

            conn.execute(
                """
                UPDATE runs
                SET trade_trace_count = (
                    SELECT COUNT(*)
                    FROM trade_traces
                    WHERE trade_traces.run_id = runs.run_id
                )
                WHERE run_id = ?
                """,
                (trace.run_id,),
            )

    def finish_run(
        self,
        run_id: str,
        *,
        finished_at: datetime | str,
        status: str = "completed",
        final_equity: float | None = None,
        realized_pnl: float | None = None,
        benchmark_return: float | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Mark a run complete and persist its final summary."""
        valid_statuses = {
            "pending",
            "running",
            "completed",
            "failed",
            "cancelled",
        }

        if status not in valid_statuses:
            raise ValueError(
                f"Unsupported run status: {status!r}. "
                f"Expected one of: {sorted(valid_statuses)}"
            )

        with self._connection() as conn:
            cursor = conn.execute(
                """
                UPDATE runs
                SET
                    finished_at = ?,
                    status = ?,
                    final_equity = ?,
                    realized_pnl = ?,
                    benchmark_return = ?,
                    metadata_json = COALESCE(?, metadata_json),
                    trade_trace_count = (
                        SELECT COUNT(*)
                        FROM trade_traces
                        WHERE trade_traces.run_id = runs.run_id
                    )
                WHERE run_id = ?
                """,
                (
                    _isoformat(finished_at),
                    status,
                    final_equity,
                    realized_pnl,
                    benchmark_return,
                    _json_dumps(metadata) if metadata is not None else None,
                    run_id,
                ),
            )

            if cursor.rowcount == 0:
                raise KeyError(f"Unknown run_id: {run_id}")

    def get_run(self, run_id: str) -> dict[str, Any] | None:
        """Return one run summary."""
        with self._connection() as conn:
            row = conn.execute(
                "SELECT * FROM runs WHERE run_id = ?",
                (run_id,),
            ).fetchone()

        if row is None:
            return None

        result = dict(row)
        result["tickers"] = json.loads(result.pop("tickers_json"))
        result["metadata"] = json.loads(result.pop("metadata_json") or "{}")

        return result

    def list_runs(self) -> list[dict[str, Any]]:
        """Return run summaries ordered newest first."""
        with self._connection() as conn:
            rows = conn.execute(
                "SELECT * FROM runs ORDER BY started_at DESC",
            ).fetchall()

        return [
            {
                **{
                    key: value
                    for key, value in dict(row).items()
                    if key not in {"tickers_json", "metadata_json"}
                },
                "tickers": json.loads(row["tickers_json"]),
                "metadata": json.loads(row["metadata_json"] or "{}"),
            }
            for row in rows
        ]

    def get_trade_traces(self, run_id: str) -> list[dict[str, Any]]:
        """Return complete serialized TradeTrace records for a run."""
        with self._connection() as conn:
            rows = conn.execute(
                """
                SELECT trace_json
                FROM trade_traces
                WHERE run_id = ?
                ORDER BY simulated_date, ticker, id
                """,
                (run_id,),
            ).fetchall()

        return [json.loads(row["trace_json"]) for row in rows]

    def delete_run(self, run_id: str) -> None:
        """Delete one run and its persisted trade traces."""
        with self._connection() as conn:
            cursor = conn.execute(
                "DELETE FROM runs WHERE run_id = ?",
                (run_id,),
            )

            if cursor.rowcount == 0:
                raise KeyError(f"Unknown run_id: {run_id}")

    # ---------- Internal helpers ----------

    @contextmanager
    def _connection(self) -> Iterator[Any]:
        """Yield a SQLite connection with transactional semantics."""
        conn = get_connection(self.db_path)

        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()


# ---------- Module-level helpers ----------


def _isoformat(value: datetime | str | None) -> str | None:
    """Normalize a datetime or string to ISO-8601 text."""
    if value is None:
        return None

    if isinstance(value, datetime):
        return value.isoformat()

    return value


def _bool_to_int(value: bool | None) -> int | None:
    """Convert an optional boolean to a SQLite-compatible 0/1 integer."""
    if value is None:
        return None

    return int(value)


def _json_dumps(obj: Any) -> str:
    """Serialize an object as compact JSON."""
    return json.dumps(
        obj,
        separators=(",", ":"),
        default=str,
    )


def _normalize_signal_direction(value: Any) -> str | None:
    """
    Normalize a signal direction to the values allowed by trade_traces.

    Database values:
    - bullish
    - bearish
    - neutral
    """
    if value is None:
        return None

    if hasattr(value, "value"):
        value = value.value

    direction = str(value).strip().lower()

    if direction.startswith("signaldirection."):
        direction = direction.rsplit(".", maxsplit=1)[-1]

    aliases = {
        "buy": "bullish",
        "long": "bullish",
        "bull": "bullish",
        "bullish": "bullish",
        "sell": "bearish",
        "short": "bearish",
        "bear": "bearish",
        "bearish": "bearish",
        "hold": "neutral",
        "flat": "neutral",
        "none": "neutral",
        "neutral": "neutral",
    }

    normalized = aliases.get(direction)

    if normalized is None:
        raise ValueError(
            "Unsupported signal direction. "
            f"Expected bullish, bearish, or neutral; received {value!r}."
        )

    return normalized


def _normalize_execution_side(value: Any) -> str | None:
    """
    Normalize an execution side to the values allowed by trade_traces.

    Database values:
    - BUY
    - SELL
    """
    if value is None:
        return None

    if hasattr(value, "value"):
        value = value.value

    side = str(value).strip().upper()

    if side.startswith("ORDERSIDE."):
        side = side.rsplit(".", maxsplit=1)[-1]

    aliases = {
        "BUY": "BUY",
        "LONG": "BUY",
        "SELL": "SELL",
        "SHORT": "SELL",
    }

    normalized = aliases.get(side)

    if normalized is None:
        raise ValueError(
            f"Unsupported execution side. Expected BUY or SELL; received {value!r}."
        )

    return normalized


def _trace_key(
    *,
    run_id: str,
    ticker: str,
    simulated_date: datetime | None,
    tick_id: Any,
) -> str:
    """
    Build a stable idempotency key for one ticker/session trace.

    A TradeTrace must be unique for its run, ticker, and historical
    simulated session. tick_id is retained as additional context.
    """
    session_identity = (
        _isoformat(simulated_date) if simulated_date is not None else "unscheduled"
    )

    tick_identity = str(tick_id) if tick_id is not None else "no-tick-id"

    return f"{run_id}:{ticker}:{session_identity}:{tick_identity}"
