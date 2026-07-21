"""Persisted event log for surveillance runs: a lightweight SQLite-backed
record of every state transition (INTRUDER, WARNING, ABANDONED, PANIC,
SUSPICIOUS_ACTIVITY, and recoveries back to NORMAL/ATTENDED) so a run's
history can be queried after the fact -- including by the MCP server in
mcp_server/server.py, which is a read-only consumer of this same store.

Track IDs (person_id/bag_id) are only unique *within* a single run (ByteTrack
restarts numbering each run), so every query is scoped by run_id.
"""
from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

ALERT_STATES = ("INTRUDER", "WARNING", "ABANDONED", "PANIC", "SUSPICIOUS_ACTIVITY")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    run_id INTEGER PRIMARY KEY AUTOINCREMENT,
    source TEXT NOT NULL,
    started_at TEXT NOT NULL,
    ended_at TEXT
);

CREATE TABLE IF NOT EXISTS events (
    event_id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id INTEGER NOT NULL REFERENCES runs(run_id),
    frame_id INTEGER NOT NULL,
    timestamp TEXT NOT NULL,
    entity_type TEXT NOT NULL,
    entity_id INTEGER NOT NULL,
    state TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_events_run ON events(run_id);
CREATE INDEX IF NOT EXISTS idx_events_state ON events(state);
CREATE INDEX IF NOT EXISTS idx_events_entity ON events(run_id, entity_type, entity_id);
"""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


@contextmanager
def _connect(db_path: str | Path) -> Iterator[sqlite3.Connection]:
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db(db_path: str | Path) -> None:
    with _connect(db_path) as conn:
        conn.executescript(_SCHEMA)


def start_run(db_path: str | Path, source: str) -> int:
    with _connect(db_path) as conn:
        cursor = conn.execute(
            "INSERT INTO runs (source, started_at, ended_at) VALUES (?, ?, NULL)",
            (source, _now()),
        )
        return int(cursor.lastrowid)


def end_run(db_path: str | Path, run_id: int) -> None:
    with _connect(db_path) as conn:
        conn.execute("UPDATE runs SET ended_at = ? WHERE run_id = ?", (_now(), run_id))


def log_state_change(
    db_path: str | Path, run_id: int, frame_id: int, entity_type: str, entity_id: int, state: str
) -> None:
    with _connect(db_path) as conn:
        conn.execute(
            "INSERT INTO events (run_id, frame_id, timestamp, entity_type, entity_id, state) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (run_id, frame_id, _now(), entity_type, entity_id, state),
        )


def list_runs(db_path: str | Path) -> List[Dict[str, Any]]:
    with _connect(db_path) as conn:
        rows = conn.execute("SELECT * FROM runs ORDER BY run_id").fetchall()
        return [dict(row) for row in rows]


def list_alerts(
    db_path: str | Path,
    run_id: Optional[int] = None,
    state: Optional[str] = None,
    start_time: Optional[str] = None,
    end_time: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Lists logged ALERT_STATES events (excludes recoveries to NORMAL/ATTENDED
    unless `state` explicitly asks for one of those), optionally filtered by
    run, a specific state, and an ISO-8601 timestamp range.
    """
    states = (state,) if state else ALERT_STATES
    query = f"SELECT * FROM events WHERE state IN ({','.join('?' for _ in states)})"
    params: List[Any] = list(states)

    if run_id is not None:
        query += " AND run_id = ?"
        params.append(run_id)
    if start_time is not None:
        query += " AND timestamp >= ?"
        params.append(start_time)
    if end_time is not None:
        query += " AND timestamp <= ?"
        params.append(end_time)
    query += " ORDER BY timestamp"

    with _connect(db_path) as conn:
        rows = conn.execute(query, params).fetchall()
        return [dict(row) for row in rows]


def get_track_history(db_path: str | Path, run_id: int, entity_type: str, entity_id: int) -> List[Dict[str, Any]]:
    """Full state-transition timeline (including recoveries) for one specific
    tracked person or bag within a single run.
    """
    with _connect(db_path) as conn:
        rows = conn.execute(
            "SELECT * FROM events WHERE run_id = ? AND entity_type = ? AND entity_id = ? ORDER BY timestamp",
            (run_id, entity_type, entity_id),
        ).fetchall()
        return [dict(row) for row in rows]


def get_run_summary(db_path: str | Path, run_id: int) -> Dict[str, Any]:
    with _connect(db_path) as conn:
        run_row = conn.execute("SELECT * FROM runs WHERE run_id = ?", (run_id,)).fetchone()
        if run_row is None:
            raise ValueError(f"No run with run_id={run_id}")

        state_counts = {
            row["state"]: row["n"]
            for row in conn.execute(
                "SELECT state, COUNT(*) AS n FROM events WHERE run_id = ? GROUP BY state", (run_id,)
            ).fetchall()
        }
        entity_counts = {
            row["entity_type"]: row["n"]
            for row in conn.execute(
                "SELECT entity_type, COUNT(DISTINCT entity_id) AS n FROM events WHERE run_id = ? GROUP BY entity_type",
                (run_id,),
            ).fetchall()
        }
        total_alerts = sum(state_counts.get(s, 0) for s in ALERT_STATES)

    return {
        "run": dict(run_row),
        "state_counts": state_counts,
        "unique_entity_counts": entity_counts,
        "total_alert_events": total_alerts,
    }


__all__ = [
    "ALERT_STATES",
    "init_db",
    "start_run",
    "end_run",
    "log_state_change",
    "list_runs",
    "list_alerts",
    "get_track_history",
    "get_run_summary",
]
