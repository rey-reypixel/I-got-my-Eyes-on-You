"""MCP server exposing read-only query tools over the surveillance event log
(src/event_log.py). Standalone process, decoupled from the live video
pipeline -- it only ever reads the SQLite database that main.py writes to
when run with --event-log <path>. No shared process, no threading with the
OpenCV loop.

Run directly (stdio transport, the standard for local MCP clients):
    python mcp_server/server.py --db-path surveillance_events.db

Or point it at an existing log via the SURVEILLANCE_EVENT_LOG_DB env var.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mcp.server.fastmcp import FastMCP

from src import event_log

DEFAULT_DB_PATH = ROOT / "surveillance_events.db"
DB_PATH = Path(os.environ.get("SURVEILLANCE_EVENT_LOG_DB", str(DEFAULT_DB_PATH)))

mcp = FastMCP("surveillance-events")


@mcp.tool()
def list_runs() -> List[Dict[str, Any]]:
    """List every recorded surveillance run (video source, start/end time)."""
    return event_log.list_runs(DB_PATH)


@mcp.tool()
def list_alerts(
    run_id: Optional[int] = None,
    state: Optional[str] = None,
    start_time: Optional[str] = None,
    end_time: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """List logged alert events (INTRUDER, WARNING, ABANDONED, PANIC,
    SUSPICIOUS_ACTIVITY), optionally filtered by run, a specific state, and
    an ISO-8601 timestamp range (e.g. "2026-07-21T10:00:00").
    """
    return event_log.list_alerts(DB_PATH, run_id=run_id, state=state, start_time=start_time, end_time=end_time)


@mcp.tool()
def get_track_history(run_id: int, entity_type: str, entity_id: int) -> List[Dict[str, Any]]:
    """Get the full state-transition timeline (including recoveries back to
    NORMAL/ATTENDED) for one specific tracked person or bag within a run.
    entity_type must be "person" or "bag"; entity_id is only unique within
    that run.
    """
    return event_log.get_track_history(DB_PATH, run_id, entity_type, entity_id)


@mcp.tool()
def get_run_summary(run_id: int) -> Dict[str, Any]:
    """Get summary statistics for one recorded run: source, duration, alert
    counts by state, and unique tracked-entity counts.
    """
    return event_log.get_run_summary(DB_PATH, run_id)


def main() -> None:
    parser = argparse.ArgumentParser(description="MCP server for the surveillance event log")
    parser.add_argument("--db-path", type=str, default=None, help="Path to the SQLite event log (overrides SURVEILLANCE_EVENT_LOG_DB)")
    args = parser.parse_args()

    global DB_PATH
    if args.db_path:
        DB_PATH = Path(args.db_path)

    event_log.init_db(DB_PATH)
    mcp.run()


if __name__ == "__main__":
    main()
