"""Demo script: shows the MCP server's tools answering real questions about
a recorded run, without needing a full MCP client handshake set up live.
Run this AFTER a main.py run with --event-log has produced some alerts.

    python main.py --source <video> --no-show --event-log demo_events.db
    python demo_mcp_query.py demo_events.db
"""
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))


async def main():
    db_path = sys.argv[1] if len(sys.argv) > 1 else "demo_events.db"

    from mcp_server import server
    from src import event_log
    server.DB_PATH = db_path

    # Idempotent (CREATE TABLE IF NOT EXISTS): safe to call even if this path
    # already has data. Without this, pointing the script at a path that was
    # never initialized by a prior `main.py --event-log` run (wrong filename,
    # typo, or a run started without --event-log) crashes with a raw
    # sqlite3.OperationalError: no such table: runs instead of the friendly
    # "No runs found" message below -- exactly the kind of thing you don't
    # want surfacing live mid-demo.
    event_log.init_db(db_path)

    print(f"\n=== Querying {db_path} via the MCP server's tools ===\n")

    runs = (await server.mcp.call_tool("list_runs", {}))[1]["result"]
    print(f"Recorded runs: {len(runs)}")
    for r in runs:
        print(f"  run_id={r['run_id']}  source={r['source']}  started={r['started_at']}")

    if not runs:
        print("\nNo runs found -- run main.py with --event-log first.")
        return

    run_id = runs[-1]["run_id"]
    print(f"\n--- Alerts for run_id={run_id} ---")
    alerts = (await server.mcp.call_tool("list_alerts", {"run_id": run_id}))[1]["result"]
    if not alerts:
        print("  (no alerts triggered in this run)")
    for a in alerts:
        print(f"  frame={a['frame_id']:<5} {a['entity_type']:<6} id={a['entity_id']:<3} -> {a['state']}")

    print(f"\n--- Summary for run_id={run_id} ---")
    summary = (await server.mcp.call_tool("get_run_summary", {"run_id": run_id}))[1]["result"]
    print(f"  Source: {summary['run']['source']}")
    print(f"  Alert counts by state: {summary['state_counts']}")
    print(f"  Unique tracked entities: {summary['unique_entity_counts']}")
    print(f"  Total alert events: {summary['total_alert_events']}")
    print()


if __name__ == "__main__":
    asyncio.run(main())
