# I-got-my-Eyes-on-You (sweet summer of '26 <3)
Smart CCTV Surveillance with Threat Detection

Could this repository BE any more of an AI-powered CCTV surveillance prototype? It uses YOLOv8, tracking (ByteTrack), dynamic external configurations, thread-safe memory eviction, and state-machine reasoning to detect:
- **Suspicious activities** (normalized panic motion spikes — basically, the system's version of "we were on a break!")
- **Unauthorized access** (dynamic polygon-based region of interest — an invisible line you should not have crossed)
- **Abandoned objects** (hysteresis-based temporal asset warning/alerts — for when your bag becomes the Ross of the frame: alone, panicking, and everyone's staring)

---

## 📁 Repository Structure
So there's structure. Try to contain your excitement.
*   `src/`: Core logic modules ([detection.py](file:///f:/Summer%20Internship/I-got-my-Eyes-on-You/src/detection.py), [tracking.py](file:///f:/Summer%20Internship/I-got-my-Eyes-on-You/src/tracking.py), [state_machine.py](file:///f:/Summer%20Internship/I-got-my-Eyes-on-You/src/state_machine.py), [engineering_layers.py](file:///f:/Summer%20Internship/I-got-my-Eyes-on-You/src/engineering_layers.py), [event_log.py](file:///f:/Summer%20Internship/I-got-my-Eyes-on-You/src/event_log.py)) — the part that actually does the thinking, such as it is, plus a memory that outlives the process
*   `mcp_server/`: A standalone MCP server ([server.py](file:///f:/Summer%20Internship/I-got-my-Eyes-on-You/mcp_server/server.py)) exposing 4 read-only tools over the event log — list_runs, list_alerts, get_track_history, get_run_summary. Only ever reads the SQLite file; shares no process with the live video loop. See `demo_mcp_query.py` for a working example of calling it.
*   `benchmark/`: Benchmark evaluation entry points and telemetries ([baseline_eval.py](file:///f:/Summer%20Internship/I-got-my-Eyes-on-You/benchmark/baseline_eval.py), [custom_eval.py](file:///f:/Summer%20Internship/I-got-my-Eyes-on-You/benchmark/custom_eval.py), [evaluate_against_ground_truth.py](file:///f:/Summer%20Internship/I-got-my-Eyes-on-You/benchmark/evaluate_against_ground_truth.py)) — where the pipeline gets timed like it's auditioning for something, and where it gets graded against real VIRAT ground truth so the timing means something
*   `metrics_report/`: Performance summary utilities ([summarize_metrics.py](file:///f:/Summer%20Internship/I-got-my-Eyes-on-You/metrics_report/summarize_metrics.py)) — turns raw numbers into numbers you can show people
*   `tests/`: Automated unit tests for configuration, metrics, and state handling — the part that yells at you before production does (66 tests and counting)
*   `paper/`: The IEEE-style paper draft documenting this project's methodology and results.
*   `config.json`: Dynamic boundaries and threat threshold configurations ([config.json](file:///f:/Summer%20Internship/I-got-my-Eyes-on-You/config.json)) — knobs, for turning
*   `bugs_and_debugs.txt`: A running, root-caused log of every defect found and fixed (37 entries and counting) — the project's honesty ledger.
*   `please_study.txt`: Full study notes — architecture, math, tools, defense-prep Q&A. Start here if you need to explain this project cold.

---

## 🚀 Setup & Execution

### 1. Verification & Diagnostics
Run the diagnostic check to make sure all your Python packages showed up, the directory structure didn't wander off, and hardware acceleration is actually configured and not just, you know, vibing:
```bash
python check.py
```

### 2. Live Surveillance Execution
Run the live pipeline using the CLI entry point. This is the "could I BE any more operational" step:
```bash
# Run on default discovered video with config overrides
python main.py --config config.json --max-frames 100

# Run headlessly (no visual overlay window, for people who trust the process)
python main.py --no-show

# Run on a custom video file or camera feed (e.g. webcam index 0)
python main.py --source 0
```

### 3. Benchmarking & Evaluations
Clock performance metrics on both the baseline and the custom enhanced pipeline, because vague confidence is not a metric:
```bash
# Run baseline evaluation
python benchmark/baseline_eval.py --max-frames 100

# Run custom state-machine evaluation
python benchmark/custom_eval.py --max-frames 100 --config config.json

# Summarize performance metrics (latency, FPS, unique IDs)
python metrics_report/summarize_metrics.py
```

### 4. Unit Tests
Run the unit test suites. Consider it the system's therapy session — it airs its issues so you don't find out about them live:
```bash
python -m unittest discover -s tests
```

### 5. Queryable Alert History (MCP)
Run a live surveillance pass with event logging enabled, then ask questions about what happened after the fact — because a surveillance system that forgets everything the moment you close it is just expensive anxiety, not a system:
```bash
# Record every alert/state-change during this run to a SQLite log
python main.py --source <video_path> --event-log demo_events.db

# Query it: run history, alerts, and a summary, all via the same MCP tools
# a real MCP client would call (this script calls them directly, in Python,
# to demo the tool layer without needing a live LLM client set up)
python demo_mcp_query.py demo_events.db

# Or point any real MCP-compatible client (Claude Desktop, Claude Code, etc.)
# directly at the server itself:
python mcp_server/server.py --db-path demo_events.db
```
Note: the MCP tool layer is real and spec-compliant, but no LLM is actually wired into this
project — `demo_mcp_query.py` calls the tools with hardcoded Python to prove the data path
works, not to demonstrate an LLM reasoning about which tool to call.
