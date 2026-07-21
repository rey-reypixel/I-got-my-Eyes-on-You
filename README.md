# I-got-my-Eyes-on-You (sweet summer of '26 <3)
Smart CCTV Surveillance with Threat Detection

An AI-powered CCTV surveillance prototype that uses YOLOv8, tracking (ByteTrack), dynamic external configurations, thread-safe memory eviction, and state-machine reasoning to detect:
- **Suspicious activities** (normalized panic motion spikes)
- **Unauthorized access** (dynamic polygon-based region of interest)
- **Abandoned objects** (hysteresis-based temporal asset warning/alerts)

---

## 📁 Repository Structure
*   `src/`: Core logic modules ([detection.py](file:///f:/Summer%20Internship/I-got-my-Eyes-on-You/src/detection.py), [tracking.py](file:///f:/Summer%20Internship/I-got-my-Eyes-on-You/src/tracking.py), [state_machine.py](file:///f:/Summer%20Internship/I-got-my-Eyes-on-You/src/state_machine.py), [engineering_layers.py](file:///f:/Summer%20Internship/I-got-my-Eyes-on-You/src/engineering_layers.py))
*   `benchmark/`: Benchmark evaluation entry points and telemetries ([baseline_eval.py](file:///f:/Summer%20Internship/I-got-my-Eyes-on-You/benchmark/baseline_eval.py), [custom_eval.py](file:///f:/Summer%20Internship/I-got-my-Eyes-on-You/benchmark/custom_eval.py))
*   `metrics_report/`: Performance summary utilities ([summarize_metrics.py](file:///f:/Summer%20Internship/I-got-my-Eyes-on-You/metrics_report/summarize_metrics.py))
*   `tests/`: Automated unit tests for configuration, metrics, and state handling
*   `config.json`: Dynamic boundaries and threat threshold configurations ([config.json](file:///f:/Summer%20Internship/I-got-my-Eyes-on-You/config.json))

---

## 🚀 Setup & Execution

### 1. Verification & Diagnostics
Run the diagnostic check to ensure all Python packages are present, directory structures are intact, and hardware acceleration is configured:
```bash
python check.py
```

### 2. Live Surveillance Execution
Run the live pipeline using the CLI entry point:
```bash
# Run on default discovered video with config overrides
python main.py --config config.json --max-frames 100

# Run headlessly (no visual overlay window)
python main.py --no-show

# Run on a custom video file or camera feed (e.g. webcam index 0)
python main.py --source 0
```

### 3. Benchmarking & Evaluations
Clock performance metrics on both baseline and custom enhanced pipelines:
```bash
# Run baseline evaluation
python benchmark/baseline_eval.py --max-frames 100

# Run custom state-machine evaluation
python benchmark/custom_eval.py --max-frames 100 --config config.json

# Summarize performance metrics (latency, FPS, unique IDs)
python metrics_report/summarize_metrics.py
```

### 4. Unit Tests
Run the unit test suites:
```bash
python -m unittest discover -s tests
```
