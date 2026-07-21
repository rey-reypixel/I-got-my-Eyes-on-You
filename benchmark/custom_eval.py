from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path
from typing import Optional

import cv2

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ultralytics import YOLO

from src.detection import letterbox_resize
from src.tracking import process_tracking_state_machine


def discover_video_source(explicit_source: Optional[str] = None) -> Path:
    """Locate a benchmark video from CLI input or the repository's raw data directories."""
    if explicit_source:
        candidate = Path(explicit_source)
        if candidate.is_file():
            return candidate
        if not candidate.is_absolute():
            candidate = ROOT / candidate
        if candidate.is_file():
            return candidate

    for pattern in ("*.mp4", "*.avi", "*.mkv", "*.mov", "*.mpg"):
        for candidate in sorted(ROOT.rglob(pattern)):
            if candidate.is_file():
                return candidate

    raise FileNotFoundError("No video file was found under the repository data directories.")


def run_custom_evaluation(
    video_source: Optional[str] = None,
    max_frames: int = 10,
    config_path: str = "config.json",
) -> Path:
    """Run the state-machine-enhanced evaluation loop and write telemetry rows to a CSV file."""
    import json
    video_path = discover_video_source(video_source)
    output_path = ROOT / "benchmark" / "custom_logs.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    config_data = {}
    resolved_config = ROOT / config_path
    if resolved_config.is_file():
        try:
            with resolved_config.open("r", encoding="utf-8") as f:
                config_data = json.load(f)
        except Exception as e:
            print(f"Warning: Failed to load configuration from {resolved_config}: {e}")

    # Read parameters
    imgsz = config_data.get("detection", {}).get("imgsz", 640)
    confidence = config_data.get("detection", {}).get("confidence", 0.25)
    device = config_data.get("detection", {}).get("device", "cpu")
    state_machine_config = config_data.get("state_machine", {})
    state_machine_config["imgsz"] = imgsz  # keep clipping-boundary checks in sync with the real engine
    th_frames = state_machine_config.get("th_frames", 10)

    # Decode and scale polygon vertices
    raw_polygon = config_data.get("roi_polygon")
    if raw_polygon:
        roi_vertices = []
        for pt in raw_polygon:
            px, py = pt
            if isinstance(px, float) and 0.0 <= px <= 1.0:
                px = px * imgsz
            if isinstance(py, float) and 0.0 <= py <= 1.0:
                py = py * imgsz
            roi_vertices.append((px, py))
    else:
        roi_vertices = [(64, 64), (576, 96), (608, 544), (128, 576)]

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open video source: {video_path}")

    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow([
            "frame_id",
            "object_id",
            "class_label",
            "assigned_owner_id",
            "current_state",
            "threat_alert_triggered",
            "latency_seconds",
            "instantaneous_fps",
        ])

        frame_id = 0
        model = YOLO("yolov8n.pt")

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret or frame is None:
                break

            frame_id += 1
            if max_frames and frame_id > max_frames:
                break

            frame_resized = letterbox_resize(frame, imgsz)

            results = model.track(
                frame_resized,
                persist=True,
                tracker="bytetrack.yaml",
                conf=confidence,
                device=device,
                imgsz=imgsz,
                classes=[0, 24, 26, 28],
            )
            
            # 2. Clock ONLY our custom engineered state machine layers
            t_logic_start = time.perf_counter()
            custom_metrics = process_tracking_state_machine(
                results,
                frame_id,
                polygon_vertices=roi_vertices,
                th_frames=th_frames,
                state_machine_config=state_machine_config,
            )
            t_logic_end = time.perf_counter()

            # Isolate the logic execution latency
            latency = t_logic_end - t_logic_start
            fps = 1.0 / latency if latency > 0 else 0.0

            for metric in custom_metrics:
                writer.writerow([
                    frame_id,
                    metric["object_id"],
                    metric["class_label"],
                    metric["assigned_owner_id"],
                    metric["current_state"],
                    metric["threat_alert_triggered"],
                    f"{latency:.6f}",
                    f"{fps:.2f}",
                ])

    cap.release()
    return output_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the custom evaluation loop")
    parser.add_argument("--video-source", type=str, default=None, help="Optional explicit path to a benchmark video")
    parser.add_argument("--max-frames", type=int, default=10, help="Maximum number of frames to process")
    parser.add_argument("--config", type=str, default="config.json", help="Path to the config.json file")
    args = parser.parse_args()

    output_path = run_custom_evaluation(
        video_source=args.video_source,
        max_frames=args.max_frames,
        config_path=args.config,
    )
    print(f"Custom evaluation completed. Wrote {output_path}")
