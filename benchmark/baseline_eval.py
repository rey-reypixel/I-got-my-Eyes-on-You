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


def run_baseline_evaluation(video_source: Optional[str] = None, max_frames: int = 10) -> Path:
    """Run the baseline tracker loop and write telemetry rows to a CSV file."""
    video_path = discover_video_source(video_source)
    output_path = ROOT / "benchmark" / "baseline_logs.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open video source: {video_path}")

    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["frame_id", "object_id", "class_label", "latency_seconds", "instantaneous_fps"])

        frame_id = 0
        model = YOLO("yolov8n.pt")

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret or frame is None:
                break

            frame_id += 1
            if max_frames and frame_id > max_frames:
                break

            frame_resized = cv2.resize(frame, (640, 640))
            t_start = time.perf_counter()
            results = model.track(frame_resized, persist=True, tracker="bytetrack.yaml")
            t_end = time.perf_counter()

            latency = t_end - t_start
            fps = 1.0 / latency if latency > 0 else 0.0

            if results and getattr(results[0].boxes, "id", None) is not None:
                boxes = results[0].boxes.xyxy.cpu().numpy()
                ids = results[0].boxes.id.cpu().numpy().astype(int)
                clss = results[0].boxes.cls.cpu().numpy().astype(int)

                for obj_id, cls in zip(ids, clss):
                    class_label = model.names[cls]
                    writer.writerow([frame_id, obj_id, class_label, f"{latency:.6f}", f"{fps:.2f}"])

    cap.release()
    return output_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the baseline evaluation loop")
    parser.add_argument("--video-source", type=str, default=None, help="Optional explicit path to a benchmark video")
    parser.add_argument("--max-frames", type=int, default=10, help="Maximum number of frames to process")
    args = parser.parse_args()

    output_path = run_baseline_evaluation(video_source=args.video_source, max_frames=args.max_frames)
    print(f"Baseline evaluation completed. Wrote {output_path}")
