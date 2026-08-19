"""GPU model-size / confidence / tracker ablation sweep against real VIRAT
ground truth -- the paper's new ablation core (project plan Phase 2).

Replaces the single CPU/YOLOv8n-nano data point the paper previously had
(9.4 FPS, one clip) with a real FPS-vs-precision/recall Pareto frontier
across model sizes, confidence thresholds, and (once configs/trackers/
botsort_reid.yaml has been re-verified against real footage) tracker
choices -- affordable now on a B200 in a way it wasn't on CPU.

Reuses benchmark/evaluate_against_ground_truth.py's already-verified
load_ground_truth/run_detection_pass/evaluate functions directly rather than
reimplementing detection-pass or IoU-matching logic -- this sweep only adds
the grid iteration and CSV/latency measurement around them.

PLACEHOLDER STATUS: written without GPU access in this environment, not yet
run. Each sweep point downloads its own model weights on first use (standard
ultralytics auto-download behavior) -- expect the first run to be slower
than subsequent ones purely from that, and account for it separately from
actual inference latency when reporting numbers (don't include download time
in a latency figure).
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path
from typing import List, Tuple

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmark.evaluate_against_ground_truth import evaluate, load_ground_truth, run_detection_pass

# Default sweep grid. Model names verified against the live Ultralytics
# release feed at write-time (2026-08-19): latest ultralytics is 8.4.123,
# current flagship family is YOLO26 (yolo26n.pt confirmed to exist, along
# with s/m/l/x variants and a yolo26n-pose.pt for the pose signal). Re-check
# these are still the current recommendation before running a real sweep --
# Ultralytics ships new generations quickly and this list will age.
DEFAULT_MODELS = ["yolov8n.pt", "yolo26n.pt", "yolo26s.pt", "yolo26m.pt"]
DEFAULT_CONFIDENCES = [0.25, 0.35, 0.45]
DEFAULT_TRACKERS = ["bytetrack.yaml"]  # add "configs/trackers/botsort_reid.yaml" once re-verified


def run_sweep(
    video_path: Path,
    gt_path: Path,
    start_frame: int,
    max_frames: int,
    models: List[str],
    confidences: List[float],
    trackers: List[str],
    device: str,
    half: bool,
    iou_threshold: float,
    output_csv: Path,
) -> None:
    end_frame = start_frame + max_frames

    import cv2

    cap = cv2.VideoCapture(str(video_path))
    src_w, src_h = cap.get(cv2.CAP_PROP_FRAME_WIDTH), cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    cap.release()
    model_input_scale = min(640 / src_w, 640 / src_h)

    gt_by_frame = load_ground_truth(gt_path, start_frame, end_frame)

    rows: List[Tuple] = []
    header = [
        "model_path", "device", "half", "tracker", "confidence",
        "precision", "recall", "id_switches",
        "small_object_recall", "large_object_recall",
        "wall_seconds", "frames_processed", "fps",
    ]

    for model_path in models:
        for tracker in trackers:
            for confidence in confidences:
                print(f"\n--- model={model_path} tracker={tracker} conf={confidence} device={device} half={half} ---")
                t0 = time.perf_counter()
                det_by_frame = run_detection_pass(
                    video_path, start_frame, end_frame, confidence,
                    model_path=model_path, device=device, tracker=tracker, half=half,
                )
                wall_seconds = time.perf_counter() - t0
                frames_processed = len(det_by_frame)
                fps = frames_processed / wall_seconds if wall_seconds > 0 else 0.0

                metrics = evaluate(gt_by_frame, det_by_frame, iou_threshold, model_input_scale=model_input_scale)
                print(f"  precision={metrics['precision']:.4f} recall={metrics['recall']:.4f} "
                      f"id_switches={metrics['id_switches']} fps={fps:.2f}")

                rows.append((
                    model_path, device, half, tracker, confidence,
                    f"{metrics['precision']:.4f}", f"{metrics['recall']:.4f}", metrics["id_switches"],
                    f"{metrics['small_object_recall']:.4f}", f"{metrics['large_object_recall']:.4f}",
                    f"{wall_seconds:.3f}", frames_processed, f"{fps:.2f}",
                ))

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)
    print(f"\nWrote {len(rows)} sweep rows to {output_csv}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Model-size/confidence/tracker ablation sweep against VIRAT ground truth")
    parser.add_argument("--video", type=str, required=True)
    parser.add_argument("--ground-truth", type=str, required=True)
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--max-frames", type=int, default=300)
    parser.add_argument("--iou-threshold", type=float, default=0.5)
    parser.add_argument("--device", type=str, default="cuda", help='Sweep is intended for GPU; pass "cpu" only for a small smoke test')
    parser.add_argument("--half", action="store_true")
    parser.add_argument("--models", type=str, nargs="+", default=DEFAULT_MODELS)
    parser.add_argument("--confidences", type=float, nargs="+", default=DEFAULT_CONFIDENCES)
    parser.add_argument("--trackers", type=str, nargs="+", default=DEFAULT_TRACKERS)
    parser.add_argument("--output", type=str, default="benchmark/model_sweep_results.csv")
    args = parser.parse_args()

    run_sweep(
        video_path=ROOT / args.video,
        gt_path=ROOT / args.ground_truth,
        start_frame=args.start_frame,
        max_frames=args.max_frames,
        models=args.models,
        confidences=args.confidences,
        trackers=args.trackers,
        device=args.device,
        half=args.half,
        iou_threshold=args.iou_threshold,
        output_csv=ROOT / args.output,
    )


if __name__ == "__main__":
    main()
