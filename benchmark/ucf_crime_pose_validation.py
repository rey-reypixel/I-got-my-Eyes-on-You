"""Lightweight, explicitly-scoped validation of the pose-based limb-motion
suspicious-activity signal (src/tracking.py compute_limb_motion_score /
process_behavioral_smoothing) against real violence/vandalism footage.

This is deliberately NOT a full AUC-ROC benchmark protocol. The UCF-Crime
subset already present under data/raw/suspicious_activities/UFC_Crime/ (450
videos across 5 of the official 13 anomaly categories -- confirmed during
project exploration) has no Normal-class videos and no
Temporal_Anomaly_Annotation file, so there is no frame-level ground truth to
compute precision/recall/AUC against. Attempting to fake that with the
official 290-video test split would need a large (~130GB) separate download
of the full official UCF-Crime set -- explicitly out of scope for now (see
project plan Phase 3e).

What this script DOES give: a bounded, honest upgrade from the paper's
current self-admitted limitation ("architecturally complete... has not, as
of this writing, been empirically confirmed to fire correctly on real
violent or altercation footage" -- paper_draft.txt Section VI) to a real,
scoped observation: across the Fighting and Vandalism clips already on disk,
what fraction of clips ever trigger the limb-motion flag at all, and what
fraction of processed frames does it fire on. This is a coverage/activation
statistic, not a precision/recall claim -- report it as such in the paper,
do not describe it as validated accuracy.

PLACEHOLDER STATUS: written without GPU/torch access in this environment,
not yet run. Run on the GPU machine, e.g.:
  python benchmark/ucf_crime_pose_validation.py --category Fighting --max-clips 10
  python benchmark/ucf_crime_pose_validation.py --category Vandalism --max-clips 10
  python benchmark/ucf_crime_pose_validation.py --category Fighting --category Vandalism  (all clips, slow)
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

UCF_CRIME_DIR = ROOT / "data" / "raw" / "suspicious_activities" / "UFC_Crime"

# Only these two categories are relevant to validating the limb-motion signal
# specifically (sustained-stillness + violent limb motion). The other three
# on-disk categories (Burglary, Robbery, Stealing) are not primarily
# characterized by that signature and are out of scope for this script.
DEFAULT_CATEGORIES = ["Fighting", "Vandalism"]


def evaluate_clip(video_path: Path, config_path: str, model_path: str, pose_model_path: str) -> Dict[str, Any]:
    """Runs the full engine (including the gated pose model) over one clip and
    returns per-clip activation statistics for both the kinematic and
    limb-motion signals.
    """
    from src.detection import MultiModelDetectionEngine
    from src.state_machine import tracked_people

    engine = MultiModelDetectionEngine(
        source=str(video_path),
        model_path=model_path,
        pose_model_path=pose_model_path,
        config_path=config_path,
        show_window=False,
    )
    engine.initialize_stream()

    frame_count = 0
    kinematic_fire_frames = 0
    limb_fire_frames = 0
    pose_gated_frames = 0  # frames where at least one person was eligible for pose inference

    try:
        while engine.capture.isOpened():
            ret, frame = engine.capture.read()
            if not ret or frame is None:
                break
            frame_count += 1

            resized_frame = engine._resize_frame(frame)
            results = engine._run_inference(resized_frame)
            engine._route_tracking_updates(resized_frame, results, time.time(), frame_id=frame_count)

            frame_kinematic = any(p.get("kinematic_suspicious_flag") for p in tracked_people.values())
            frame_limb = any(p.get("limb_suspicious_flag") for p in tracked_people.values())
            frame_gated = any(p.get("stationary_counter", 0) > 0 for p in tracked_people.values())

            kinematic_fire_frames += 1 if frame_kinematic else 0
            limb_fire_frames += 1 if frame_limb else 0
            pose_gated_frames += 1 if frame_gated else 0
    finally:
        if engine.capture is not None:
            engine.capture.release()

    return {
        "clip": video_path.name,
        "frames_processed": frame_count,
        "kinematic_fire_frames": kinematic_fire_frames,
        "limb_fire_frames": limb_fire_frames,
        "pose_gated_frames": pose_gated_frames,
        "kinematic_ever_fired": kinematic_fire_frames > 0,
        "limb_ever_fired": limb_fire_frames > 0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Lightweight coverage/activation check of the pose-based limb-motion signal on real UCF-Crime footage"
    )
    parser.add_argument("--category", action="append", default=None,
                         help=f"UCF-Crime category subfolder to test (repeatable). Default: {DEFAULT_CATEGORIES}")
    parser.add_argument("--max-clips", type=int, default=10, help="Max clips per category (these clips run at real length -- 50/category is slow)")
    parser.add_argument("--config", type=str, default="config.json")
    parser.add_argument("--model-path", type=str, default="yolov8n.pt")
    parser.add_argument("--pose-model-path", type=str, default="yolov8n-pose.pt")
    args = parser.parse_args()

    categories = args.category or DEFAULT_CATEGORIES

    if not UCF_CRIME_DIR.is_dir():
        print(f"UCF-Crime directory not found: {UCF_CRIME_DIR}")
        sys.exit(1)

    all_results: List[Dict[str, Any]] = []
    for category in categories:
        category_dir = UCF_CRIME_DIR / category
        if not category_dir.is_dir():
            print(f"Skipping unknown category directory: {category_dir}")
            continue

        clips = sorted(category_dir.glob("*.mp4"))[: args.max_clips]
        print(f"\n=== {category}: {len(clips)} clip(s) ===")
        for clip_path in clips:
            result = evaluate_clip(clip_path, args.config, args.model_path, args.pose_model_path)
            result["category"] = category
            all_results.append(result)
            print(f"  {result['clip']}: {result['frames_processed']} frames, "
                  f"kinematic fired {result['kinematic_fire_frames']} frames, "
                  f"limb-motion fired {result['limb_fire_frames']} frames "
                  f"(pose-gated on {result['pose_gated_frames']} frames)")

    if not all_results:
        print("No clips evaluated.")
        return

    print("\n=== Summary (coverage/activation statistics, NOT precision/recall -- see module docstring) ===")
    for category in categories:
        cat_results = [r for r in all_results if r["category"] == category]
        if not cat_results:
            continue
        n = len(cat_results)
        limb_clip_coverage = sum(1 for r in cat_results if r["limb_ever_fired"]) / n
        kinematic_clip_coverage = sum(1 for r in cat_results if r["kinematic_ever_fired"]) / n
        total_frames = sum(r["frames_processed"] for r in cat_results)
        limb_frame_rate = sum(r["limb_fire_frames"] for r in cat_results) / total_frames if total_frames else 0.0
        print(f"  {category} ({n} clips, {total_frames} frames total):")
        print(f"    limb-motion fired in {limb_clip_coverage * 100:.1f}% of clips, "
              f"on {limb_frame_rate * 100:.2f}% of all processed frames")
        print(f"    kinematic signal fired in {kinematic_clip_coverage * 100:.1f}% of clips")


if __name__ == "__main__":
    main()
