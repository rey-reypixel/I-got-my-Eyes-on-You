"""Frame-level AUC-ROC evaluator for Objective 3 (suspicious-activity detection)
against the standard CUHK Avenue benchmark protocol.

This is the field-standard metric for video anomaly detection (used to report
results on Avenue, UCSD Ped1/Ped2, ShanghaiTech, UCF-Crime, etc. across the
anomaly-detection literature), and gives Objective 3 a real, comparable number
for the first time -- previously only validated qualitatively (real-video
testing + bug-fixing, see paper_draft.txt Section VI).

PLACEHOLDER / NOT YET RUNNABLE: the Avenue video files already present under
data/raw/suspicious_activities/Avenue Dataset/ are the complete, correct
standard 16-train/21-test set (confirmed during project exploration -- frame
counts match the published benchmark exactly), but the frame-level ground-truth
labels needed for this protocol are NOT included in this checkout. They ship
separately from the video data as `ground_truth_demo.zip` from the CUHK Avenue
dataset page (search "CUHK Avenue Dataset ground_truth_demo" -- the exact
current URL should be confirmed at download time, department pages move).
Per this project's policy on downloading external files, get explicit
confirmation before fetching it, then extract so that
`data/raw/suspicious_activities/Avenue Dataset/ground_truth_demo/testing_label_mask/`
contains one .mat file per test video (matching testing_videos/01.avi..21.avi).

Anomaly score design (Objective 3's own two signals, reused rather than a new
model): for each processed frame, take the maximum over all currently-tracked
people of (kinematic_suspicious_flag OR limb_suspicious_flag) as a 0/1 score,
OR -- if a continuous score is preferred for a smoother ROC curve -- the
smoothed behavioral confidence value already computed by
process_behavioral_smoothing (src/tracking.py). This module supports both via
--score-mode.

Ground-truth .mat format: scipy.io.loadmat() on each test video's mask file
returns a MATLAB struct; the exact variable name inside it is not hardcoded
here (recall from the CUHK-Avenue's known layout: a per-frame pixel mask
array, "any nonzero pixel in a frame's mask" == that frame is anomalous, the
same convention as the 'vol' key already confirmed present in this project's
own testing_vol/*.mat files) -- instead the loader picks the first key that
isn't one of scipy's own metadata keys (__header__/__version__/__globals__)
and validates its shape looks like a per-frame stack before trusting it,
rather than assuming a specific name that may not hold across dataset
mirrors. Verify this against the real downloaded file before trusting a
result (print the discovered key/shape -- this module does, at INFO level).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

AVENUE_DIR = ROOT / "data" / "raw" / "suspicious_activities" / "Avenue Dataset"
GT_MASK_DIR = AVENUE_DIR / "ground_truth_demo" / "testing_label_mask"
TEST_VIDEO_DIR = AVENUE_DIR / "testing_videos"


def load_frame_labels_from_mat(mat_path: Path) -> "Any":
    """Loads one test video's ground-truth mask stack and reduces it to a
    per-frame binary anomaly label (1 if any pixel in that frame's mask is
    nonzero, else 0). Returns a 1D numpy array of length num_frames.

    Defensive about the exact struct key, see module docstring.
    """
    import numpy as np
    from scipy.io import loadmat

    mat = loadmat(str(mat_path))
    candidate_keys = [k for k in mat.keys() if not k.startswith("__")]
    if not candidate_keys:
        raise ValueError(f"No non-metadata keys found in {mat_path}; cannot locate the mask array.")

    # Prefer a 3D array (H, W, num_frames) if present -- that's the documented
    # shape of this project's own testing_vol/*.mat files and is the expected
    # shape for a per-frame pixel mask stack.
    best_key = None
    for key in candidate_keys:
        val = mat[key]
        if hasattr(val, "ndim") and val.ndim == 3:
            best_key = key
            break
    if best_key is None:
        best_key = candidate_keys[0]

    mask_stack = mat[best_key]
    print(f"    [avenue_auc_eval] {mat_path.name}: using key '{best_key}', shape {getattr(mask_stack, 'shape', None)}")

    if hasattr(mask_stack, "ndim") and mask_stack.ndim == 3:
        # Assume (H, W, num_frames) per this project's own vol*.mat convention;
        # if the real ground_truth_demo files use (num_frames, H, W) instead,
        # this axis assumption needs flipping -- verify against the printed
        # shape above (num_frames should match the video's actual frame count).
        num_frames = mask_stack.shape[-1]
        frame_labels = np.array([1 if np.any(mask_stack[..., i]) else 0 for i in range(num_frames)])
    elif hasattr(mask_stack, "ndim") and mask_stack.ndim == 1:
        # Already a flat per-frame label/cell array.
        frame_labels = np.array([1 if np.any(mask_stack[i]) else 0 for i in range(mask_stack.shape[0])])
    else:
        raise ValueError(f"Unexpected mask array shape {getattr(mask_stack, 'shape', None)} in {mat_path}; "
                          f"inspect this file manually and adjust load_frame_labels_from_mat.")

    return frame_labels


def compute_frame_anomaly_scores(
    video_path: Path,
    config_path: str = "config.json",
    score_mode: str = "binary",
    model_path: str = "yolov8n.pt",
    pose_model_path: str = "yolov8n-pose.pt",
) -> List[float]:
    """Runs the live suspicious-activity reasoning layer over `video_path` and
    returns one anomaly score per frame, aggregated across all people tracked
    in that frame (max, since one anomalous person should flag the frame
    regardless of how many other people are present and calm).

    Drives the real MultiModelDetectionEngine frame-by-frame (its private
    _run_inference/_route_tracking_updates methods, the same sequence run()
    uses -- see src/detection.py:606-608) rather than reimplementing the loop
    against the bare tracking module, specifically so the gated pose model
    actually fires here. An earlier version of this function called
    update_tracking_states directly, which skips _process_limb_motion_gate
    entirely (that gate lives on the engine) -- meaning score_mode="binary"
    would have silently reflected only the kinematic half of Objective 3's
    two-signal design, not the full signal the live pipeline actually alerts
    on. show_window=False avoids the interactive cv2.imshow/keyboard-handling
    branch of run(), which isn't appropriate for a headless batch evaluator.

    score_mode="binary": max(kinematic_suspicious_flag, limb_suspicious_flag) per
    person, aggregated by OR across people -> 0.0 or 1.0 per frame. Matches
    exactly what the live pipeline actually alerts on.

    score_mode="smoothed": uses anomaly_score_ema (src/tracking.py
    process_behavioral_smoothing's EMA output -- confirmed field name, not
    guessed) instead, for a continuous score and a less degenerate ROC curve;
    binary scores collapse AUC-ROC to a single operating point in effect,
    which is a legitimate but weaker way to report this metric than a
    continuous score would be. Only set for people who were actually
    pose-gated in on a given frame; 0.0 otherwise.
    """
    import time

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

    scores: List[float] = []
    frame_count = 0
    try:
        while engine.capture.isOpened():
            ret, frame = engine.capture.read()
            if not ret or frame is None:
                break

            frame_count += 1
            timestamp = time.time()
            resized_frame = engine._resize_frame(frame)
            results = engine._run_inference(resized_frame)
            engine._route_tracking_updates(resized_frame, results, timestamp, frame_id=frame_count)

            if score_mode == "binary":
                frame_score = 0.0
                for person in tracked_people.values():
                    if person.get("kinematic_suspicious_flag") or person.get("limb_suspicious_flag"):
                        frame_score = 1.0
                        break
            elif score_mode == "smoothed":
                frame_score = max(
                    (person.get("anomaly_score_ema", 0.0) for person in tracked_people.values()),
                    default=0.0,
                )
            else:
                raise ValueError(f"Unknown score_mode: {score_mode}")

            scores.append(frame_score)
    finally:
        if engine.capture is not None:
            engine.capture.release()

    return scores


def compute_auc_roc(frame_labels, frame_scores) -> float:
    from sklearn.metrics import roc_auc_score

    if len(frame_labels) != len(frame_scores):
        # Length mismatch between ground truth and computed scores is a real
        # failure mode here (dropped/failed frame reads, off-by-one in frame
        # counting) -- truncate to the shorter length rather than silently
        # padding, and surface it loudly rather than let roc_auc_score fail
        # cryptically on mismatched arrays.
        n = min(len(frame_labels), len(frame_scores))
        print(f"    [avenue_auc_eval] WARNING: label/score length mismatch "
              f"({len(frame_labels)} vs {len(frame_scores)}) -- truncating to {n}. "
              f"Investigate before trusting this AUC.")
        frame_labels, frame_scores = frame_labels[:n], frame_scores[:n]

    if len(set(frame_labels.tolist() if hasattr(frame_labels, "tolist") else frame_labels)) < 2:
        raise ValueError("Ground truth has only one class present (all normal or all anomalous) -- "
                          "AUC-ROC is undefined for this clip alone.")

    return roc_auc_score(frame_labels, frame_scores)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Frame-level AUC-ROC evaluation of suspicious-activity detection against Avenue ground truth"
    )
    parser.add_argument("--test-video", type=str, default=None,
                         help="Single test video filename under testing_videos/, e.g. 01.avi (omit to run all 21)")
    parser.add_argument("--config", type=str, default="config.json")
    parser.add_argument("--score-mode", type=str, default="binary", choices=["binary", "smoothed"])
    parser.add_argument("--model-path", type=str, default="yolov8n.pt")
    parser.add_argument("--pose-model-path", type=str, default="yolov8n-pose.pt")
    args = parser.parse_args()

    if not GT_MASK_DIR.is_dir():
        print(f"Ground-truth mask directory not found: {GT_MASK_DIR}")
        print("Download ground_truth_demo.zip from the CUHK Avenue dataset page and extract it there "
              "(ask for confirmation before fetching an external file, per this project's download policy) "
              "before running this evaluator.")
        sys.exit(1)

    if args.test_video:
        video_names = [args.test_video]
    else:
        video_names = sorted(p.name for p in TEST_VIDEO_DIR.glob("*.avi"))

    per_clip_auc = []
    for video_name in video_names:
        video_path = TEST_VIDEO_DIR / video_name
        stem = Path(video_name).stem  # e.g. "01"
        mat_path = GT_MASK_DIR / f"{stem}_label.mat"
        if not mat_path.is_file():
            print(f"  Skipping {video_name}: no matching ground-truth mask at {mat_path}")
            continue

        print(f"\n=== {video_name} ===")
        frame_labels = load_frame_labels_from_mat(mat_path)
        frame_scores = compute_frame_anomaly_scores(
            video_path, config_path=args.config, score_mode=args.score_mode,
            model_path=args.model_path, pose_model_path=args.pose_model_path,
        )

        try:
            auc = compute_auc_roc(frame_labels, frame_scores)
            print(f"  AUC-ROC: {auc:.4f}")
            per_clip_auc.append(auc)
        except ValueError as e:
            print(f"  Skipped AUC computation: {e}")

    if per_clip_auc:
        mean_auc = sum(per_clip_auc) / len(per_clip_auc)
        print(f"\n=== Mean AUC-ROC across {len(per_clip_auc)} clip(s): {mean_auc:.4f} ===")
        print("Compare against published Avenue AUC-ROC figures from the literature before citing "
              "this in the paper -- frame this as an interpretable/real-time/no-training-data "
              "heuristic vs. specialized learned models, not a beat-SOTA claim (see project plan).")


if __name__ == "__main__":
    main()
