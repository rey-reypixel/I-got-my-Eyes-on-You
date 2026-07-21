"""Evaluates person detection/tracking accuracy against real VIRAT ground-truth
annotations, instead of the hardcoded "assume 5 objects" heuristic used by
metrics_report/summarize_metrics.py (see bugs_and_debugs.txt #31).

Ground truth format (VIRAT viratdata.objects.txt): object_id, object_duration,
current_frame, bbox_left_x, bbox_top_y, bbox_width, bbox_height, object_type
(object_type 1 == person). Per-frame boxes are matched to our tracker's
detections via greedy IoU (no scipy in this environment, so no Hungarian
assignment -- greedy-by-descending-IoU is a standard, defensible simplification
for this purpose).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import cv2

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ultralytics import YOLO

from src.detection import letterbox_resize

PERSON_OBJECT_TYPE = 1


def load_ground_truth(
    objects_path: Path, start_frame: int, end_frame: int
) -> Dict[int, List[Tuple[int, float, float, float, float]]]:
    """Returns {frame_index: [(gt_object_id, x1, y1, x2, y2), ...]} for person-only
    boxes with start_frame <= frame_index < end_frame. frame_index stays in the
    video's own absolute numbering (not re-based to 0) so it lines up directly
    with the detection pass below, which seeks to start_frame the same way.
    """
    gt_by_frame: Dict[int, List[Tuple[int, float, float, float, float]]] = {}
    with objects_path.open("r", encoding="utf-8") as f:
        for line in f:
            parts = line.split()
            if len(parts) != 8:
                continue
            obj_id, _duration, cur_frame, x, y, w, h, obj_type = map(int, parts)
            if obj_type != PERSON_OBJECT_TYPE or not (start_frame <= cur_frame < end_frame):
                continue
            gt_by_frame.setdefault(cur_frame, []).append((obj_id, float(x), float(y), float(x + w), float(y + h)))
    return gt_by_frame


def iou_xyxy(a: Tuple[float, float, float, float], b: Tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    inter_x1, inter_y1 = max(ax1, bx1), max(ay1, by1)
    inter_x2, inter_y2 = min(ax2, bx2), min(ay2, by2)
    inter_w, inter_h = max(0.0, inter_x2 - inter_x1), max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h
    if inter_area <= 0.0:
        return 0.0
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    return inter_area / (area_a + area_b - inter_area)


def run_detection_pass(
    video_path: Path, start_frame: int, end_frame: int, confidence: float, imgsz: int = 640
) -> Dict[int, List[Tuple[int, float, float, float, float]]]:
    """Returns {frame_index: [(track_id, x1, y1, x2, y2), ...]} in ORIGINAL video
    coordinate space (the letterbox transform is inverted so boxes are directly
    comparable to the ground truth, which is defined in original-frame pixels).
    frame_index is the video's absolute frame number, matching load_ground_truth.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open video: {video_path}")
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    src_w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    src_h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    scale = min(imgsz / src_w, imgsz / src_h)
    pad_x = (imgsz - src_w * scale) / 2.0
    pad_y = (imgsz - src_h * scale) / 2.0

    model = YOLO("yolov8n.pt")
    detections_by_frame: Dict[int, List[Tuple[int, float, float, float, float]]] = {}

    frame_idx = start_frame
    while cap.isOpened() and frame_idx < end_frame:
        ret, frame = cap.read()
        if not ret or frame is None:
            break

        frame_resized = letterbox_resize(frame, imgsz)
        results = model.track(
            frame_resized, persist=True, tracker="bytetrack.yaml",
            conf=confidence, imgsz=imgsz, device="cpu", classes=[0, 24, 26, 28], verbose=False,
        )

        boxes_this_frame = []
        if results and getattr(results[0].boxes, "id", None) is not None:
            xyxy = results[0].boxes.xyxy.cpu().numpy()
            ids = results[0].boxes.id.cpu().numpy().astype(int)
            clss = results[0].boxes.cls.cpu().numpy().astype(int)
            for (x1, y1, x2, y2), track_id, cls in zip(xyxy, ids, clss):
                if cls != 0:  # person only
                    continue
                orig_x1 = (x1 - pad_x) / scale
                orig_y1 = (y1 - pad_y) / scale
                orig_x2 = (x2 - pad_x) / scale
                orig_y2 = (y2 - pad_y) / scale
                boxes_this_frame.append((int(track_id), orig_x1, orig_y1, orig_x2, orig_y2))

        detections_by_frame[frame_idx] = boxes_this_frame
        frame_idx += 1

    cap.release()
    return detections_by_frame


def evaluate(
    gt_by_frame: Dict[int, List[Tuple[int, float, float, float, float]]],
    det_by_frame: Dict[int, List[Tuple[int, float, float, float, float]]],
    iou_threshold: float = 0.5,
    size_threshold_px: float = 40.0,
    model_input_scale: float = 1.0,
) -> Dict[str, float]:
    """size_threshold_px is applied in MODEL-INPUT space (after the letterbox
    scale), not original-video pixels -- a box that looks large in a 1280x720
    frame can still be tiny in the 640x640 image YOLO actually processes.
    """
    tp = fp = fn = 0
    id_switches = 0
    last_matched_det_id: Dict[int, int] = {}  # gt_object_id -> last matched track_id
    small_total = small_matched = large_total = large_matched = 0

    for frame_idx in sorted(gt_by_frame.keys()):
        gt_boxes = gt_by_frame[frame_idx]
        det_boxes = det_by_frame.get(frame_idx, [])

        pairs = []
        for gi, (gt_id, *gt_box) in enumerate(gt_boxes):
            for di, (det_id, *det_box) in enumerate(det_boxes):
                iou = iou_xyxy(tuple(gt_box), tuple(det_box))
                if iou >= iou_threshold:
                    pairs.append((iou, gi, di))
        pairs.sort(key=lambda p: p[0], reverse=True)

        matched_gt, matched_det = set(), set()
        for iou, gi, di in pairs:
            if gi in matched_gt or di in matched_det:
                continue
            matched_gt.add(gi)
            matched_det.add(di)
            gt_id = gt_boxes[gi][0]
            det_id = det_boxes[di][0]
            tp += 1
            if gt_id in last_matched_det_id and last_matched_det_id[gt_id] != det_id:
                id_switches += 1
            last_matched_det_id[gt_id] = det_id

        fn += len(gt_boxes) - len(matched_gt)
        fp += len(det_boxes) - len(matched_det)

        for gi, (gt_id, x1, y1, x2, y2) in enumerate(gt_boxes):
            is_small = max(x2 - x1, y2 - y1) * model_input_scale < size_threshold_px
            if is_small:
                small_total += 1
                small_matched += 1 if gi in matched_gt else 0
            else:
                large_total += 1
                large_matched += 1 if gi in matched_gt else 0

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    return {
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "precision": precision,
        "recall": recall,
        "id_switches": id_switches,
        "small_object_count": small_total,
        "small_object_recall": (small_matched / small_total) if small_total > 0 else 0.0,
        "large_object_count": large_total,
        "large_object_recall": (large_matched / large_total) if large_total > 0 else 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate person tracking against VIRAT ground truth")
    parser.add_argument("--video", type=str, required=True)
    parser.add_argument("--ground-truth", type=str, required=True)
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--max-frames", type=int, default=300, help="Number of frames to evaluate, starting at --start-frame")
    parser.add_argument("--iou-threshold", type=float, default=0.5)
    args = parser.parse_args()

    video_path = ROOT / args.video
    gt_path = ROOT / args.ground_truth
    end_frame = args.start_frame + args.max_frames

    cap = cv2.VideoCapture(str(video_path))
    src_w, src_h = cap.get(cv2.CAP_PROP_FRAME_WIDTH), cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    cap.release()
    model_input_scale = min(640 / src_w, 640 / src_h)
    print(f"Source video: {int(src_w)}x{int(src_h)}, model-input scale: {model_input_scale:.4f}")
    print(f"Evaluating frames [{args.start_frame}, {end_frame})")

    gt_by_frame = load_ground_truth(gt_path, args.start_frame, end_frame)
    gt_object_count = len({oid for boxes in gt_by_frame.values() for oid, *_ in boxes})
    total_gt_boxes = sum(len(v) for v in gt_by_frame.values())
    print(f"Loaded ground truth: {total_gt_boxes} person boxes across {len(gt_by_frame)} frames, "
          f"{gt_object_count} unique person tracks.")

    for label, confidence in (("baseline (conf=0.25)", 0.25), ("custom (conf=0.35)", 0.35)):
        print(f"\nRunning detection pass: {label} ...")
        det_by_frame = run_detection_pass(video_path, args.start_frame, end_frame, confidence)
        metrics = evaluate(gt_by_frame, det_by_frame, args.iou_threshold, model_input_scale=model_input_scale)
        print(f"=== {label} vs ground truth (IoU >= {args.iou_threshold}) ===")
        for key, value in metrics.items():
            if key in ("precision", "recall", "small_object_recall", "large_object_recall"):
                print(f"  {key}: {value:.4f}")
            else:
                print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
