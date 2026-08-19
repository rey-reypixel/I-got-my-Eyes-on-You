"""Event-level evaluator for Objective 2 (abandoned-object detection) against
real AVSS 2007 / i-LIDS ViPER-GT XML ground truth.

Closes a gap the paper draft states plainly (please_study.txt / paper_draft.txt
Section VI): only Objective 1 (person detection) has ever had a quantitative,
ground-truth-backed accuracy number. Objectives 2 and 3 were validated only
qualitatively (real-video testing + bug-fixing). This module gives Objective 2
a real one, using the genuine ViPER-GT annotations already present under
data/raw/abandoned_objects/AVSS 2007/ (EASY/MEDIUM/HARD -- confirmed present
and correctly formatted during project exploration).

Ground truth format: ViPER-GT XML. Each <sourcefile> contains <object
framespan="start:end" id="..." name="PutObject|AbandonedObject|StolenObject">
elements, each carrying one static (dynamic="false") <data:bbox> -- the
annotated object doesn't move, so one bounding box covers its whole framespan.
"AbandonedObject"'s framespan start is the ground-truth onset of abandonment;
that is what a system ABANDONED alert should be compared against.

PLACEHOLDER STATUS: load_ground_truth() below has been run against the real
AVSS 2007 XML files and is verified correct (see the __main__ smoke-check at
the bottom of this file). evaluate_abandoned_object_detection(), however, has
NOT been run -- this was written on a machine with no GPU and no
torch/ultralytics installed (see project plan). Run it for real on GPU
hardware before citing any number from it in the paper, the same discipline
this project already applies everywhere else (bugs_and_debugs.txt's own
recurring lesson: a fix that only passes synthetic tests is not yet trusted
here).
"""
from __future__ import annotations

import argparse
import sys
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

VIPER_NS = {
    "viper": "http://lamp.cfar.umd.edu/viper#",
    "data": "http://lamp.cfar.umd.edu/viperdata#",
}

ABANDONMENT_EVENT_NAMES = {"PutObject", "AbandonedObject", "StolenObject"}


@dataclass
class GroundTruthEvent:
    object_id: str
    name: str  # "PutObject" | "AbandonedObject" | "StolenObject"
    start_frame: int
    end_frame: int
    bbox: Tuple[float, float, float, float]  # x1, y1, x2, y2, original video pixel space


@dataclass
class GroundTruthClip:
    xml_path: Path
    source_filename: str
    num_frames: Optional[int] = None
    frame_rate: Optional[float] = None
    frame_size: Optional[Tuple[int, int]] = None  # (width, height)
    events: List[GroundTruthEvent] = field(default_factory=list)

    def events_named(self, name: str) -> List[GroundTruthEvent]:
        return [e for e in self.events if e.name == name]


def load_ground_truth(xml_path: Path) -> GroundTruthClip:
    """Parses one AVSS2007/i-LIDS ViPER-GT XML annotation file."""
    xml_path = Path(xml_path)
    tree = ET.parse(xml_path)
    root = tree.getroot()

    sourcefile = root.find("viper:data/viper:sourcefile", VIPER_NS)
    if sourcefile is None:
        raise ValueError(f"No <sourcefile> element found in {xml_path}")
    source_filename = sourcefile.get("filename", "")

    num_frames = frame_rate = None
    frame_w = frame_h = None
    info_file = sourcefile.find("viper:file[@name='Information']", VIPER_NS)
    if info_file is not None:
        for attr in info_file.findall("viper:attribute", VIPER_NS):
            name = attr.get("name")
            dval = attr.find("data:dvalue", VIPER_NS)
            fval = attr.find("data:fvalue", VIPER_NS)
            if name == "NUMFRAMES" and dval is not None:
                num_frames = int(dval.get("value"))
            elif name == "FRAMERATE" and fval is not None:
                frame_rate = float(fval.get("value"))
            elif name == "H-FRAME-SIZE" and dval is not None:
                frame_w = int(dval.get("value"))
            elif name == "V-FRAME-SIZE" and dval is not None:
                frame_h = int(dval.get("value"))

    events: List[GroundTruthEvent] = []
    for obj in sourcefile.findall("viper:object", VIPER_NS):
        name = obj.get("name", "")
        if name not in ABANDONMENT_EVENT_NAMES:
            continue
        framespan = obj.get("framespan", "")
        try:
            start_s, end_s = framespan.split(":")
            start_frame, end_frame = int(start_s), int(end_s)
        except ValueError:
            continue

        bbox = (0.0, 0.0, 0.0, 0.0)
        bbox_attr = obj.find("viper:attribute[@name='BoundingBox']", VIPER_NS)
        if bbox_attr is not None:
            bbox_el = bbox_attr.find("data:bbox", VIPER_NS)
            if bbox_el is not None:
                x = float(bbox_el.get("x", 0))
                y = float(bbox_el.get("y", 0))
                w = float(bbox_el.get("width", 0))
                h = float(bbox_el.get("height", 0))
                bbox = (x, y, x + w, y + h)

        events.append(
            GroundTruthEvent(
                object_id=obj.get("id", ""),
                name=name,
                start_frame=start_frame,
                end_frame=end_frame,
                bbox=bbox,
            )
        )

    frame_size = (frame_w, frame_h) if frame_w and frame_h else None
    return GroundTruthClip(
        xml_path=xml_path,
        source_filename=source_filename,
        num_frames=num_frames,
        frame_rate=frame_rate,
        frame_size=frame_size,
        events=events,
    )


def evaluate_abandoned_object_detection(
    video_path: Path,
    gt: GroundTruthClip,
    config_path: str = "config.json",
    match_radius_px: float = 100.0,
) -> Dict[str, Any]:
    """Runs the live abandoned-object state machine over `video_path` and
    compares the system's ABANDONED-alert onset for each tracked bag against
    the ground truth's AbandonedObject events.

    Matching is spatial, not by a shared ID: the ground-truth bbox center (at
    the moment the system's tracked-bag entity first transitions to
    ABANDONED) is compared to that entity's own center, and the closest match
    within match_radius_px (model-input pixel space) is accepted. This is a
    deliberate simplification, not an oversight -- track identity is not
    guaranteed stable across a whole abandonment window (see Section IV-D /
    bugs_and_debugs.txt #29, #32-#36), so matching by position at the onset
    frame is more robust than requiring one ID to persist end-to-end.

    Reports, per ground-truth AbandonedObject event: hit/miss, detection
    delay (system onset frame minus ground-truth onset frame), and delay in
    seconds using the clip's own frame rate. Also reports false alarms: bag
    entities the system alerted ABANDONED on that don't spatially match any
    ground-truth AbandonedObject event.
    """
    import json

    import cv2
    from ultralytics import YOLO

    from src.detection import letterbox_resize
    from src.state_machine import reset_state, tracked_bags
    from src.tracking import process_tracking_state_machine

    config_data: Dict[str, Any] = {}
    resolved_config = ROOT / config_path
    if resolved_config.is_file():
        with resolved_config.open("r", encoding="utf-8") as f:
            config_data = json.load(f)

    imgsz = config_data.get("detection", {}).get("imgsz", 640)
    confidence = config_data.get("detection", {}).get("confidence", 0.35)
    device = config_data.get("detection", {}).get("device", "cpu")
    tracker = config_data.get("detection", {}).get("tracker", "bytetrack.yaml")
    half = config_data.get("detection", {}).get("half", False)
    model_path = config_data.get("detection", {}).get("model_path", "yolov8n.pt")
    state_machine_config = dict(config_data.get("state_machine", {}))
    state_machine_config["imgsz"] = imgsz
    th_frames = state_machine_config.get("th_frames", 10)

    reset_state()
    model = YOLO(model_path)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open video: {video_path}")

    src_w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    src_h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    scale = min(imgsz / src_w, imgsz / src_h)
    pad_x = (imgsz - src_w * scale) / 2.0
    pad_y = (imgsz - src_h * scale) / 2.0

    abandoned_events = gt.events_named("AbandonedObject")
    last_event_frame = max((e.end_frame for e in gt.events), default=(gt.num_frames or 10**9))

    first_abandoned_frame: Dict[int, int] = {}
    first_abandoned_center: Dict[int, Tuple[float, float]] = {}

    frame_idx = 0
    while cap.isOpened() and frame_idx <= last_event_frame:
        ret, frame = cap.read()
        if not ret or frame is None:
            break

        frame_resized = letterbox_resize(frame, imgsz)
        results = model.track(
            frame_resized,
            persist=True,
            tracker=tracker,
            conf=confidence,
            imgsz=imgsz,
            device=device,
            half=half,
            classes=[0, 24, 26, 28],
            verbose=False,
        )
        process_tracking_state_machine(
            results,
            frame_idx,
            polygon_vertices=[],
            th_frames=th_frames,
            state_machine_config=state_machine_config,
        )

        for bag_id, bag in tracked_bags.items():
            if bag.get("state") == "ABANDONED" and bag_id not in first_abandoned_frame:
                first_abandoned_frame[bag_id] = frame_idx
                first_abandoned_center[bag_id] = bag.get("center_coords", (0.0, 0.0))

        frame_idx += 1

    cap.release()

    def gt_bbox_to_model_space(bbox: Tuple[float, float, float, float]) -> Tuple[float, float]:
        x1, y1, x2, y2 = bbox
        cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        return cx * scale + pad_x, cy * scale + pad_y

    event_results: List[Dict[str, Any]] = []
    matched_bag_ids = set()
    for event in abandoned_events:
        gt_center = gt_bbox_to_model_space(event.bbox)
        best_bag_id, best_dist = None, None
        for bag_id, center in first_abandoned_center.items():
            dist = ((center[0] - gt_center[0]) ** 2 + (center[1] - gt_center[1]) ** 2) ** 0.5
            if best_dist is None or dist < best_dist:
                best_bag_id, best_dist = bag_id, dist

        if best_bag_id is not None and best_dist is not None and best_dist <= match_radius_px:
            onset_frame = first_abandoned_frame[best_bag_id]
            delay_frames = onset_frame - event.start_frame
            event_results.append({
                "gt_object_id": event.object_id,
                "gt_onset_frame": event.start_frame,
                "matched_bag_id": best_bag_id,
                "match_distance_px": best_dist,
                "system_onset_frame": onset_frame,
                "delay_frames": delay_frames,
                "delay_seconds": (delay_frames / gt.frame_rate) if gt.frame_rate else None,
                "hit": True,
            })
            matched_bag_ids.add(best_bag_id)
        else:
            event_results.append({
                "gt_object_id": event.object_id,
                "gt_onset_frame": event.start_frame,
                "matched_bag_id": None,
                "match_distance_px": best_dist,
                "system_onset_frame": None,
                "delay_frames": None,
                "delay_seconds": None,
                "hit": False,
            })

    false_alarm_bag_ids = [bid for bid in first_abandoned_frame if bid not in matched_bag_ids]

    return {
        "clip": gt.source_filename,
        "frames_processed": frame_idx,
        "num_gt_events": len(abandoned_events),
        "num_hits": sum(1 for r in event_results if r["hit"]),
        "num_misses": sum(1 for r in event_results if not r["hit"]),
        "num_false_alarms": len(false_alarm_bag_ids),
        "events": event_results,
    }


def _discover_avss_clips(avss_dir: Path) -> List[Tuple[Path, Path]]:
    """Pairs each AVSS *.txt ground-truth file with its same-stem video file."""
    pairs = []
    for xml_file in sorted(avss_dir.glob("*.txt")):
        for ext in (".mpg", ".mp4", ".avi", ".mkv", ".mov"):
            candidate = xml_file.with_suffix(ext)
            if candidate.is_file():
                pairs.append((candidate, xml_file))
                break
    return pairs


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Event-level evaluation of abandoned-object detection against AVSS 2007 ground truth"
    )
    parser.add_argument("--video", type=str, default=None, help="Path to a single AVSS clip (omit to run all EASY/MEDIUM/HARD clips)")
    parser.add_argument("--ground-truth", type=str, default=None, help="Path to the matching ViPER-GT .txt file (required if --video is set)")
    parser.add_argument("--avss-dir", type=str, default="data/raw/abandoned_objects/AVSS 2007",
                         help="Directory to auto-discover clip/ground-truth pairs from when --video is omitted")
    parser.add_argument("--config", type=str, default="config.json")
    parser.add_argument("--match-radius", type=float, default=100.0, help="Max spatial match distance in model-input pixels")
    args = parser.parse_args()

    if args.video:
        if not args.ground_truth:
            parser.error("--ground-truth is required when --video is set")
        pairs = [(ROOT / args.video, ROOT / args.ground_truth)]
    else:
        pairs = _discover_avss_clips(ROOT / args.avss_dir)
        if not pairs:
            parser.error(f"No clip/ground-truth pairs found under {args.avss_dir}")

    all_results = []
    for video_path, xml_path in pairs:
        print(f"\n=== {xml_path.name} ===")
        gt = load_ground_truth(xml_path)
        print(f"  Source: {gt.source_filename}, {gt.num_frames} frames @ {gt.frame_rate} fps, "
              f"{len(gt.events_named('AbandonedObject'))} AbandonedObject event(s)")
        result = evaluate_abandoned_object_detection(video_path, gt, config_path=args.config, match_radius_px=args.match_radius)
        all_results.append(result)
        print(f"  Hits: {result['num_hits']}/{result['num_gt_events']}  "
              f"Misses: {result['num_misses']}  False alarms: {result['num_false_alarms']}")
        for ev in result["events"]:
            if ev["hit"]:
                print(f"    gt_object {ev['gt_object_id']}: detected, delay = {ev['delay_frames']} frames "
                      f"({ev['delay_seconds']:.2f}s)" if ev["delay_seconds"] is not None else "")
            else:
                print(f"    gt_object {ev['gt_object_id']}: MISSED (no matching ABANDONED alert within {args.match_radius}px)")

    total_events = sum(r["num_gt_events"] for r in all_results)
    total_hits = sum(r["num_hits"] for r in all_results)
    total_false_alarms = sum(r["num_false_alarms"] for r in all_results)
    print(f"\n=== Aggregate across {len(all_results)} clip(s) ===")
    print(f"  Hit rate: {total_hits}/{total_events}"
          + (f" ({100.0 * total_hits / total_events:.1f}%)" if total_events else ""))
    print(f"  Total false alarms: {total_false_alarms}")


if __name__ == "__main__":
    main()
