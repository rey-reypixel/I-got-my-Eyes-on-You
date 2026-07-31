import math
from collections import deque
from typing import Any, Dict, List, Optional, Tuple

from .state_machine import (
    behavioral_smoothing_buffer,
    recently_evicted_bags,
    tracked_bags,
    tracked_people,
    behavioral_smoothing_buffer_lock,
    recently_evicted_bags_lock,
    tracked_bags_lock,
    tracked_people_lock,
)
from .engineering_layers import (
    calculate_normalized_kinematics,
    is_inside_polygon,
    process_abandoned_logic,
    verify_panic_anomaly,
)


def resolve_person_state(person_state: dict) -> str:
    """Derive the single displayed state label from independent per-detector flags.

    Fixed priority order: an unauthorized-access alert must never be silently
    dropped because a kinematic spike or limb-motion flag fired the same frame.
    """
    if person_state.get("intruder_flag"):
        return "INTRUDER"
    if person_state.get("panic_flag"):
        return "PANIC"
    if person_state.get("kinematic_suspicious_flag") or person_state.get("limb_suspicious_flag"):
        return "SUSPICIOUS_ACTIVITY"
    return "NORMAL"


def process_behavioral_smoothing(track_id: int, current_raw_score: float, config: dict) -> None:
    """Smooth a real behavioral/limb-motion anomaly score (0..1) via EMA and set
    the limb_suspicious_flag. NOTE: this must be fed an actual behavior signal
    (e.g. pose-based limb-motion variance), never YOLO detection confidence --
    confidence measures how clearly a person was detected, not what they're doing.
    """
    # 1. Retrieve the instance context dictionary
    track_meta = tracked_people[track_id]

    # Safely get window config whether config is root or state_machine sub-dict
    if "state_machine" in config:
        window_cfg = config["state_machine"]["behavioral_smoothing"]
    elif "behavioral_smoothing" in config:
        window_cfg = config["behavioral_smoothing"]
    else:
        window_cfg = {"window_size": 15, "activation_threshold": 0.75, "beta": 0.85}

    beta = window_cfg.get("beta", 0.85)

    # 2. Update sliding history buffer (retaining window for compatibility if needed elsewhere)
    if "anomaly_score_window" in track_meta:
        track_meta["anomaly_score_window"].append(current_raw_score)

    # 3. Compute Exponential Moving Average (EMA)
    prev_ema = track_meta.get("anomaly_score_ema", None)
    if prev_ema is None:
        current_ema = current_raw_score
    else:
        current_ema = beta * prev_ema + (1.0 - beta) * current_raw_score

    track_meta["anomaly_score_ema"] = current_ema

    # 4. Apply strict activation transition constraints using the running EMA
    track_meta["limb_suspicious_flag"] = current_ema >= window_cfg["activation_threshold"]
    track_meta["state"] = resolve_person_state(track_meta)


# COCO-17 keypoint layout used by yolov8*-pose models.
LIMB_KEYPOINT_INDICES = {
    "left_shoulder": 5,
    "right_shoulder": 6,
    "left_elbow": 7,
    "right_elbow": 8,
    "left_wrist": 9,
    "right_wrist": 10,
    "left_hip": 11,
    "right_hip": 12,
}
TORSO_KEYPOINTS = ("left_shoulder", "right_shoulder", "left_hip", "right_hip")
LIMB_MOTION_KEYPOINTS = ("left_elbow", "right_elbow", "left_wrist", "right_wrist")


def get_pose_gated_person_ids(state_machine_config: Optional[dict] = None) -> List[int]:
    """Return the track IDs currently eligible for the (expensive) pose-based
    limb-motion check: people whose centroid has stayed near-stationary for
    long enough that kinematic tracking alone cannot tell idle from a
    stationary altercation (fighting, vandalism, shoplifting).
    """
    limb_cfg = (state_machine_config or {}).get("limb_motion", {})
    gate_frames = limb_cfg.get("stationary_frames_gate", 12)
    return [
        pid
        for pid, meta in tracked_people.items()
        # stale_counter == 0 means this person was actively redetected in the
        # current frame. Without this check, a person's frozen stationary_counter
        # from before they stopped being detected would keep them "eligible"
        # indefinitely (until full eviction), even though there's no one left
        # to actually run the pose check on.
        if meta.get("stationary_counter", 0) >= gate_frames and meta.get("stale_counter", 0) == 0
    ]


def compute_limb_motion_score(
    person_id: int,
    keypoints_xy: Any,
    keypoints_conf: Any,
    norm_h: float,
    state_machine_config: Optional[dict] = None,
) -> Optional[float]:
    """Turn one frame's pose keypoints into a rolling torso-relative limb-motion
    variance score in roughly [0, 1], or None if this frame's keypoints aren't
    reliable enough to use (mirrors the confidence-gating already applied to
    centroid kinematics, to avoid reintroducing joint-level jitter as false
    positives).
    """
    state_machine_config = state_machine_config or {}
    limb_cfg = state_machine_config.get("limb_motion", {})
    min_conf = limb_cfg.get("keypoint_min_confidence", 0.5)
    min_kps = limb_cfg.get("min_keypoints_required", 2)
    window_size = state_machine_config.get("behavioral_smoothing", {}).get("window_size", 10)
    variance_scale = limb_cfg.get("variance_scale", 0.02)

    def _kp(name: str):
        idx = LIMB_KEYPOINT_INDICES[name]
        x, y = float(keypoints_xy[idx][0]), float(keypoints_xy[idx][1])
        conf = float(keypoints_conf[idx])
        return x, y, conf

    torso_points = []
    for name in TORSO_KEYPOINTS:
        x, y, conf = _kp(name)
        if conf >= min_conf:
            torso_points.append((x, y))
    if len(torso_points) < 2:
        return None  # torso reference is unreliable this frame

    torso_x = sum(p[0] for p in torso_points) / len(torso_points)
    torso_y = sum(p[1] for p in torso_points) / len(torso_points)

    offsets = []
    for name in LIMB_MOTION_KEYPOINTS:
        x, y, conf = _kp(name)
        if conf >= min_conf:
            offsets.append(((x - torso_x) / norm_h, (y - torso_y) / norm_h))
    if len(offsets) < min_kps:
        return None  # not enough reliable limb keypoints this frame

    person_state = tracked_people.get(person_id)
    if person_state is None:
        return None

    if "limb_offset_history" not in person_state:
        person_state["limb_offset_history"] = deque(maxlen=window_size)
    person_state["limb_offset_history"].append(offsets)

    history = person_state["limb_offset_history"]
    if len(history) < 3:
        return None  # need a few samples before variance is meaningful

    # Per-frame mean limb-to-torso distance, then variance of that across the window.
    magnitudes = [
        sum(math.sqrt(dx * dx + dy * dy) for dx, dy in frame_offsets) / len(frame_offsets)
        for frame_offsets in history
    ]
    mean_mag = sum(magnitudes) / len(magnitudes)
    variance = sum((m - mean_mag) ** 2 for m in magnitudes) / len(magnitudes)

    return min(1.0, variance / variance_scale)


def find_recoverable_bag_id(bx: float, by: float, active_bag_ids: set) -> Optional[Tuple[int, str]]:
    """Find a recently-vanished bag within recovery radius of (bx, by),
    preferring the closest one, searching both bags still sitting in
    `tracked_bags` (not currently active) and the short-lived
    `recently_evicted_bags` echo buffer (carried bags deleted outright on
    dropout -- see bugs_and_debugs.txt #29). Returns (source_id, "tracked" |
    "echo") or None. A bag still actively detected under its own ID this frame
    is never a candidate, even if it happens to be close by -- that's a
    genuinely distinct, still-alive bag, not something to merge.
    """
    recovered_id = None
    recovered_source = None
    recovered_dist = float("inf")

    for other_id, other_info in tracked_bags.items():
        if other_id in active_bag_ids:
            continue
        ox, oy = other_info["center_coords"]
        dist = math.sqrt((ox - bx) ** 2 + (oy - by) ** 2)
        if dist < 80.0 and dist < recovered_dist:
            recovered_id, recovered_source, recovered_dist = other_id, "tracked", dist

    with recently_evicted_bags_lock:
        for other_id, echo in recently_evicted_bags.items():
            ox, oy = echo["center_coords"]
            dist = math.sqrt((ox - bx) ** 2 + (oy - by) ** 2)
            if dist < 80.0 and dist < recovered_dist:
                recovered_id, recovered_source, recovered_dist = other_id, "echo", dist

    if recovered_id is None:
        return None
    return recovered_id, recovered_source


def update_tracking_states(
    yolo_results: Any,
    polygon_vertices: List[Tuple[float, float]],
    th_frames: int = 150,
    state_machine_config: Optional[dict] = None,
):
    """
    Main orchestration function called every frame by src/detection.py.
    Parses active frame detections and updates our localized memory buffers.
    """
    if yolo_results is None:
        return

    if state_machine_config is None:
        state_machine_config = {}

    th_frames = state_machine_config.get("th_frames", th_frames)
    panic_threshold_devs = state_machine_config.get("panic_threshold_devs", 3.0)
    bag_disconnect_dist = state_machine_config.get("bag_disconnect_dist", 50.0)
    bag_reconnect_dist = state_machine_config.get("bag_reconnect_dist", 30.0)
    state_max_stale_frames = state_machine_config.get("state_max_stale_frames", 900)
    kinematic_velocity_threshold = state_machine_config.get("kinematic_velocity_threshold", 0.05)
    min_velocity_history_frames = state_machine_config.get("min_velocity_history_frames", 5)
    panic_recovery_frames = state_machine_config.get("panic_recovery_frames", 30)
    bag_carried_dropout_grace_frames = state_machine_config.get("bag_carried_dropout_grace_frames", 5)
    bag_stationary_window = state_machine_config.get("bag_stationary_window", 6)
    bag_stationary_px = state_machine_config.get("bag_stationary_px", 10.0)

    detections = []
    for result in yolo_results:
        track_ids = getattr(result.boxes, "id", None)
        if track_ids is None:
            continue
        boxes = getattr(result.boxes, "xywh", None)
        confs = getattr(result.boxes, "conf", None)
        labels = getattr(result.boxes, "cls", None)
        names = getattr(result, "names", None)
        if boxes is None:
            continue

        for index, box in enumerate(boxes):
            x_center, y_center, width, height = map(float, box)
            confidence = float(confs[index]) if confs is not None and len(confs) > index else 0.0
            label = int(labels[index]) if labels is not None and len(labels) > index else -1
            class_label = str(label)
            if names is not None:
                try:
                    class_label = names.get(label, str(label))
                except AttributeError:
                    class_label = str(label)
            track_id = int(track_ids[index])
            detections.append(
                {
                    "person_id": track_id,
                    "x_center": x_center,
                    "y_center": y_center,
                    "width": width,
                    "height": height,
                    "confidence": confidence,
                    "label": label,
                    "class_label": class_label,
                }
            )

    COCO_PERSON_INDEX = 0
    COCO_BAG_INDICES = {24, 26, 28}  # backpack, handbag, suitcase

    bag_candidates = [
        d for d in detections 
        if d["class_label"] in {"backpack", "handbag", "suitcase"} or d["label"] in COCO_BAG_INDICES
    ]

    active_people_ids = set()
    active_bag_ids = set()

    for detection in detections:
        is_person = (detection["class_label"] == "person") or (detection["label"] == COCO_PERSON_INDEX)
        if not is_person:
            continue

        person_id = detection["person_id"]
        active_people_ids.add(person_id)
        x_center = detection["x_center"]
        y_center = detection["y_center"]
        width = detection["width"]
        height = detection["height"]

        if person_id not in tracked_people:
            tracked_people[person_id] = {
                "path_history": [],
                "velocities": [],
                "state": "NORMAL",
                "intruder_flag": False,
                "panic_flag": False,
                "panic_cooldown": 0,
                "kinematic_suspicious_flag": False,
                "limb_suspicious_flag": False,
                "stationary_counter": 0,
                "panic_counter": 0,
                "stale_counter": 0,
                "previous_center": None,
                "displacement_history": [],
                "max_observed_height": 0.0,
                "previous_box_raw": None,
                "last_known_center": None,
                "last_known_width": 40.0,
                "last_known_height": 100.0,
            }

        if "anomaly_score_window" not in tracked_people[person_id]:
            window_size = 15
            if state_machine_config and "behavioral_smoothing" in state_machine_config:
                window_size = state_machine_config["behavioral_smoothing"].get("window_size", 15)
            tracked_people[person_id]["anomaly_score_window"] = deque(maxlen=window_size)

        foot_x = x_center
        foot_y = y_center + (height / 2.0)

        # 1. Evaluate Spatial Boundary (Unauthorized Access)
        # Recomputed every frame -- this flag has no stickiness of its own, so
        # leaving/re-entering the ROI is always reflected immediately.
        tracked_people[person_id]["intruder_flag"] = is_inside_polygon(foot_x, foot_y, polygon_vertices)

        tracked_people[person_id]["path_history"].append((foot_x, foot_y))
        if len(tracked_people[person_id]["path_history"]) > 30:
            tracked_people[person_id]["path_history"].pop(0)

        # Update maximum observed height for this person
        prev_box_raw = tracked_people[person_id].get("previous_box_raw")
        tracked_people[person_id]["previous_box_raw"] = (x_center, y_center, height)

        # Last-known visual box, kept unconditionally (unlike the kinematic history
        # above, which gets reset on clipping/scale-jump suppression) so a brief
        # YOLO detection dropout can still render the person's last position
        # instead of the box vanishing outright (mirrors the bag persistence logic).
        tracked_people[person_id]["last_known_center"] = (x_center, y_center)
        tracked_people[person_id]["last_known_width"] = width
        tracked_people[person_id]["last_known_height"] = height

        max_h = max(tracked_people[person_id].get("max_observed_height", 0.0), height)
        tracked_people[person_id]["max_observed_height"] = max_h

        # Compute perspective-scaled height estimate to protect against edge cut-offs
        perspective_h = y_center * 0.45
        norm_h = max(max_h, perspective_h, 50.0)

        # Detect sudden height changes (occlusion, clipping, scale jumps)
        suppress_spike = False
        if prev_box_raw is not None:
            prev_h = prev_box_raw[2]
            if abs(height - prev_h) > (prev_h * 0.25):
                suppress_spike = True

        # Check if the bounding box is clipped by the image boundaries
        imgsz_val = state_machine_config.get("imgsz", 640)
        is_clipped = (
            (x_center - width / 2.0) <= 2.0 or
            (x_center + width / 2.0) >= (imgsz_val - 2.0) or
            (y_center - height / 2.0) <= 2.0 or
            (y_center + height / 2.0) >= (imgsz_val - 2.0)
        )

        panic_triggered = False
        if is_clipped or suppress_spike:
            # Reset motion state history to prevent spikes due to clipping / detection scale jumps
            tracked_people[person_id]["previous_center"] = None
            tracked_people[person_id]["previous_box"] = None
            tracked_people[person_id]["displacement_history"] = []
            tracked_people[person_id]["velocities"] = []
            # No reliable motion data this frame -- do not claim a kinematic anomaly off of it,
            # and don't count an unreliable frame towards the pose-gate stationary streak.
            tracked_people[person_id]["kinematic_suspicious_flag"] = False
            tracked_people[person_id]["stationary_counter"] = 0
        else:
            # 2. Evaluate Kinematic Spikes (Suspicious Activity / Panic)
            previous_box = tracked_people[person_id].get("previous_box")
            current_box = (x_center, y_center, norm_h)
            if previous_box is not None:
                normalized_velocity = calculate_normalized_kinematics(current_box, previous_box)
                tracked_people[person_id]["velocities"].append(normalized_velocity)
                if len(tracked_people[person_id]["velocities"]) > 15:
                    tracked_people[person_id]["velocities"].pop(0)

                if verify_panic_anomaly(tracked_people[person_id]["velocities"], threshold_devs=panic_threshold_devs):
                    panic_triggered = True
                    tracked_people[person_id]["panic_counter"] += 1

            tracked_people[person_id]["previous_box"] = current_box

            # 3. Evaluate Frame-to-Frame Centroid Displacement (Suspicious Activity)
            previous_center = tracked_people[person_id].get("previous_center")
            current_center = (x_center, y_center)
            tracked_people[person_id]["kinematic_suspicious_flag"] = False
            if previous_center is not None:
                dx = current_center[0] - previous_center[0]
                dy = current_center[1] - previous_center[1]
                euc_dist_pixel = math.sqrt(dx * dx + dy * dy)
                norm_disp = euc_dist_pixel / norm_h

                if "displacement_history" not in tracked_people[person_id]:
                    tracked_people[person_id]["displacement_history"] = []

                tracked_people[person_id]["displacement_history"].append(norm_disp)
                required_transitions = max(min_velocity_history_frames - 1, 1)
                if len(tracked_people[person_id]["displacement_history"]) > required_transitions:
                    tracked_people[person_id]["displacement_history"].pop(0)

                if len(tracked_people[person_id]["displacement_history"]) >= required_transitions:
                    avg_disp = sum(tracked_people[person_id]["displacement_history"]) / len(tracked_people[person_id]["displacement_history"])
                    # Recomputed every frame from the smoothed average -- no stickiness,
                    # so this recovers on its own once the person's motion settles down.
                    tracked_people[person_id]["kinematic_suspicious_flag"] = avg_disp > kinematic_velocity_threshold

                    # Track how long this person has been essentially motionless -- this is
                    # the blind spot centroid kinematics cannot see into (e.g. fighting in
                    # place), and gates the more expensive pose-based limb-motion check.
                    if tracked_people[person_id]["kinematic_suspicious_flag"]:
                        tracked_people[person_id]["stationary_counter"] = 0
                    else:
                        tracked_people[person_id]["stationary_counter"] = (
                            tracked_people[person_id].get("stationary_counter", 0) + 1
                        )
            tracked_people[person_id]["previous_center"] = current_center

        # Panic recovery: stays latched for panic_recovery_frames after the last
        # triggering frame, instead of either flickering every frame or sticking forever.
        if panic_triggered:
            tracked_people[person_id]["panic_flag"] = True
            tracked_people[person_id]["panic_cooldown"] = panic_recovery_frames
        elif tracked_people[person_id].get("panic_cooldown", 0) > 0:
            tracked_people[person_id]["panic_cooldown"] -= 1
            if tracked_people[person_id]["panic_cooldown"] <= 0:
                tracked_people[person_id]["panic_flag"] = False
        else:
            tracked_people[person_id]["panic_flag"] = False

        # Resolve the single displayed state label from the independent flags above.
        # limb_suspicious_flag is set separately (see process_behavioral_smoothing)
        # once it is fed a real behavior signal instead of detection confidence.
        tracked_people[person_id]["state"] = resolve_person_state(tracked_people[person_id])
    for bag in bag_candidates:
        bag_id = int(bag["person_id"])
        active_bag_ids.add(bag_id)

    # Process both actively detected bags and missing bags that are still tracked
    all_bag_ids_to_process = set(tracked_bags.keys()) | active_bag_ids

    for bag_id in all_bag_ids_to_process:
        # If the bag was evicted (popped) during duplicate suppression in this loop, skip it
        if bag_id not in active_bag_ids and bag_id not in tracked_bags:
            continue

        closest_person = None
        closest_dist = float("inf")
        person_height = 0.0
        px_closest, py_closest = 0.0, 0.0

        closest_person_id = None

        # Retrieve last known coordinates of the bag to calculate closest person
        if bag_id in active_bag_ids:
            bag_data = next((b for b in bag_candidates if int(b["person_id"]) == bag_id), None)
            if bag_data is None:
                continue
            bx = bag_data["x_center"]
            by = bag_data["y_center"]
        else:
            bx, by = tracked_bags[bag_id]["center_coords"]

        for detection in detections:
            is_person = (detection["class_label"] == "person") or (detection["label"] == COCO_PERSON_INDEX)
            if is_person:
                px = detection["x_center"]
                py = detection["y_center"]
                p_h = detection["height"]
                p_foot_x = px
                p_foot_y = py + (p_h / 2.0)
                
                # Proximity to either the person's feet (placed bags) or their centroid (carried bags)
                d_feet = math.sqrt((p_foot_x - bx) ** 2 + (p_foot_y - by) ** 2)
                d_centroid = math.sqrt((px - bx) ** 2 + (py - by) ** 2)
                d = min(d_feet, d_centroid)
                
                if d < closest_dist:
                    closest_dist = d
                    # Pass the coordinate that is closest to the bag center
                    closest_person = (p_foot_x, p_foot_y) if d_feet < d_centroid else (px, py)
                    person_height = p_h
                    px_closest, py_closest = px, py
                    closest_person_id = detection["person_id"]

        # Update bag coordinates based on active/dropout status
        if bag_id in active_bag_ids:
            bag_data = next((b for b in bag_candidates if int(b["person_id"]) == bag_id), None)
            bx = bag_data["x_center"]
            by = bag_data["y_center"]
            bw = bag_data["width"]
            bh = bag_data["height"]
            
            # Check if carried: vertical distance to centroid vs feet -- but only
            # for a person actually within attending range of the bag. Without this
            # gate, whichever person is globally nearest (even someone far across
            # the frame, well outside any plausible carrying distance) still gets
            # run through the vertical check, and incidental perspective alignment
            # (a bag placed farther back can sit level with a nearby person's torso
            # rather than their feet) can flip is_carried to True for a bag nobody
            # is touching. That misfire is fatal down the line: a dropout on a
            # bag believed "carried" is evicted outright with no persisted box
            # (see bugs_and_debugs.txt #29's follow-up note on this exact suspicion).
            is_carried = False
            if closest_person is not None:
                max_carry_dist = max(bag_disconnect_dist, person_height * 0.45)
                if closest_dist <= max_carry_dist:
                    dy_centroid = abs(py_closest - by)
                    dy_feet = abs((py_closest + person_height / 2.0) - by)
                    if dy_centroid <= dy_feet:
                        is_carried = True

            # Motion-stability override: a bag riding along with a moving person
            # displaces frame to frame; a bag resting on the floor, even one a
            # person is standing right next to (passing the proximity gate above),
            # does not. Reported from real footage (AVSS 2007 AVSSS07_EASY.mpg):
            # without this, a bag that's actually just sitting there near someone
            # can stay misclassified "carried" purely from vertical/perspective
            # coincidence, which is fatal down the line -- a "carried" bag's
            # dropout is treated with much less patience than a placed one's (see
            # bugs_and_debugs.txt #33/#35). Track recent active-detection positions
            # and require net displacement over that window to exceed a small
            # threshold before believing "carried" at all; a bag that hasn't
            # actually gone anywhere overrides straight to "placed" regardless of
            # what the vertical heuristic said this frame.
            position_history = (
                tracked_bags[bag_id].setdefault("position_history", deque(maxlen=bag_stationary_window))
                if bag_id in tracked_bags
                else deque(maxlen=bag_stationary_window)
            )
            position_history.append((bx, by))
            if is_carried and len(position_history) >= bag_stationary_window:
                oldest_x, oldest_y = position_history[0]
                net_displacement = math.hypot(bx - oldest_x, by - oldest_y)
                if net_displacement <= bag_stationary_px:
                    is_carried = False

            if bag_id not in tracked_bags:
                # Recover accumulated state (state/timer) from a recently-vanished
                # bag at (nearly) the same position instead of resetting to
                # ATTENDED/timer=0. A bag is stationary, so a brand-new track ID
                # appearing within a small radius of where one just vanished is
                # overwhelmingly likely to be the *same* physical bag reacquired
                # under a new ID (ByteTrack has no Re-ID/appearance matching --
                # see bugs_and_debugs.txt), not a coincidental second bag. This is
                # deliberately not applied to people: two different individuals
                # could plausibly cross the same spot, which would make the same
                # heuristic unsafe there.
                recovery = find_recoverable_bag_id(bx, by, active_bag_ids)

                if recovery is not None:
                    recovered_id, recovered_source = recovery
                    if recovered_source == "tracked":
                        recovered = tracked_bags.pop(recovered_id)
                    else:
                        with recently_evicted_bags_lock:
                            recovered = recently_evicted_bags.pop(recovered_id)
                    tracked_bags[bag_id] = {
                        "owner_id": closest_person_id if closest_person_id is not None else recovered.get("owner_id"),
                        "center_coords": (bx, by),
                        "width": bw,
                        "height": bh,
                        "state": recovered["state"],
                        "timer": recovered["timer"],
                        "stale_counter": 0,
                        "is_carried": is_carried,
                        "carried_miss_streak": 0,
                        "position_history": position_history,
                    }
                else:
                    tracked_bags[bag_id] = {
                        "owner_id": closest_person_id,
                        "center_coords": (bx, by),
                        "width": bw,
                        "height": bh,
                        "state": "ATTENDED",
                        "timer": 0,
                        "stale_counter": 0,
                        "is_carried": is_carried,
                        "carried_miss_streak": 0,
                        "position_history": position_history,
                    }
            else:
                tracked_bags[bag_id]["center_coords"] = (bx, by)
                tracked_bags[bag_id]["width"] = bw
                tracked_bags[bag_id]["height"] = bh
                tracked_bags[bag_id]["is_carried"] = is_carried
                tracked_bags[bag_id]["carried_miss_streak"] = 0
                if closest_person_id is not None:
                    tracked_bags[bag_id]["owner_id"] = closest_person_id
        else:
            # Missing bag (dropout)
            if tracked_bags[bag_id].get("is_carried", False):
                # A bag believed carried gets a short grace window (config
                # bag_carried_dropout_grace_frames, default 5) before eviction,
                # instead of zero tolerance. This exists because the carried ->
                # placed transition itself routinely causes a missed detection: a
                # bag lying flat looks nothing like one riding on a shoulder, and the
                # drop motion tends to blur the frame(s) it happens on. Zero grace
                # meant that ordinary transition miss deleted a just-set-down bag
                # outright, with no persisted box ever shown -- it wasn't abandoned
                # quietly, it was un-rendered quietly (bugs_and_debugs.txt #32, #33).
                #
                # This window was briefly raised all the way to 900 (#34/#35) to
                # bridge a real ~649-frame (~21.6s) YOLO dropout found on ABODA
                # video1.avi -- but that bag was actually stationary/placed the
                # whole time, just misclassified "carried" by vertical/perspective
                # coincidence. The motion-stability check above (bag_stationary_window
                # / bag_stationary_px) now catches that misclassification directly,
                # so a genuinely-stationary bag is routed through the *placed* path
                # (state_max_stale_frames / bag_persist_frames, still 900) instead of
                # this one. That freed this grace window to go back to being short:
                # a bag that really is still being carried, moving with a live
                # person, needs only a couple of frames to bridge an ordinary
                # detection blip -- it does not need hundreds. process_abandoned_logic
                # still runs below using the frozen position every grace frame, so a
                # genuinely-departing owner keeps accumulating WARNING time regardless
                # of whether detection has resumed, and once the grace budget is
                # exhausted this still falls back to the original instant eviction.
                #
                # Reported from real footage (AVSS 2007 AVSSS07_EASY.mpg): a person
                # who genuinely carried a bag out of frame with them left behind a
                # long-lived, false "ABANDONED (PERSISTED)" ghost box under the old
                # 900-frame grace -- exactly the #20/#29 ghost-duplicate risk that
                # window's size had reopened, now observed directly rather than just
                # predicted. Shrinking the grace back down plus the stability check
                # above fixes both real-world patterns at once: a truly-placed bag
                # (ABODA) gets the long window via the *right* path, and a truly-
                # carried bag that leaves with its owner (AVSS) is no longer kept
                # alive as a stale ghost for minutes after it's actually gone.
                miss_streak = tracked_bags[bag_id].get("carried_miss_streak", 0) + 1
                tracked_bags[bag_id]["carried_miss_streak"] = miss_streak
                if miss_streak > bag_carried_dropout_grace_frames:
                    # Grace exhausted: evict to prevent ghost bounding boxes and false
                    # alarms (a stale, frozen ghost position could otherwise fall behind
                    # a bag that's still actually with its owner, just undetected, and
                    # wrongly accumulate towards WARNING/ABANDONED). Stash a short-lived
                    # echo of its accumulated state first, so that if the same physical
                    # bag is reacquired nearby moments later (very likely, since bags are
                    # stationary -- see bugs_and_debugs.txt #29), it doesn't have to start
                    # over from ATTENDED/timer=0.
                    evicted = tracked_bags.pop(bag_id)
                    with recently_evicted_bags_lock:
                        recently_evicted_bags[bag_id] = {
                            "center_coords": evicted["center_coords"],
                            "state": evicted["state"],
                            "timer": evicted["timer"],
                            "owner_id": evicted.get("owner_id"),
                            "ttl": 30,
                        }
                    continue
                # Still within the grace window: fall through and hold the last-known
                # position/state, same as a non-carried bag's dropout below.

            bx, by = tracked_bags[bag_id]["center_coords"]
            bw = tracked_bags[bag_id].get("width", 40.0)
            bh = tracked_bags[bag_id].get("height", 40.0)

        # Scale disconnect/reconnect thresholds based on proximity
        # We use 0.45 and 0.30 as multipliers representing a close body scale limit
        d_disc = max(bag_disconnect_dist, person_height * 0.45)
        d_rec = max(bag_reconnect_dist, person_height * 0.30)

        process_abandoned_logic(
            bag_id,
            tracked_bags[bag_id],
            closest_person,
            th_frames,
            d_disconnect=d_disc,
            d_reconnect=d_rec,
        )

    # 3. Evict stale track IDs to prevent memory leaks (OOM)
    with tracked_people_lock:
        to_evict_people = []
        for pid in list(tracked_people.keys()):
            if pid not in active_people_ids:
                tracked_people[pid]["stale_counter"] = tracked_people[pid].get("stale_counter", 0) + 1
                if tracked_people[pid]["stale_counter"] > state_max_stale_frames:
                    to_evict_people.append(pid)
            else:
                tracked_people[pid]["stale_counter"] = 0
        for pid in to_evict_people:
            tracked_people.pop(pid, None)
            with behavioral_smoothing_buffer_lock:
                behavioral_smoothing_buffer.pop(pid, None)

    with tracked_bags_lock:
        to_evict_bags = []
        for bid in list(tracked_bags.keys()):
            if bid not in active_bag_ids:
                tracked_bags[bid]["stale_counter"] = tracked_bags[bid].get("stale_counter", 0) + 1
                if tracked_bags[bid]["stale_counter"] > state_max_stale_frames:
                    to_evict_bags.append(bid)
            else:
                tracked_bags[bid]["stale_counter"] = 0

        # Also find and remove duplicate/overlapping bags (keep only the newest track ID)
        all_bids = sorted(list(tracked_bags.keys()))
        for i in range(len(all_bids)):
            bid1 = all_bids[i]
            if bid1 in to_evict_bags or bid1 not in tracked_bags:
                continue
            for j in range(i + 1, len(all_bids)):
                bid2 = all_bids[j]
                if bid2 in to_evict_bags or bid2 not in tracked_bags:
                    continue
                # Calculate distance between centers
                c1 = tracked_bags[bid1]["center_coords"]
                c2 = tracked_bags[bid2]["center_coords"]
                dist = math.sqrt((c1[0] - c2[0]) ** 2 + (c1[1] - c2[1]) ** 2)
                # If they are within 80 pixels of each other, evict the older one (bid1)
                if dist < 80.0:
                    to_evict_bags.append(bid1)
                    break

        for bid in to_evict_bags:
            tracked_bags.pop(bid, None)

    # Decay the recently-evicted-carried-bag echo buffer: it exists to bridge
    # a short reacquisition gap, not to persist indefinitely.
    with recently_evicted_bags_lock:
        expired = []
        for bid, echo in recently_evicted_bags.items():
            echo["ttl"] -= 1
            if echo["ttl"] <= 0:
                expired.append(bid)
        for bid in expired:
            recently_evicted_bags.pop(bid, None)


def process_tracking_state_machine(
    yolo_results: Any,
    frame_id: int,
    polygon_vertices: Optional[List[Tuple[float, float]]] = None,
    th_frames: int = 10,
    state_machine_config: Optional[dict] = None,
) -> List[Dict[str, Any]]:
    """Return a list of metric dictionaries for benchmark logging and downstream analytics."""
    if polygon_vertices is None:
        polygon_vertices = []

    update_tracking_states(
        yolo_results,
        polygon_vertices,
        th_frames=th_frames,
        state_machine_config=state_machine_config,
    )

    custom_metrics: List[Dict[str, Any]] = []
    if yolo_results is None:
        return custom_metrics

    COCO_PERSON_INDEX = 0
    COCO_BAG_INDICES = {24, 26, 28}

    for result in yolo_results:
        track_ids = getattr(result.boxes, "id", None)
        if track_ids is None:
            continue
        boxes = getattr(result.boxes, "xywh", None)
        labels = getattr(result.boxes, "cls", None)
        if boxes is None:
            continue

        names = getattr(result, "names", None)
        for index, _box in enumerate(boxes):
            track_id = int(track_ids[index])
            label_index = int(labels[index]) if labels is not None and len(labels) > index else -1
            class_label = str(label_index)
            if names is not None:
                try:
                    class_label = names.get(label_index, str(label_index))  # type: ignore[union-attr]
                except AttributeError:
                    class_label = str(label_index)

            is_person = (class_label == "person") or (label_index == COCO_PERSON_INDEX)
            is_bag = (class_label in {"backpack", "handbag", "suitcase"}) or (label_index in COCO_BAG_INDICES)

            state = "NORMAL"
            owner_id = None
            if is_person:
                state = tracked_people.get(track_id, {}).get("state", "NORMAL") if track_id in tracked_people else "NORMAL"
            elif is_bag:
                state = tracked_bags.get(track_id, {}).get("state", "ATTENDED") if track_id in tracked_bags else "ATTENDED"
                if track_id in tracked_bags:
                    owner_id = tracked_bags[track_id].get("owner_id")

            custom_metrics.append(
                {
                    "frame_id": frame_id,
                    "object_id": track_id,
                    "class_label": class_label,
                    "assigned_owner_id": owner_id,
                    "current_state": state,
                    "threat_alert_triggered": state in {"PANIC", "WARNING", "ABANDONED", "INTRUDER", "SUSPICIOUS_ACTIVITY"},
                }
            )

    return custom_metrics


__all__ = [
    "update_tracking_states",
    "process_tracking_state_machine",
    "process_behavioral_smoothing",
    "resolve_person_state",
    "get_pose_gated_person_ids",
    "compute_limb_motion_score",
    "find_recoverable_bag_id",
]
