import math
from typing import Any, Dict, List, Optional, Tuple

from .state_machine import (
    behavioral_smoothing_buffer,
    tracked_bags,
    tracked_people,
)
from .engineering_layers import (
    calculate_normalized_kinematics,
    is_inside_polygon,
    process_abandoned_logic,
    update_behavioral_filter,
    verify_panic_anomaly,
)


def update_tracking_states(yolo_results: Any, polygon_vertices: List[Tuple[float, float]], th_frames: int = 150):
    """
    Main orchestration function called every frame by src/detection.py.
    Parses active frame detections and updates our localized memory buffers.
    """
    if yolo_results is None:
        return

    detections = []
    for result in yolo_results:
        boxes = getattr(result.boxes, "xywh", None)
        confs = getattr(result.boxes, "conf", None)
        labels = getattr(result.boxes, "cls", None)
        if boxes is None:
            continue

        for index, box in enumerate(boxes):
            x_center, y_center, width, height = map(float, box)
            confidence = float(confs[index]) if confs is not None and len(confs) > index else 0.0
            label = int(labels[index]) if labels is not None and len(labels) > index else -1
            detections.append(
                {
                    "person_id": index,
                    "x_center": x_center,
                    "y_center": y_center,
                    "width": width,
                    "height": height,
                    "confidence": confidence,
                    "label": label,
                }
            )

    for detection in detections:
        person_id = detection["person_id"]
        x_center = detection["x_center"]
        y_center = detection["y_center"]
        height = detection["height"]
        confidence = detection["confidence"]

        if person_id not in tracked_people:
            tracked_people[person_id] = {
                "path_history": [],
                "velocities": [],
                "state": "NORMAL",
                "panic_counter": 0,
            }

        foot_x = x_center
        foot_y = y_center + (height / 2.0)

        if is_inside_polygon(foot_x, foot_y, polygon_vertices):
            tracked_people[person_id]["state"] = "PANIC"
            tracked_people[person_id]["panic_counter"] += 1
        else:
            tracked_people[person_id]["state"] = "NORMAL"

        tracked_people[person_id]["path_history"].append((foot_x, foot_y))
        if len(tracked_people[person_id]["path_history"]) > 30:
            tracked_people[person_id]["path_history"].pop(0)

        previous_box = None
        if tracked_people[person_id].get("previous_box") is not None:
            previous_box = tracked_people[person_id]["previous_box"]

        current_box = (foot_x, foot_y, max(height, 1.0))
        if previous_box is not None:
            normalized_velocity = calculate_normalized_kinematics(current_box, previous_box)
            tracked_people[person_id]["velocities"].append(normalized_velocity)
            if len(tracked_people[person_id]["velocities"]) > 15:
                tracked_people[person_id]["velocities"].pop(0)
            if verify_panic_anomaly(tracked_people[person_id]["velocities"]):
                tracked_people[person_id]["state"] = "PANIC"
                tracked_people[person_id]["panic_counter"] += 1

        tracked_people[person_id]["previous_box"] = current_box
        update_behavioral_filter(person_id, confidence)

    bag_candidates = [d for d in detections if d["label"] == 0]
    for bag in bag_candidates:
        bag_id = int(bag["person_id"])
        if bag_id not in tracked_bags:
            tracked_bags[bag_id] = {
                "owner_id": None,
                "center_coords": (bag["x_center"], bag["y_center"]),
                "state": "ATTENDED",
                "timer": 0,
            }

        person_coords = None
        for detection in detections:
            if detection["label"] != 0:
                person_coords = (detection["x_center"], detection["y_center"])
                break

        if person_coords is not None:
            process_abandoned_logic(bag_id, tracked_bags[bag_id], person_coords, th_frames)


__all__ = ["update_tracking_states"]
