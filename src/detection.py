"""Primary multi-model synchronization loop for live video analysis."""

from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Any, Optional

try:
    import cv2
except ImportError:  # pragma: no cover - depends on runtime environment
    cv2 = None

try:
    from ultralytics import YOLO
except ImportError:  # pragma: no cover - depends on runtime environment
    YOLO = None

from .state_machine import (
    tracked_bags,
    tracked_people,
)
from .tracking import update_tracking_states


def is_inside_polygon(x_person: float, y_person: float, polygon_vertices: list) -> bool:
    """Evaluate whether a tracked foot coordinate falls inside a slanted ROI polygon."""
    intersect_count = 0
    num_vertices = len(polygon_vertices)

    for i in range(num_vertices):
        A = polygon_vertices[i]
        B = polygon_vertices[(i + 1) % num_vertices]

        if (A[1] <= y_person < B[1]) or (B[1] <= y_person < A[1]):
            if x_person < min(A[0], B[0]):
                intersect_count += 1
            elif x_person > max(A[0], B[0]):
                continue
            else:
                x_intersection = A[0] + ((y_person - A[1]) * (B[0] - A[0])) / (B[1] - A[1])
                if x_person <= x_intersection:
                    intersect_count += 1

    return intersect_count % 2 != 0


def process_abandoned_logic(bag_id: int, bag_state: dict, person_coords: tuple, th_frames: int):
    """Manage temporal abandoned-object state transitions with proximity hysteresis."""
    D_DISCONNECT = 50.0
    D_RECONNECT = 30.0

    d = math.sqrt((person_coords[0] - bag_state["center_coords"][0]) ** 2 + (person_coords[1] - bag_state["center_coords"][1]) ** 2)

    if bag_state["state"] == "ATTENDED":
        if d > D_DISCONNECT:
            bag_state["state"] = "WARNING"
            bag_state["timer"] = 1
    elif bag_state["state"] in ("WARNING", "ABANDONED"):
        if d < D_RECONNECT:
            bag_state["state"] = "ATTENDED"
            bag_state["timer"] = 0
        elif bag_state["state"] == "WARNING":
            bag_state["timer"] += 1
            if bag_state["timer"] >= th_frames:
                bag_state["state"] = "ABANDONED"


def calculate_normalized_kinematics(current_box: tuple, previous_box: tuple) -> float:
    """Compute invariant body-height normalized motion for perspective stabilization."""
    x_t, y_t, h_t = current_box
    x_prev, y_prev, _ = previous_box

    pixel_displacement = math.sqrt((x_t - x_prev) ** 2 + (y_t - y_prev) ** 2)
    return pixel_displacement / h_t


def verify_panic_anomaly(velocity_history: list, threshold_devs: float = 3.0) -> bool:
    """Detect acceleration spikes from a rolling velocity window."""
    if len(velocity_history) < 5:
        return False

    v_current = velocity_history[-1]
    v_historical = velocity_history[:-1]
    mean_v = sum(v_historical) / len(v_historical)
    if v_current > (mean_v * threshold_devs):
        return True
    return False


class MultiModelDetectionEngine:
    """Coordinate live frame processing, tracking updates, and overlays."""

    def __init__(
        self,
        source: Optional[str] = None,
        model_path: Optional[str] = None,
        device: str = "cpu",
        confidence: float = 0.25,
        imgsz: int = 640,
    ) -> None:
        self.source = source or "data/sample_videos/your_video.mp4"
        self.model_path = model_path or "yolov8n.pt"
        self.device = device
        self.confidence = confidence
        self.imgsz = imgsz
        self.model: Optional[Any] = None
        self.capture: Optional[Any] = None
        self.roi_polygon = [
            (0.1 * self.imgsz, 0.1 * self.imgsz),
            (0.9 * self.imgsz, 0.15 * self.imgsz),
            (0.95 * self.imgsz, 0.85 * self.imgsz),
            (0.2 * self.imgsz, 0.9 * self.imgsz),
        ]

    def load_model(self) -> Any:
        if YOLO is None:
            raise RuntimeError("ultralytics is required to run the detection pipeline")
        if self.model is None:
            self.model = YOLO(self.model_path)
        return self.model

    def initialize_stream(self) -> Any:
        if cv2 is None:
            raise RuntimeError("cv2 is required to run the detection pipeline")
        self.capture = cv2.VideoCapture(self.source)
        if not self.capture.isOpened():
            raise RuntimeError(f"Unable to open video source: {self.source}")
        return self.capture

    def _resize_frame(self, frame: Any) -> Any:
        if cv2 is None:
            raise RuntimeError("cv2 is required to process frames")
        return cv2.resize(frame, (self.imgsz, self.imgsz), interpolation=cv2.INTER_AREA)

    @staticmethod
    def _overlay_color(state_label: str) -> tuple[int, int, int]:
        if state_label in {"WARNING", "ABANDONED", "PANIC", "INTRUDER"}:
            return (0, 0, 255)
        return (0, 255, 0)

    def _run_inference(self, frame: Any) -> Any:
        model = self.load_model()
        return model.track(
            frame,
            persist=True,
            tracker="bytetrack.yaml",
            conf=self.confidence,
            imgsz=self.imgsz,
            device=self.device,
        )

    def _route_tracking_updates(self, frame: Any, results: Any, timestamp: float) -> None:
        """Synchronize detections into the in-memory tracking structures."""
        update_tracking_states(results, self.roi_polygon, th_frames=10)

        if cv2 is None or frame is None:
            return

        for result in results:
            boxes = getattr(result.boxes, "xywh", None)
            confs = getattr(result.boxes, "conf", None)
            labels = getattr(result.boxes, "cls", None)
            track_ids = getattr(result.boxes, "id", None)
            if boxes is None:
                continue

            for index, box in enumerate(boxes):
                x_center, y_center, width, height = map(float, box)
                person_id = int(track_ids[index]) if track_ids is not None and len(track_ids) > index else index
                confidence = float(confs[index]) if confs is not None and len(confs) > index else 0.0
                label = int(labels[index]) if labels is not None and len(labels) > index else -1

                state_label = tracked_people[person_id].get("state", "NORMAL") if person_id in tracked_people else "NORMAL"
                bag_state = tracked_bags[person_id].get("state", "ATTENDED") if person_id in tracked_bags else "ATTENDED"
                color = self._overlay_color(state_label)

                x1 = int(x_center - width / 2.0)
                y1 = int(y_center - height / 2.0)
                x2 = int(x_center + width / 2.0)
                y2 = int(y_center + height / 2.0)

                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(
                    frame,
                    f"ID:{person_id} {label} | {state_label} | {bag_state}",
                    (x1, max(y1 - 5, 0)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    color,
                    2,
                )

    def run(self) -> None:
        """Drive the live inference loop from a video source."""
        self.initialize_stream()
        try:
            while self.capture.isOpened():
                ret, frame = self.capture.read()  # type: ignore[union-attr]
                if not ret or frame is None:
                    break

                timestamp = time.time()
                resized_frame = self._resize_frame(frame)
                results = self._run_inference(resized_frame)
                self._route_tracking_updates(resized_frame, results, timestamp)

                if cv2 is not None:
                    cv2.imshow("Multi-Model Vigilance", resized_frame)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
        finally:
            self._release()

    def _release(self) -> None:
        if self.capture is not None:
            self.capture.release()
        if cv2 is not None:
            cv2.destroyAllWindows()


def run_live_pipeline(source: Optional[str] = None, model_path: Optional[str] = None) -> MultiModelDetectionEngine:
    """Convenience entry point for the live pipeline."""
    engine = MultiModelDetectionEngine(source=source, model_path=model_path)
    engine.run()
    return engine


__all__ = ["MultiModelDetectionEngine", "run_live_pipeline"]
