"""Primary multi-model synchronization loop for live video analysis."""

from __future__ import annotations

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
    append_behavioral_confidence,
    behavioral_smoothing_buffer_lock,
    register_bag,
    register_person,
    tracked_bags,
    tracked_bags_lock,
    tracked_people,
    tracked_people_lock,
)


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
        self.source = source or str(Path("data/sample_videos"))
        self.model_path = model_path or "yolov8n.pt"
        self.device = device
        self.confidence = confidence
        self.imgsz = imgsz
        self.model: Optional[Any] = None
        self.capture: Optional[Any] = None

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

    def _run_inference(self, frame: Any) -> Any:
        model = self.load_model()
        return model(frame, stream=True, conf=self.confidence, imgsz=self.imgsz, device=self.device)

    def _route_tracking_updates(self, frame: Any, results: Any, timestamp: float) -> None:
        """Synchronize detections into the in-memory tracking structures."""
        for result in results:
            boxes = getattr(result.boxes, "xywh", None)
            confs = getattr(result.boxes, "conf", None)
            if boxes is None:
                continue

            for index, box in enumerate(boxes):
                x_center, y_center, width, height = map(float, box)
                person_id = index
                confidence = float(confs[index]) if confs is not None and len(confs) > index else 0.0

                with tracked_people_lock:
                    tracked_people.setdefault(
                        person_id,
                        {
                            "path_history": [],
                            "velocities": [],
                            "state": "NORMAL",
                            "panic_counter": 0,
                        },
                    )
                    tracked_people[person_id]["path_history"].append((x_center, y_center))
                    if len(tracked_people[person_id]["path_history"]) > 30:
                        tracked_people[person_id]["path_history"].pop(0)

                with tracked_bags_lock:
                    tracked_bags.setdefault(person_id, {"owner_id": person_id, "center_coords": (x_center, y_center), "state": "ATTENDED", "timer": 0})
                    tracked_bags[person_id]["center_coords"] = (x_center, y_center)

                append_behavioral_confidence(person_id, confidence)

                if cv2 is not None and frame is not None:
                    cv2.putText(
                        frame,
                        f"P{person_id}:{confidence:.2f}",
                        (int(x_center), int(y_center)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 255, 0),
                        2,
                    )

    def run(self) -> None:
        """Drive the live inference loop from a video source."""
        self.initialize_stream()
        try:
            while True:
                ok, frame = self.capture.read()  # type: ignore[union-attr]
                if not ok or frame is None:
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
