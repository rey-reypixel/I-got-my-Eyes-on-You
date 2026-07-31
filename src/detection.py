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
    import numpy as np
except ImportError:  # pragma: no cover - depends on runtime environment
    np = None

try:
    from ultralytics import YOLO
except ImportError:  # pragma: no cover - depends on runtime environment
    YOLO = None

from . import event_log
from .state_machine import (
    reset_state,
    tracked_bags,
    tracked_people,
)
from .tracking import (
    compute_limb_motion_score,
    get_pose_gated_person_ids,
    process_behavioral_smoothing,
    update_tracking_states,
)


def letterbox_resize(frame: Any, imgsz: int) -> Any:
    """Letterbox to imgsz x imgsz: scale preserving aspect ratio, then pad with
    neutral gray. A naive stretch-to-square resize distorts human proportions
    non-uniformly on any non-square source video -- e.g. 320x240 -> 640x640
    stretches width 2.0x but height 2.667x -- which can plausibly hurt
    detection recall on top of whatever the source footage's own quality
    issues are. Shared by MultiModelDetectionEngine and the benchmark scripts
    so a baseline-vs-custom comparison isolates the engineering layer, not an
    incidental difference in preprocessing.
    """
    if cv2 is None:
        raise RuntimeError("cv2 is required to process frames")
    if np is None:
        raise RuntimeError("numpy is required to process frames")

    src_h, src_w = frame.shape[:2]
    if src_w == imgsz and src_h == imgsz:
        return frame

    scale = min(imgsz / src_w, imgsz / src_h)
    new_w = max(1, int(round(src_w * scale)))
    new_h = max(1, int(round(src_h * scale)))
    resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)

    canvas = np.full((imgsz, imgsz, 3), 114, dtype=resized.dtype)
    pad_x = (imgsz - new_w) // 2
    pad_y = (imgsz - new_h) // 2
    canvas[pad_y : pad_y + new_h, pad_x : pad_x + new_w] = resized
    return canvas


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


def process_abandoned_logic(
    bag_id: int,
    bag_state: dict,
    person_coords: Optional[tuple],
    th_frames: int,
    d_disconnect: float = 50.0,
    d_reconnect: float = 30.0,
):
    """Manage temporal abandoned-object state transitions with proximity hysteresis."""
    if person_coords is None:
        d = float("inf")
    else:
        d = math.sqrt((person_coords[0] - bag_state["center_coords"][0]) ** 2 + (person_coords[1] - bag_state["center_coords"][1]) ** 2)

    if bag_state["state"] == "ATTENDED":
        if d > d_disconnect:
            bag_state["state"] = "WARNING"
            bag_state["timer"] = 1
    elif bag_state["state"] in ("WARNING", "ABANDONED"):
        if d < d_reconnect:
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
    if len(velocity_history) < 6:
        return False

    # Smooth the current velocity over the last 3 frames to filter out single-frame tracking glitches/jitter
    v_current = sum(velocity_history[-3:]) / 3.0
    v_historical = velocity_history[:-3]
    mean_v = sum(v_historical) / len(v_historical)
    
    # Require the current smoothed velocity to be at least 0.18 (18% of body height/frame, representing actual running/sprinting)
    if v_current < 0.18:
        return False

    if v_current > (mean_v * threshold_devs):
        return True
    return False


class MultiModelDetectionEngine:
    """Coordinate live frame processing, tracking updates, and overlays."""

    def __init__(
        self,
        source: Optional[str] = None,
        model_path: Optional[str] = None,
        pose_model_path: Optional[str] = None,
        device: str = "cpu",
        confidence: float = 0.25,
        imgsz: int = 640,
        config_path: Optional[str] = None,
        max_frames: Optional[int] = None,
        show_window: bool = True,
        event_log_path: Optional[str] = None,
    ) -> None:
        self.source = source or "data/sample_videos/your_video.mp4"
        self.model_path = model_path or "yolov8n.pt"
        self.pose_model_path = pose_model_path or "yolov8n-pose.pt"
        self.event_log_path = event_log_path
        self._event_run_id: Optional[int] = None
        self._logged_prev_states: dict[str, str] = {}
        self.device = device
        self.confidence = confidence
        self.imgsz = imgsz
        self.max_frames = max_frames
        self.show_window = show_window

        self.config_data = {}
        if config_path:
            import json
            try:
                with open(config_path, "r", encoding="utf-8") as f:
                    self.config_data = json.load(f)
            except Exception as e:
                print(f"Warning: Failed to load config from {config_path}: {e}")

        # Override from config if available
        self.confidence = self.config_data.get("detection", {}).get("confidence", self.confidence)
        self.imgsz = self.config_data.get("detection", {}).get("imgsz", self.imgsz)
        self.device = self.config_data.get("detection", {}).get("device", self.device)
        self.state_machine_config = self.config_data.get("state_machine", {})
        # Inject imgsz for normalized coordinate calculations
        self.state_machine_config["imgsz"] = self.imgsz

        self.model: Optional[Any] = None
        self.pose_model: Optional[Any] = None
        self.capture: Optional[Any] = None

        raw_polygon = self.config_data.get("roi_polygon")
        
        # If the polygon matches the hardcoded AVSS train tracks safety boundary,
        # we only apply it if the video source path is indeed an AVSS video.
        if raw_polygon == [[0.0, 0.0], [0.22, 0.0], [0.12, 1.0], [0.0, 1.0]]:
            source_str = str(self.source).upper()
            if "AVSS" not in source_str:
                self.roi_polygon = []
                return

        if raw_polygon:
            self.roi_polygon = []
            for pt in raw_polygon:
                px, py = pt
                if isinstance(px, float) and 0.0 <= px <= 1.0:
                    px = px * self.imgsz
                if isinstance(py, float) and 0.0 <= py <= 1.0:
                    py = py * self.imgsz
                self.roi_polygon.append((px, py))
        else:
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

    def load_pose_model(self) -> Any:
        """Lazily loads the pose model -- only touched on frames where at least
        one tracked person is gated in (see get_pose_gated_person_ids), so an
        idle/walking-only scene never pays for it.
        """
        if YOLO is None:
            raise RuntimeError("ultralytics is required to run the detection pipeline")
        if self.pose_model is None:
            self.pose_model = YOLO(self.pose_model_path)
        return self.pose_model

    def _reset_after_seek(self) -> None:
        """Clear all tracking state after a manual rewind/skip.

        Both `model.track(..., persist=True)` (ByteTrack's own internal Kalman
        filters and lost/removed-track buffers) and our own tracking layer
        (tracked_people/tracked_bags/stale_counter/last_known_* etc.) are built
        on the assumption of continuous, monotonically-increasing frame input.
        A manual seek breaks that assumption outright: ByteTrack's motion
        prediction and ID assignment become meaningless once the video jumps
        discontinuously, and our own bookkeeping (stale_counter, ghost/PERSISTED
        positions, accumulated WARNING/ABANDONED timers) reflects a temporal
        position that no longer matches whatever the video jumped to. Without
        this, bounding boxes and their persistence become visibly incoherent
        after any rewind/skip -- ghost boxes frozen at positions from the wrong
        point in time, or IDs and states left over from a future/past segment
        the video no longer shows. Reported from real use (rewind/forward
        keys during playback). The only correct-enough fix is a hard reset:
        start tracking completely fresh from wherever the video lands.
        """
        reset_state()
        if self.model is not None:
            predictor = getattr(self.model, "predictor", None)
            trackers = getattr(predictor, "trackers", None) if predictor is not None else None
            if trackers:
                for tracker in trackers:
                    reset_fn = getattr(tracker, "reset", None)
                    if callable(reset_fn):
                        reset_fn()

    def initialize_stream(self) -> Any:
        if cv2 is None:
            raise RuntimeError("cv2 is required to run the detection pipeline")
        self.capture = cv2.VideoCapture(self.source)
        if not self.capture.isOpened():
            raise RuntimeError(f"Unable to open video source: {self.source}")
        return self.capture

    def _resize_frame(self, frame: Any) -> Any:
        return letterbox_resize(frame, self.imgsz)

    @staticmethod
    def _overlay_color(state_label: str) -> tuple[int, int, int]:
        if state_label in {"WARNING", "ABANDONED", "PANIC", "INTRUDER", "SUSPICIOUS_ACTIVITY"}:
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
            classes=[0, 24, 26, 28],
            verbose=False,
        )

    def _run_pose_inference(self, frame: Any) -> Any:
        """Whole-frame pose pass (no tracker of its own -- its detections are
        matched back to the main tracked person IDs by nearest centroid, since
        it's an independent model with an independent detection set).
        """
        model = self.load_pose_model()
        return model.predict(
            frame,
            conf=self.confidence,
            imgsz=self.imgsz,
            device=self.device,
            classes=[0],
            verbose=False,
        )

    def _process_limb_motion_gate(self, frame: Any, results: Any) -> None:
        """Runs the pose model only when at least one tracked person has been
        gated in by sustained centroid stillness, matches its detections back
        to tracked person IDs by nearest centroid, and feeds the resulting
        limb-motion score through the same EMA smoothing used for behavioral
        state (see process_behavioral_smoothing in src/tracking.py).
        """
        gated_ids = get_pose_gated_person_ids(self.state_machine_config)
        if not gated_ids:
            return

        # Map gated person IDs to their current centroid, from this frame's detections.
        person_centroids: dict[int, tuple[float, float]] = {}
        for result in results:
            track_ids = getattr(result.boxes, "id", None)
            if track_ids is None:
                continue
            boxes = getattr(result.boxes, "xywh", None)
            labels = getattr(result.boxes, "cls", None)
            names = getattr(result, "names", None)
            if boxes is None:
                continue
            for index, box in enumerate(boxes):
                x_center, y_center, _w, _h = map(float, box)
                label = int(labels[index]) if labels is not None and len(labels) > index else -1
                class_label = names.get(label, str(label)) if names is not None else str(label)
                if class_label == "person" or label == 0:
                    person_centroids[int(track_ids[index])] = (x_center, y_center)

        gated_with_centroid = {pid: person_centroids[pid] for pid in gated_ids if pid in person_centroids}
        if not gated_with_centroid:
            return

        pose_results = self._run_pose_inference(frame)
        for pose_result in pose_results:
            pose_boxes = getattr(pose_result.boxes, "xywh", None)
            keypoints = getattr(pose_result, "keypoints", None)
            if pose_boxes is None or keypoints is None:
                continue
            kp_xy_all = keypoints.xy
            kp_conf_all = keypoints.conf
            if kp_conf_all is None:
                continue

            for index, box in enumerate(pose_boxes):
                px, py, _w, _h = map(float, box)

                # Nearest-centroid match to a gated tracked person within a tolerant radius.
                best_pid, best_dist = None, float("inf")
                for pid, (cx, cy) in gated_with_centroid.items():
                    dist = math.hypot(px - cx, py - cy)
                    if dist < best_dist:
                        best_pid, best_dist = pid, dist
                if best_pid is None or best_dist > 60.0:
                    continue

                norm_h = tracked_people.get(best_pid, {}).get("max_observed_height", 50.0)
                score = compute_limb_motion_score(
                    best_pid,
                    kp_xy_all[index],
                    kp_conf_all[index],
                    norm_h=norm_h,
                    state_machine_config=self.state_machine_config,
                )
                if score is not None:
                    process_behavioral_smoothing(best_pid, score, self.state_machine_config)

    def _log_state_changes(self, frame_id: int) -> None:
        """Persists a row to the event log for every entity whose state
        differs from what was last logged for it (both new alerts and
        recoveries back to NORMAL/ATTENDED), so a run's full history can be
        queried afterwards -- e.g. by the MCP server in mcp_server/server.py.
        """
        for pid, meta in tracked_people.items():
            key = f"person:{pid}"
            state = meta.get("state", "NORMAL")
            if self._logged_prev_states.get(key) != state:
                event_log.log_state_change(self.event_log_path, self._event_run_id, frame_id, "person", pid, state)
                self._logged_prev_states[key] = state

        for bid, meta in tracked_bags.items():
            key = f"bag:{bid}"
            state = meta.get("state", "ATTENDED")
            if self._logged_prev_states.get(key) != state:
                event_log.log_state_change(self.event_log_path, self._event_run_id, frame_id, "bag", bid, state)
                self._logged_prev_states[key] = state

    def _route_tracking_updates(self, frame: Any, results: Any, timestamp: float, frame_id: int = 0) -> None:
        """Synchronize detections into the in-memory tracking structures."""
        update_tracking_states(
            results,
            self.roi_polygon,
            th_frames=self.state_machine_config.get("th_frames", 10),
            state_machine_config=self.state_machine_config,
        )

        if self.event_log_path and self._event_run_id is not None:
            self._log_state_changes(frame_id)

        if frame is not None:
            self._process_limb_motion_gate(frame, results)

        if cv2 is None or frame is None:
            return

        drawn_bag_ids = set()
        drawn_person_ids = set()
        live_person_centers: list[tuple[float, float]] = []

        for result in results:
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
                person_id = int(track_ids[index])
                confidence = float(confs[index]) if confs is not None and len(confs) > index else 0.0
                label = int(labels[index]) if labels is not None and len(labels) > index else -1

                class_label = str(label)
                if names is not None:
                    try:
                        class_label = names.get(label, str(label))
                    except AttributeError:
                        class_label = str(label)

                is_person = (class_label == "person") or (label == 0)
                is_bag = (class_label in {"backpack", "handbag", "suitcase"}) or (label in {24, 26, 28})

                state_label = "NORMAL"
                bag_state = "ATTENDED"
                status_label = f"ID:{person_id} {class_label}"
                if is_person:
                    drawn_person_ids.add(person_id)
                    live_person_centers.append((x_center, y_center))
                    state_label = tracked_people[person_id].get("state", "NORMAL") if person_id in tracked_people else "NORMAL"
                    status_label = f"Person {person_id}: {state_label}"
                elif is_bag:
                    drawn_bag_ids.add(person_id)
                    bag_state = tracked_bags[person_id].get("state", "ATTENDED") if person_id in tracked_bags else "ATTENDED"
                    status_label = f"Bag {person_id}: {bag_state}"

                current_state_for_color = state_label if is_person else (bag_state if is_bag else "NORMAL")
                color = self._overlay_color(current_state_for_color)

                x1 = int(x_center - width / 2.0)
                y1 = int(y_center - height / 2.0)
                x2 = int(x_center + width / 2.0)
                y2 = int(y_center + height / 2.0)

                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(
                    frame,
                    status_label,
                    (x1, max(y1 - 5, 0)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    color,
                    2,
                )

        # Persist visual rendering for tracked people during brief YOLO detection
        # dropouts, mirroring the bag persistence below. A momentary confidence
        # dip (pose change, lighting, brief occlusion) should not make an
        # otherwise-still-tracked person's box vanish outright.
        #
        # Identity fragmentation (ByteTrack has no Re-ID/appearance matching --
        # see bugs_and_debugs.txt #26) means the *same* physical person can leave
        # behind several different tracked IDs over time. Without deduplication,
        # every one of those IDs independently persists its own ghost box, so a
        # single person can appear to be wearing 3-4 stacked "PERSISTED" boxes
        # long after she's actually left. Claim each screen position once,
        # preferring the most recently seen (lowest stale_counter) candidate.
        # Unlike a stationary bag, a person is mobile -- a ghost box frozen in
        # place for the bag's full 5-second window would misrepresent reality
        # ("she's still standing there") long after she's actually walked off.
        # This gets its own, much shorter window: enough to bridge a 1-2 frame
        # confidence dip, not long enough to become a stale, wrong claim.
        person_persist_frames = self.state_machine_config.get("person_persist_frames", 30)
        ghost_candidates = [
            (person_id, person_info)
            for person_id, person_info in tracked_people.items()
            if person_id not in drawn_person_ids
            and person_info.get("stale_counter", 0) < person_persist_frames
            and person_info.get("last_known_center") is not None
        ]
        ghost_candidates.sort(key=lambda item: item[1].get("stale_counter", 0))

        claimed_centers = list(live_person_centers)
        DUPLICATE_GHOST_RADIUS = 80.0  # matches the bag duplicate-suppression distance

        for person_id, person_info in ghost_candidates:
            px, py = person_info["last_known_center"]
            if any(math.hypot(px - cx, py - cy) < DUPLICATE_GHOST_RADIUS for cx, cy in claimed_centers):
                continue
            claimed_centers.append((px, py))

            pw = person_info.get("last_known_width", 40.0)
            ph = person_info.get("last_known_height", 100.0)
            state_label = person_info.get("state", "NORMAL")
            color = self._overlay_color(state_label)
            status_label = f"Person {person_id}: {state_label} (PERSISTED)"

            x1 = int(px - pw / 2.0)
            y1 = int(py - ph / 2.0)
            x2 = int(px + pw / 2.0)
            y2 = int(py + ph / 2.0)

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                frame,
                status_label,
                (x1, max(y1 - 5, 0)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                2,
            )

        # Persist visual rendering for all tracked bags during YOLO detection dropouts.
        # The cutoff (config bag_persist_frames, default 900 / 30s at 30fps) is set from
        # a real observed gap, not a guess: tracing a genuinely left-behind bag through
        # ABODA video1.avi frame-by-frame found YOLO detecting zero backpack-class objects
        # at all for 649 consecutive frames (~21.6s) before reacquiring it -- far beyond the
        # previous hardcoded 150-frame (5s) cutoff, which let the box go dark for most of
        # that real gap despite the bag never having actually moved. Matches
        # state_max_stale_frames (also raised to 900) so the box keeps rendering for the
        # entry's entire in-memory lifetime rather than going dark before eviction.
        bag_persist_frames = self.state_machine_config.get("bag_persist_frames", 900)
        for bag_id, bag_info in tracked_bags.items():
            if bag_id not in drawn_bag_ids:
                if bag_info.get("stale_counter", 0) < bag_persist_frames:
                    bx, by = bag_info["center_coords"]
                    bw = bag_info.get("width", 40.0)
                    bh = bag_info.get("height", 40.0)
                    bag_state = bag_info["state"]
                    color = self._overlay_color(bag_state)
                    status_label = f"Bag {bag_id}: {bag_state} (PERSISTED)"

                    x1 = int(bx - bw / 2.0)
                    y1 = int(by - bh / 2.0)
                    x2 = int(bx + bw / 2.0)
                    y2 = int(by + bh / 2.0)

                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(
                        frame,
                        status_label,
                        (x1, max(y1 - 5, 0)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        color,
                        2,
                    )

    def run(self) -> None:
        """Drive the live inference loop from a video source."""
        self.initialize_stream()
        if self.event_log_path:
            event_log.init_db(self.event_log_path)
            self._event_run_id = event_log.start_run(self.event_log_path, str(self.source))
        try:
            frame_count = 0
            while self.capture.isOpened():
                ret, frame = self.capture.read()  # type: ignore[union-attr]
                if not ret or frame is None:
                    break

                frame_count += 1
                if self.max_frames is not None and frame_count > self.max_frames:
                    break

                timestamp = time.time()
                resized_frame = self._resize_frame(frame)
                results = self._run_inference(resized_frame)
                self._route_tracking_updates(resized_frame, results, timestamp, frame_id=frame_count)

                if cv2 is not None and self.show_window:
                    cv2.imshow("Multi-Model Vigilance", resized_frame)
                    key = cv2.waitKey(1) & 0xFF
                    
                    if key == ord("p") or key == 32:  # 'p' or Spacebar to pause
                        while True:
                            resume_key = cv2.waitKey(0) & 0xFF
                            if resume_key == ord("q"):
                                key = ord("q")
                                break
                            elif resume_key in (ord("r"), 81, ord("[")):
                                # Rewind 10 seconds while paused
                                fps = self.capture.get(cv2.CAP_PROP_FPS)
                                if fps <= 0:
                                    fps = 30.0
                                current_frame = self.capture.get(cv2.CAP_PROP_POS_FRAMES)
                                target_frame = max(0.0, current_frame - (10.0 * fps) - 1.0)
                                self.capture.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
                                frame_count = int(target_frame)
                                self._reset_after_seek()
                                key = ord("p")  # Keep paused after updating frame
                                break
                            elif resume_key in (ord("f"), 83, ord("]")):
                                # Skip 10 seconds while paused
                                fps = self.capture.get(cv2.CAP_PROP_FPS)
                                if fps <= 0:
                                    fps = 30.0
                                total_frames = self.capture.get(cv2.CAP_PROP_FRAME_COUNT)
                                current_frame = self.capture.get(cv2.CAP_PROP_POS_FRAMES)
                                target_frame = min(total_frames, current_frame + (10.0 * fps) - 1.0)
                                self.capture.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
                                frame_count = int(target_frame)
                                self._reset_after_seek()
                                key = ord("p")  # Keep paused after updating frame
                                break
                            elif resume_key == ord("p") or resume_key == 32:
                                break
                            else:
                                continue

                    if key == ord("q"):
                        break
                    elif key in (ord("r"), 81, ord("[")):
                        # Rewind 10 seconds during active playback
                        fps = self.capture.get(cv2.CAP_PROP_FPS)
                        if fps <= 0:
                            fps = 30.0
                        current_frame = self.capture.get(cv2.CAP_PROP_POS_FRAMES)
                        target_frame = max(0.0, current_frame - (10.0 * fps) - 1.0)
                        self.capture.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
                        frame_count = int(target_frame)
                        self._reset_after_seek()
                    elif key in (ord("f"), 83, ord("]")):
                        # Skip 10 seconds during active playback
                        fps = self.capture.get(cv2.CAP_PROP_FPS)
                        if fps <= 0:
                            fps = 30.0
                        total_frames = self.capture.get(cv2.CAP_PROP_FRAME_COUNT)
                        current_frame = self.capture.get(cv2.CAP_PROP_POS_FRAMES)
                        target_frame = min(total_frames, current_frame + (10.0 * fps) - 1.0)
                        self.capture.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
                        frame_count = int(target_frame)
                        self._reset_after_seek()
        finally:
            if self.event_log_path and self._event_run_id is not None:
                event_log.end_run(self.event_log_path, self._event_run_id)
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
