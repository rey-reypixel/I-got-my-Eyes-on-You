"""Thread-safe in-memory state containers for frame-by-frame tracking updates."""

import threading
import time
from typing import Any, Dict, List, Optional

# 1. Human Tracking Memory
# Key: person_id (int) -> Value: dict of kinematic and spatial parameters
tracked_people_lock = threading.Lock()
tracked_people: Dict[int, Dict[str, Any]] = {
    # Example Schema:
    # person_id: {
    #     "path_history": [],
    #     "velocities": [],
    #     "state": "NORMAL",
    #     "panic_counter": 0,
    # }
}

# 2. Asset Tracking Memory
# Key: bag_id (int) -> Value: dict of temporal state machine attributes
tracked_bags_lock = threading.Lock()
tracked_bags: Dict[int, Dict[str, Any]] = {
    # Example Schema:
    # bag_id: {
    #     "owner_id": None,
    #     "center_coords": (0, 0),
    #     "state": "ATTENDED",
    #     "timer": 0,
    # }
}

# 3. Behavioral Window Buffer (UCF-Crime)
# Key: tracklet_id/person_id (int) -> Value: list of model confidence scores
behavioral_smoothing_buffer_lock = threading.Lock()
behavioral_smoothing_buffer: Dict[int, List[float]] = {
    # Example Schema:
    # person_id: []
}

# 4. Short-lived echo of bags evicted while classified as "carried" (see
# bugs_and_debugs.txt #29's follow-up). A carried bag is deleted outright the
# instant it drops out of detection (to avoid its ghost lingering/following
# the wrong person), which otherwise loses all accumulated state/timer if the
# same physical bag is reacquired moments later under a new (or reused)
# track ID. This buffer keeps just enough of a trace, for a short frame
# window, for that reacquisition to recover the accumulated state.
recently_evicted_bags_lock = threading.Lock()
recently_evicted_bags: Dict[int, Dict[str, Any]] = {
    # Example Schema:
    # bag_id: {"center_coords": (0, 0), "state": "ATTENDED", "timer": 0, "owner_id": None, "ttl": 30}
}


def reset_state() -> None:
    """Clear all in-memory tracking structures safely."""
    with tracked_people_lock:
        tracked_people.clear()
    with tracked_bags_lock:
        tracked_bags.clear()
    with behavioral_smoothing_buffer_lock:
        behavioral_smoothing_buffer.clear()
    with recently_evicted_bags_lock:
        recently_evicted_bags.clear()


def register_person(person_id: int, state: str = "NORMAL") -> None:
    """Initialize a person entry with default tracking values."""
    with tracked_people_lock:
        tracked_people[person_id] = {
            "path_history": [],
            "velocities": [],
            "state": state,
            "panic_counter": 0,
        }


def register_bag(bag_id: int, owner_id: Optional[int] = None) -> None:
    """Initialize a bag entry with default temporal state values."""
    with tracked_bags_lock:
        tracked_bags[bag_id] = {
            "owner_id": owner_id,
            "center_coords": (0, 0),
            "state": "ATTENDED",
            "timer": 0,
        }


def append_behavioral_confidence(tracklet_id: int, confidence: float, max_window: int = 15) -> None:
    """Append a confidence sample while keeping the rolling buffer bounded."""
    with behavioral_smoothing_buffer_lock:
        history = behavioral_smoothing_buffer.setdefault(tracklet_id, [])
        history.append(float(confidence))
        if len(history) > max_window:
            history.pop(0)


__all__ = [
    "tracked_people",
    "tracked_bags",
    "behavioral_smoothing_buffer",
    "recently_evicted_bags",
    "tracked_people_lock",
    "tracked_bags_lock",
    "behavioral_smoothing_buffer_lock",
    "recently_evicted_bags_lock",
    "reset_state",
    "register_person",
    "register_bag",
    "append_behavioral_confidence",
]
