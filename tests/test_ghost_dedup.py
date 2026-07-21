import unittest
from unittest.mock import patch

import numpy as np

from src.detection import MultiModelDetectionEngine
from src.state_machine import reset_state, tracked_people


class GhostDedupTests(unittest.TestCase):
    """Regression coverage for the real-world scenario reported while testing:
    ByteTrack's lack of Re-ID means one physical person can leave behind
    several fragmented IDs, and without deduplication each one independently
    renders its own '(PERSISTED)' ghost box, stacking up on the same spot.
    """

    def setUp(self):
        reset_state()
        self.engine = MultiModelDetectionEngine(source="dummy.mp4", model_path="dummy.pt")

    def test_fragmented_ids_at_same_spot_draw_only_the_freshest_ghost(self):
        tracked_people[1] = {"state": "NORMAL", "stale_counter": 40, "last_known_center": (100.0, 100.0), "last_known_width": 50.0, "last_known_height": 120.0}
        tracked_people[4] = {"state": "NORMAL", "stale_counter": 20, "last_known_center": (102.0, 98.0), "last_known_width": 50.0, "last_known_height": 120.0}
        tracked_people[6] = {"state": "NORMAL", "stale_counter": 5, "last_known_center": (98.0, 101.0), "last_known_width": 50.0, "last_known_height": 120.0}
        tracked_people[7] = {"state": "NORMAL", "stale_counter": 25, "last_known_center": (101.0, 99.0), "last_known_width": 50.0, "last_known_height": 120.0}

        frame = np.zeros((640, 640, 3), dtype="uint8")
        with patch("src.detection.cv2.rectangle") as mock_rect, patch("src.detection.cv2.putText") as mock_text:
            self.engine._route_tracking_updates(frame, results=[], timestamp=0.0)

        self.assertEqual(mock_rect.call_count, 1)
        drawn_label = mock_text.call_args[0][1]
        self.assertIn("Person 6", drawn_label)

    def test_persons_far_apart_both_get_their_own_ghost_box(self):
        tracked_people[1] = {"state": "NORMAL", "stale_counter": 5, "last_known_center": (100.0, 100.0), "last_known_width": 50.0, "last_known_height": 120.0}
        tracked_people[2] = {"state": "NORMAL", "stale_counter": 5, "last_known_center": (500.0, 500.0), "last_known_width": 50.0, "last_known_height": 120.0}

        frame = np.zeros((640, 640, 3), dtype="uint8")
        with patch("src.detection.cv2.rectangle") as mock_rect:
            self.engine._route_tracking_updates(frame, results=[], timestamp=0.0)

        self.assertEqual(mock_rect.call_count, 2)

    def test_ghost_stops_rendering_after_person_persist_frames(self):
        self.engine.state_machine_config["person_persist_frames"] = 10
        tracked_people[1] = {"state": "NORMAL", "stale_counter": 15, "last_known_center": (100.0, 100.0), "last_known_width": 50.0, "last_known_height": 120.0}

        frame = np.zeros((640, 640, 3), dtype="uint8")
        with patch("src.detection.cv2.rectangle") as mock_rect:
            self.engine._route_tracking_updates(frame, results=[], timestamp=0.0)

        self.assertEqual(mock_rect.call_count, 0)


if __name__ == "__main__":
    unittest.main()
