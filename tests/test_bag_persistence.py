import unittest
from unittest.mock import patch

import numpy as np

from src.detection import MultiModelDetectionEngine
from src.state_machine import reset_state, tracked_bags


class BagPersistenceWindowTests(unittest.TestCase):
    """Regression coverage for a real-world gap found while tracing a
    genuinely left-behind bag through ABODA video1.avi frame-by-frame: YOLO
    detected zero backpack-class objects at all for 649 consecutive frames
    (~21.6s) before reacquiring it. The previous hardcoded 150-frame (5s)
    persistence cutoff in src/detection.py let the bag's box go dark for most
    of that real gap despite the bag never having actually moved -- not a
    state-machine logic bug, a render cutoff sized far below an observed real
    detection gap. See bugs_and_debugs.txt #34.
    """

    def setUp(self):
        reset_state()
        self.engine = MultiModelDetectionEngine(source="dummy.mp4", model_path="dummy.pt")

    def test_bag_still_renders_deep_into_a_long_dropout(self):
        # Well past the old 150-frame cutoff, comfortably inside the new default.
        tracked_bags[20] = {
            "owner_id": 11,
            "center_coords": (140.0, 417.0),
            "width": 40.0,
            "height": 40.0,
            "state": "WARNING",
            "timer": 5,
            "stale_counter": 600,
            "is_carried": False,
        }

        frame = np.zeros((640, 640, 3), dtype="uint8")
        with patch("src.detection.cv2.rectangle") as mock_rect, patch("src.detection.cv2.putText") as mock_text:
            self.engine._route_tracking_updates(frame, results=[], timestamp=0.0)

        self.assertEqual(mock_rect.call_count, 1)
        drawn_label = mock_text.call_args[0][1]
        self.assertIn("Bag 20", drawn_label)
        self.assertIn("PERSISTED", drawn_label)

    def test_bag_stops_rendering_past_the_configured_cutoff(self):
        tracked_bags[20] = {
            "owner_id": 11,
            "center_coords": (140.0, 417.0),
            "width": 40.0,
            "height": 40.0,
            "state": "WARNING",
            "timer": 5,
            "stale_counter": 950,  # past the new default (900)
            "is_carried": False,
        }

        frame = np.zeros((640, 640, 3), dtype="uint8")
        with patch("src.detection.cv2.rectangle") as mock_rect:
            self.engine._route_tracking_updates(frame, results=[], timestamp=0.0)

        self.assertEqual(mock_rect.call_count, 0)

    def test_cutoff_is_configurable(self):
        self.engine.state_machine_config["bag_persist_frames"] = 10
        tracked_bags[20] = {
            "owner_id": 11,
            "center_coords": (140.0, 417.0),
            "width": 40.0,
            "height": 40.0,
            "state": "ATTENDED",
            "timer": 0,
            "stale_counter": 15,
            "is_carried": False,
        }

        frame = np.zeros((640, 640, 3), dtype="uint8")
        with patch("src.detection.cv2.rectangle") as mock_rect:
            self.engine._route_tracking_updates(frame, results=[], timestamp=0.0)

        self.assertEqual(mock_rect.call_count, 0)


if __name__ == "__main__":
    unittest.main()
