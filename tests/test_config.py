import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from src.detection import MultiModelDetectionEngine
from src.state_machine import reset_state, tracked_people, tracked_bags
from src.tracking import update_tracking_states

class ConfigTests(unittest.TestCase):
    def test_default_config_loading(self):
        # Test loading of config and setting properties in MultiModelDetectionEngine
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "test_config.json"
            config_content = {
                "roi_polygon": [
                    [0.15, 0.15],
                    [0.85, 0.85]
                ],
                "detection": {
                    "confidence": 0.35,
                    "imgsz": 320,
                    "device": "cpu"
                },
                "state_machine": {
                    "th_frames": 25,
                    "panic_threshold_devs": 4.5,
                    "bag_disconnect_dist": 60.0,
                    "bag_reconnect_dist": 40.0
                }
            }
            with config_path.open("w", encoding="utf-8") as f:
                json.dump(config_content, f)

            engine = MultiModelDetectionEngine(
                source="dummy.mp4",
                model_path="dummy.pt",
                config_path=str(config_path)
            )

            self.assertEqual(engine.confidence, 0.35)
            self.assertEqual(engine.imgsz, 320)
            self.assertEqual(engine.device, "cpu")
            self.assertEqual(engine.state_machine_config["th_frames"], 25)
            # Verify normalized ROI scaling: 0.15 * 320 = 48.0, 0.85 * 320 = 272.0
            expected_polygon = [(48.0, 48.0), (272.0, 272.0)]
            self.assertEqual(engine.roi_polygon, expected_polygon)

    def test_missing_config_falls_back_to_defaults(self):
        engine = MultiModelDetectionEngine(
            source="dummy.mp4",
            model_path="dummy.pt",
            confidence=0.15,
            imgsz=640,
            config_path="nonexistent_config_file.json"
        )
        self.assertEqual(engine.confidence, 0.15)
        self.assertEqual(engine.imgsz, 640)
        self.assertEqual(len(engine.roi_polygon), 4)

    def test_stale_state_eviction(self):
        reset_state()
        results = [
            SimpleNamespace(
                boxes=SimpleNamespace(
                    xywh=[[10.0, 10.0, 20.0, 20.0]],
                    conf=[0.92],
                    cls=[0],
                    id=[42],
                ),
                names={0: "person"},
            )
        ]

        config = {
            "state_max_stale_frames": 2
        }

        # 1. Update with active detection: registers person 42 with stale_counter = 0
        update_tracking_states(results, polygon_vertices=[], state_machine_config=config)
        self.assertIn(42, tracked_people)
        self.assertEqual(tracked_people[42]["stale_counter"], 0)

        # 2. Update with empty results (person missing): stale_counter becomes 1
        update_tracking_states([], polygon_vertices=[], state_machine_config=config)
        self.assertIn(42, tracked_people)
        self.assertEqual(tracked_people[42]["stale_counter"], 1)

        # 3. Update with empty results again: stale_counter becomes 2 (reaches limit, not evicted yet)
        update_tracking_states([], polygon_vertices=[], state_machine_config=config)
        self.assertIn(42, tracked_people)
        self.assertEqual(tracked_people[42]["stale_counter"], 2)

        # 4. Update with empty results again: stale_counter becomes 3 (> limit), evicted!
        update_tracking_states([], polygon_vertices=[], state_machine_config=config)
        self.assertNotIn(42, tracked_people)

    def test_suspicious_activity_velocity_spike(self):
        reset_state()
        config = {
            "kinematic_velocity_threshold": 0.05,
            "min_velocity_history_frames": 2,
            "imgsz": 100
        }

        # 1. First frame - person at (30, 30)
        results_t1 = [
            SimpleNamespace(
                boxes=SimpleNamespace(
                    xywh=[[30.0, 30.0, 20.0, 20.0]],
                    conf=[0.92],
                    cls=[0],
                    id=[42],
                ),
                names={0: "person"},
            )
        ]
        update_tracking_states(results_t1, polygon_vertices=[], state_machine_config=config)
        self.assertEqual(tracked_people[42]["state"], "NORMAL")

        # 2. Second frame - person moves to (40, 40) -> displacement is sqrt(100+100)=14.14
        # Normalized displacement is 14.14 / 20.0 = 0.707 > 0.05 (threshold)
        results_t2 = [
            SimpleNamespace(
                boxes=SimpleNamespace(
                    xywh=[[40.0, 40.0, 20.0, 20.0]],
                    conf=[0.92],
                    cls=[0],
                    id=[42],
                ),
                names={0: "person"},
            )
        ]
        update_tracking_states(results_t2, polygon_vertices=[], state_machine_config=config)
        self.assertEqual(tracked_people[42]["state"], "SUSPICIOUS_ACTIVITY")

if __name__ == "__main__":
    unittest.main()
