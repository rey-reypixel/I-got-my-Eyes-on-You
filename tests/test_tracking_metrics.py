import unittest
from types import SimpleNamespace

from src.state_machine import reset_state
from src.tracking import process_tracking_state_machine


class TrackingMetricsTests(unittest.TestCase):
    def test_process_tracking_state_machine_returns_metric_rows(self):
        reset_state()
        results = [
            SimpleNamespace(
                boxes=SimpleNamespace(
                    xywh=[[10.0, 10.0, 20.0, 20.0]],
                    conf=[0.92],
                    cls=[1],
                    id=[7],
                ),
                names={1: "person"},
            )
        ]

        metrics = process_tracking_state_machine(results, frame_id=1, polygon_vertices=[])

        self.assertIsInstance(metrics, list)
        self.assertEqual(len(metrics), 1)
        self.assertEqual(metrics[0]["object_id"], 7)
        self.assertIn(metrics[0]["current_state"], {"NORMAL", "PANIC"})
        self.assertFalse(metrics[0]["threat_alert_triggered"])

    def test_process_behavioral_smoothing(self):
        reset_state()
        from src.state_machine import tracked_people
        from src.tracking import process_behavioral_smoothing

        tracked_people[1] = {
            "state": "NORMAL",
        }

        config = {
            "behavioral_smoothing": {
                "window_size": 3,
                "activation_threshold": 0.8,
                "beta": 0.8
            }
        }

        # 1. First frame - score 0.9 -> initial EMA is 0.9 >= 0.8 -> SUSPICIOUS_ACTIVITY
        process_behavioral_smoothing(1, 0.9, config)
        self.assertEqual(tracked_people[1]["state"], "SUSPICIOUS_ACTIVITY")
        self.assertAlmostEqual(tracked_people[1]["anomaly_score_ema"], 0.9)

        # 2. Second frame - score 0.5 -> EMA becomes 0.8 * 0.9 + 0.2 * 0.5 = 0.82 >= 0.8 -> remains SUSPICIOUS_ACTIVITY (persistency!)
        process_behavioral_smoothing(1, 0.5, config)
        self.assertEqual(tracked_people[1]["state"], "SUSPICIOUS_ACTIVITY")
        self.assertAlmostEqual(tracked_people[1]["anomaly_score_ema"], 0.82)

        # 3. Third frame - score 0.5 -> EMA becomes 0.8 * 0.82 + 0.2 * 0.5 = 0.756 < 0.8 -> transitions to NORMAL
        process_behavioral_smoothing(1, 0.5, config)
        self.assertEqual(tracked_people[1]["state"], "NORMAL")
        self.assertAlmostEqual(tracked_people[1]["anomaly_score_ema"], 0.756)


if __name__ == "__main__":
    unittest.main()
