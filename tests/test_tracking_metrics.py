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


if __name__ == "__main__":
    unittest.main()
