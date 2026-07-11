import csv
import tempfile
import unittest
from pathlib import Path

from metrics_report.summarize_metrics import compute_metrics


class EvaluateMetricsTests(unittest.TestCase):
    def test_compute_metrics_parses_csv_and_returns_summary_values(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "telemetry.csv"
            with csv_path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.writer(handle)
                writer.writerow(["frame_id", "object_id", "latency_seconds", "instantaneous_fps"])
                writer.writerow([1, 10, "1.0", "2.0"])
                writer.writerow([2, 11, "2.0", "3.0"])
                writer.writerow([3, 10, "1.5", "2.5"])

            metrics = compute_metrics(csv_path)

            self.assertAlmostEqual(metrics["avg_latency"], 1.5, places=6)
            self.assertAlmostEqual(metrics["avg_fps"], 2.5, places=6)
            self.assertEqual(metrics["unique_ids"], 2)
            self.assertEqual(metrics["fragmentation_errors"], 0)


if __name__ == "__main__":
    unittest.main()
