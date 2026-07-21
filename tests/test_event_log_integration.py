import tempfile
import unittest
from pathlib import Path

from src import event_log
from src.detection import MultiModelDetectionEngine
from src.state_machine import reset_state, tracked_bags, tracked_people


class EngineEventLogIntegrationTests(unittest.TestCase):
    def setUp(self):
        reset_state()
        self._tmpdir = tempfile.TemporaryDirectory()
        self.db_path = Path(self._tmpdir.name) / "events.db"

        self.engine = MultiModelDetectionEngine(
            source="dummy.mp4", model_path="dummy.pt", event_log_path=str(self.db_path)
        )
        event_log.init_db(self.db_path)
        self.engine._event_run_id = event_log.start_run(self.db_path, "dummy.mp4")

    def tearDown(self):
        self._tmpdir.cleanup()

    def test_new_alert_gets_logged(self):
        tracked_people[1] = {"state": "INTRUDER"}
        self.engine._log_state_changes(frame_id=5)

        alerts = event_log.list_alerts(self.db_path, run_id=self.engine._event_run_id)
        self.assertEqual(len(alerts), 1)
        self.assertEqual(alerts[0]["entity_type"], "person")
        self.assertEqual(alerts[0]["entity_id"], 1)
        self.assertEqual(alerts[0]["state"], "INTRUDER")
        self.assertEqual(alerts[0]["frame_id"], 5)

    def test_unchanged_state_is_not_logged_again(self):
        tracked_people[1] = {"state": "INTRUDER"}
        self.engine._log_state_changes(frame_id=5)
        self.engine._log_state_changes(frame_id=6)  # still INTRUDER, no change
        self.engine._log_state_changes(frame_id=7)  # still INTRUDER, no change

        alerts = event_log.list_alerts(self.db_path, run_id=self.engine._event_run_id)
        self.assertEqual(len(alerts), 1)

    def test_recovery_to_normal_is_logged_but_not_returned_by_list_alerts(self):
        tracked_people[1] = {"state": "INTRUDER"}
        self.engine._log_state_changes(frame_id=5)
        tracked_people[1]["state"] = "NORMAL"
        self.engine._log_state_changes(frame_id=20)

        history = event_log.get_track_history(self.db_path, self.engine._event_run_id, "person", 1)
        self.assertEqual([h["state"] for h in history], ["INTRUDER", "NORMAL"])

        alerts = event_log.list_alerts(self.db_path, run_id=self.engine._event_run_id)
        self.assertEqual(len(alerts), 1)  # only INTRUDER, NORMAL is a recovery not an alert

    def test_bag_state_changes_are_logged_as_bag_entity_type(self):
        tracked_bags[9] = {"state": "ABANDONED"}
        self.engine._log_state_changes(frame_id=42)

        alerts = event_log.list_alerts(self.db_path, run_id=self.engine._event_run_id)
        self.assertEqual(len(alerts), 1)
        self.assertEqual(alerts[0]["entity_type"], "bag")
        self.assertEqual(alerts[0]["entity_id"], 9)

    def test_run_summary_reflects_logged_events(self):
        tracked_people[1] = {"state": "INTRUDER"}
        tracked_people[2] = {"state": "PANIC"}
        tracked_bags[9] = {"state": "ABANDONED"}
        self.engine._log_state_changes(frame_id=1)
        event_log.end_run(self.db_path, self.engine._event_run_id)

        summary = event_log.get_run_summary(self.db_path, self.engine._event_run_id)
        self.assertEqual(summary["unique_entity_counts"], {"person": 2, "bag": 1})
        self.assertEqual(summary["total_alert_events"], 3)


if __name__ == "__main__":
    unittest.main()
