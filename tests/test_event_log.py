import tempfile
import unittest
from pathlib import Path

from src.event_log import (
    ALERT_STATES,
    end_run,
    get_run_summary,
    get_track_history,
    init_db,
    list_alerts,
    list_runs,
    log_state_change,
    start_run,
)


class EventLogTests(unittest.TestCase):
    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        self.db_path = Path(self._tmpdir.name) / "events.db"
        init_db(self.db_path)

    def tearDown(self):
        self._tmpdir.cleanup()

    def test_start_run_returns_incrementing_ids_and_lists_runs(self):
        run_a = start_run(self.db_path, "video_a.mp4")
        run_b = start_run(self.db_path, "video_b.mp4")
        self.assertNotEqual(run_a, run_b)

        runs = list_runs(self.db_path)
        self.assertEqual(len(runs), 2)
        self.assertEqual({r["source"] for r in runs}, {"video_a.mp4", "video_b.mp4"})
        self.assertIsNone(runs[0]["ended_at"])

    def test_end_run_sets_ended_at(self):
        run_id = start_run(self.db_path, "video.mp4")
        end_run(self.db_path, run_id)
        runs = list_runs(self.db_path)
        self.assertIsNotNone(runs[0]["ended_at"])

    def test_list_alerts_defaults_to_alert_states_only(self):
        run_id = start_run(self.db_path, "video.mp4")
        log_state_change(self.db_path, run_id, frame_id=1, entity_type="person", entity_id=1, state="NORMAL")
        log_state_change(self.db_path, run_id, frame_id=2, entity_type="person", entity_id=1, state="INTRUDER")
        log_state_change(self.db_path, run_id, frame_id=3, entity_type="bag", entity_id=5, state="ABANDONED")

        alerts = list_alerts(self.db_path, run_id=run_id)
        states = {a["state"] for a in alerts}
        self.assertEqual(states, {"INTRUDER", "ABANDONED"})
        self.assertNotIn("NORMAL", states)

    def test_list_alerts_filters_by_specific_state_even_if_not_an_alert_state(self):
        run_id = start_run(self.db_path, "video.mp4")
        log_state_change(self.db_path, run_id, frame_id=1, entity_type="person", entity_id=1, state="NORMAL")
        log_state_change(self.db_path, run_id, frame_id=2, entity_type="person", entity_id=1, state="INTRUDER")

        alerts = list_alerts(self.db_path, run_id=run_id, state="NORMAL")
        self.assertEqual(len(alerts), 1)
        self.assertEqual(alerts[0]["state"], "NORMAL")

    def test_list_alerts_scoped_by_run(self):
        run_a = start_run(self.db_path, "video_a.mp4")
        run_b = start_run(self.db_path, "video_b.mp4")
        log_state_change(self.db_path, run_a, frame_id=1, entity_type="person", entity_id=1, state="PANIC")
        log_state_change(self.db_path, run_b, frame_id=1, entity_type="person", entity_id=1, state="INTRUDER")

        alerts_a = list_alerts(self.db_path, run_id=run_a)
        self.assertEqual(len(alerts_a), 1)
        self.assertEqual(alerts_a[0]["state"], "PANIC")

    def test_get_track_history_includes_recoveries_and_is_ordered(self):
        run_id = start_run(self.db_path, "video.mp4")
        log_state_change(self.db_path, run_id, frame_id=1, entity_type="person", entity_id=7, state="NORMAL")
        log_state_change(self.db_path, run_id, frame_id=10, entity_type="person", entity_id=7, state="SUSPICIOUS_ACTIVITY")
        log_state_change(self.db_path, run_id, frame_id=40, entity_type="person", entity_id=7, state="NORMAL")
        # A different entity in the same run must not leak into this history.
        log_state_change(self.db_path, run_id, frame_id=15, entity_type="person", entity_id=8, state="PANIC")

        history = get_track_history(self.db_path, run_id, "person", 7)
        self.assertEqual([h["state"] for h in history], ["NORMAL", "SUSPICIOUS_ACTIVITY", "NORMAL"])
        self.assertEqual([h["frame_id"] for h in history], [1, 10, 40])

    def test_get_run_summary_aggregates_state_and_entity_counts(self):
        run_id = start_run(self.db_path, "video.mp4")
        log_state_change(self.db_path, run_id, frame_id=1, entity_type="person", entity_id=1, state="INTRUDER")
        log_state_change(self.db_path, run_id, frame_id=2, entity_type="person", entity_id=2, state="PANIC")
        log_state_change(self.db_path, run_id, frame_id=3, entity_type="bag", entity_id=5, state="ABANDONED")
        end_run(self.db_path, run_id)

        summary = get_run_summary(self.db_path, run_id)
        self.assertEqual(summary["run"]["source"], "video.mp4")
        self.assertEqual(summary["state_counts"], {"INTRUDER": 1, "PANIC": 1, "ABANDONED": 1})
        self.assertEqual(summary["unique_entity_counts"], {"person": 2, "bag": 1})
        self.assertEqual(summary["total_alert_events"], 3)

    def test_get_run_summary_raises_for_unknown_run(self):
        with self.assertRaises(ValueError):
            get_run_summary(self.db_path, run_id=999)

    def test_alert_states_constant_matches_expected_set(self):
        self.assertEqual(set(ALERT_STATES), {"INTRUDER", "WARNING", "ABANDONED", "PANIC", "SUSPICIOUS_ACTIVITY"})


if __name__ == "__main__":
    unittest.main()
