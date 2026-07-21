import unittest
from types import SimpleNamespace

from src.state_machine import reset_state, tracked_people
from src.tracking import resolve_person_state, update_tracking_states


def _make_result(track_id, x, y, w=20.0, h=20.0, label=0, conf=0.9):
    return [
        SimpleNamespace(
            boxes=SimpleNamespace(
                xywh=[[x, y, w, h]],
                conf=[conf],
                cls=[label],
                id=[track_id],
            ),
            names={0: "person"},
        )
    ]


class StateResolutionTests(unittest.TestCase):
    """Covers the state-machine bugs found in review: no priority between
    INTRUDER/PANIC/SUSPICIOUS_ACTIVITY, and no recovery path for PANIC or for
    kinematic-triggered SUSPICIOUS_ACTIVITY.
    """

    def test_intruder_takes_priority_over_panic_and_suspicious(self):
        person = {
            "intruder_flag": True,
            "panic_flag": True,
            "kinematic_suspicious_flag": True,
            "limb_suspicious_flag": True,
        }
        self.assertEqual(resolve_person_state(person), "INTRUDER")

    def test_panic_takes_priority_over_suspicious(self):
        person = {
            "intruder_flag": False,
            "panic_flag": True,
            "kinematic_suspicious_flag": True,
            "limb_suspicious_flag": False,
        }
        self.assertEqual(resolve_person_state(person), "PANIC")

    def test_suspicious_from_either_kinematic_or_limb_flag(self):
        self.assertEqual(resolve_person_state({"kinematic_suspicious_flag": True}), "SUSPICIOUS_ACTIVITY")
        self.assertEqual(resolve_person_state({"limb_suspicious_flag": True}), "SUSPICIOUS_ACTIVITY")

    def test_all_flags_false_is_normal(self):
        self.assertEqual(resolve_person_state({}), "NORMAL")

    def test_panic_recovers_after_cooldown_expires(self):
        reset_state()
        config = {"panic_recovery_frames": 2, "kinematic_velocity_threshold": 999.0}

        update_tracking_states(_make_result(7, 100.0, 100.0), polygon_vertices=[], state_machine_config=config)
        # Simulate a spike having latched panic on a previous frame.
        tracked_people[7]["panic_flag"] = True
        tracked_people[7]["panic_cooldown"] = 2
        tracked_people[7]["state"] = "PANIC"

        update_tracking_states(_make_result(7, 101.0, 100.0), polygon_vertices=[], state_machine_config=config)
        self.assertEqual(tracked_people[7]["panic_cooldown"], 1)
        self.assertTrue(tracked_people[7]["panic_flag"])
        self.assertEqual(tracked_people[7]["state"], "PANIC")

        update_tracking_states(_make_result(7, 102.0, 100.0), polygon_vertices=[], state_machine_config=config)
        self.assertEqual(tracked_people[7]["panic_cooldown"], 0)
        self.assertFalse(tracked_people[7]["panic_flag"])
        self.assertEqual(tracked_people[7]["state"], "NORMAL")

    def test_suspicious_activity_recovers_when_motion_settles(self):
        reset_state()
        config = {
            "kinematic_velocity_threshold": 0.05,
            "min_velocity_history_frames": 2,
            "imgsz": 200,
        }

        update_tracking_states(_make_result(42, 30.0, 30.0), polygon_vertices=[], state_machine_config=config)
        update_tracking_states(_make_result(42, 60.0, 60.0), polygon_vertices=[], state_machine_config=config)
        self.assertEqual(tracked_people[42]["state"], "SUSPICIOUS_ACTIVITY")

        # Person stops moving -- displacement drops back under threshold, so the
        # flag (and therefore the state) must recover on its own.
        for _ in range(3):
            update_tracking_states(_make_result(42, 60.0, 60.0), polygon_vertices=[], state_machine_config=config)

        self.assertEqual(tracked_people[42]["state"], "NORMAL")

    def test_intruder_alert_survives_a_simultaneous_panic_spike(self):
        # Regression test for bug #4: an intruder who also trips a kinematic
        # panic spike must not lose the unauthorized-access alert.
        reset_state()
        polygon = [(0.0, 0.0), (200.0, 0.0), (200.0, 200.0), (0.0, 200.0)]
        config = {"panic_threshold_devs": 0.001, "kinematic_velocity_threshold": 999.0}

        update_tracking_states(_make_result(9, 50.0, 50.0), polygon_vertices=polygon, state_machine_config=config)
        self.assertEqual(tracked_people[9]["state"], "INTRUDER")

        # Feed a burst of increasing displacement to try to trip verify_panic_anomaly.
        for i in range(1, 8):
            update_tracking_states(
                _make_result(9, 50.0 + i * 15.0, 50.0), polygon_vertices=polygon, state_machine_config=config
            )

        self.assertTrue(tracked_people[9]["intruder_flag"])
        self.assertEqual(tracked_people[9]["state"], "INTRUDER")


if __name__ == "__main__":
    unittest.main()
