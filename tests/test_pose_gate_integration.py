import unittest
from types import SimpleNamespace

from src.detection import MultiModelDetectionEngine
from src.state_machine import reset_state, tracked_people


def _detection_results(person_id, x, y, w=50.0, h=120.0):
    return [
        SimpleNamespace(
            boxes=SimpleNamespace(
                xywh=[[x, y, w, h]],
                conf=[0.9],
                cls=[0],
                id=[person_id],
            ),
            names={0: "person"},
        )
    ]


def _pose_result(box_xy, keypoints_xy, keypoints_conf):
    return [
        SimpleNamespace(
            boxes=SimpleNamespace(xywh=[box_xy]),
            keypoints=SimpleNamespace(xy=[keypoints_xy], conf=[keypoints_conf]),
        )
    ]


def _keypoints(elbow_l=(80.0, 120.0), elbow_r=(120.0, 120.0),
                wrist_l=(75.0, 130.0), wrist_r=(125.0, 130.0)):
    xy = [(0.0, 0.0)] * 17
    cf = [0.9] * 17
    xy[5], xy[6] = (90.0, 100.0), (110.0, 100.0)
    xy[11], xy[12] = (92.0, 140.0), (108.0, 140.0)
    xy[7], xy[8] = elbow_l, elbow_r
    xy[9], xy[10] = wrist_l, wrist_r
    return xy, cf


class PoseGateIntegrationTests(unittest.TestCase):
    def setUp(self):
        reset_state()
        self.engine = MultiModelDetectionEngine(source="dummy.mp4", model_path="dummy.pt")

    def test_ungated_person_never_triggers_pose_inference(self):
        tracked_people[100] = {"stationary_counter": 0, "max_observed_height": 100.0}
        calls = []
        self.engine._run_pose_inference = lambda frame: calls.append(1) or []

        results = _detection_results(100, 100.0, 100.0)
        self.engine._process_limb_motion_gate(frame=object(), results=results)

        self.assertEqual(calls, [])
        self.assertNotIn("anomaly_score_ema", tracked_people[100])

    def test_gated_person_gets_matched_and_scored(self):
        tracked_people[100] = {"stationary_counter": 20, "max_observed_height": 100.0}
        results = _detection_results(100, 100.0, 100.0)

        still_xy, still_cf = _keypoints()
        self.engine._run_pose_inference = lambda frame: _pose_result([100.0, 100.0, 50.0, 120.0], still_xy, still_cf)

        # Needs a few frames before compute_limb_motion_score has enough history to return a score.
        for _ in range(4):
            self.engine._process_limb_motion_gate(frame=object(), results=results)

        self.assertIn("anomaly_score_ema", tracked_people[100])
        # Perfectly still limbs relative to torso -> ~0 variance -> not flagged suspicious.
        self.assertFalse(tracked_people[100]["limb_suspicious_flag"])

    def test_far_pose_detection_is_not_matched(self):
        tracked_people[100] = {"stationary_counter": 20, "max_observed_height": 100.0}
        results = _detection_results(100, 100.0, 100.0)

        xy, cf = _keypoints()
        # Pose box is 500px away -- well outside the matching radius.
        self.engine._run_pose_inference = lambda frame: _pose_result([600.0, 600.0, 50.0, 120.0], xy, cf)

        for _ in range(4):
            self.engine._process_limb_motion_gate(frame=object(), results=results)

        self.assertNotIn("anomaly_score_ema", tracked_people[100])


if __name__ == "__main__":
    unittest.main()
