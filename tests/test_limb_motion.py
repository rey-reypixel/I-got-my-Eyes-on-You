import unittest

from src.state_machine import reset_state, tracked_people
from src.tracking import compute_limb_motion_score, get_pose_gated_person_ids


def _make_keypoints(shoulder_l=(90.0, 100.0), shoulder_r=(110.0, 100.0),
                     hip_l=(92.0, 140.0), hip_r=(108.0, 140.0),
                     elbow_l=(80.0, 120.0), elbow_r=(120.0, 120.0),
                     wrist_l=(75.0, 130.0), wrist_r=(125.0, 130.0),
                     conf=0.9):
    """Builds a 17-keypoint COCO-pose-shaped structure with the torso/limb
    points set explicitly and everything else defaulted, all at uniform
    confidence unless overridden by the caller.
    """
    xy = [(0.0, 0.0)] * 17
    cf = [conf] * 17
    xy[5], xy[6] = shoulder_l, shoulder_r
    xy[11], xy[12] = hip_l, hip_r
    xy[7], xy[8] = elbow_l, elbow_r
    xy[9], xy[10] = wrist_l, wrist_r
    return xy, cf


class PoseGatingTests(unittest.TestCase):
    def test_get_pose_gated_person_ids_respects_threshold(self):
        reset_state()
        tracked_people[1] = {"stationary_counter": 15}
        tracked_people[2] = {"stationary_counter": 3}
        config = {"limb_motion": {"stationary_frames_gate": 12}}

        gated = get_pose_gated_person_ids(config)
        self.assertIn(1, gated)
        self.assertNotIn(2, gated)

    def test_get_pose_gated_person_ids_uses_default_gate(self):
        reset_state()
        tracked_people[5] = {"stationary_counter": 12}
        gated = get_pose_gated_person_ids(None)
        self.assertIn(5, gated)

    def test_stale_person_is_excluded_even_with_a_qualifying_counter(self):
        # Regression test: a person who stopped being actively detected keeps
        # whatever stationary_counter they had when they vanished -- that value
        # is frozen, not evidence they're still there. Only a currently active
        # person (stale_counter == 0) should be considered gate-eligible.
        reset_state()
        tracked_people[1] = {"stationary_counter": 40, "stale_counter": 30}
        tracked_people[2] = {"stationary_counter": 15, "stale_counter": 0}
        config = {"limb_motion": {"stationary_frames_gate": 12}}

        gated = get_pose_gated_person_ids(config)
        self.assertNotIn(1, gated)
        self.assertIn(2, gated)


class LimbMotionScoreTests(unittest.TestCase):
    def setUp(self):
        reset_state()
        tracked_people[1] = {}
        self.config = {
            "limb_motion": {
                "keypoint_min_confidence": 0.5,
                "min_keypoints_required": 2,
                "variance_scale": 0.02,
            },
            "behavioral_smoothing": {"window_size": 10},
        }

    def test_returns_none_when_torso_unreliable(self):
        xy, cf = _make_keypoints(conf=0.1)  # everything below min_conf
        score = compute_limb_motion_score(1, xy, cf, norm_h=100.0, state_machine_config=self.config)
        self.assertIsNone(score)

    def test_returns_none_when_limb_keypoints_unreliable(self):
        xy, cf = _make_keypoints(conf=0.9)
        # Torso stays confident, but zero-out limb keypoint confidences.
        cf[7] = cf[8] = cf[9] = cf[10] = 0.1
        score = compute_limb_motion_score(1, xy, cf, norm_h=100.0, state_machine_config=self.config)
        self.assertIsNone(score)

    def test_returns_none_until_enough_history(self):
        xy, cf = _make_keypoints()
        self.assertIsNone(compute_limb_motion_score(1, xy, cf, 100.0, self.config))
        self.assertIsNone(compute_limb_motion_score(1, xy, cf, 100.0, self.config))
        # Third sample completes the minimum window for a variance calculation.
        score = compute_limb_motion_score(1, xy, cf, 100.0, self.config)
        self.assertIsNotNone(score)

    def test_still_limbs_score_lower_than_thrashing_limbs(self):
        still_xy, still_cf = _make_keypoints()
        for _ in range(6):
            still_score = compute_limb_motion_score(1, still_xy, still_cf, 100.0, self.config)

        tracked_people[2] = {}
        alt_a, cf_a = _make_keypoints(wrist_l=(40.0, 90.0), wrist_r=(160.0, 90.0), elbow_l=(60.0, 80.0), elbow_r=(140.0, 80.0))
        alt_b, cf_b = _make_keypoints(wrist_l=(75.0, 130.0), wrist_r=(125.0, 170.0), elbow_l=(80.0, 120.0), elbow_r=(120.0, 160.0))
        thrash_score = None
        for i in range(6):
            frame_xy, frame_cf = (alt_a, cf_a) if i % 2 == 0 else (alt_b, cf_b)
            thrash_score = compute_limb_motion_score(2, frame_xy, frame_cf, 100.0, self.config)

        self.assertIsNotNone(still_score)
        self.assertIsNotNone(thrash_score)
        self.assertLess(still_score, thrash_score)
        self.assertAlmostEqual(still_score, 0.0, places=6)


if __name__ == "__main__":
    unittest.main()
