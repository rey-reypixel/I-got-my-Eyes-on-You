import tempfile
import unittest
from pathlib import Path

import numpy as np
from scipy.io import savemat

from benchmark.avenue_auc_eval import compute_auc_roc, load_frame_labels_from_mat


class LoadFrameLabelsFromMatTests(unittest.TestCase):
    def test_3d_mask_stack_any_nonzero_pixel_marks_frame_anomalous(self):
        # Shape (H, W, num_frames), matching this project's own confirmed
        # testing_vol/*.mat convention (vol01.mat: shape (120, 160, 1439)).
        # Frames 0,2 all-zero (normal); frame 1 has one nonzero pixel (anomalous).
        h, w, n = 4, 4, 3
        stack = np.zeros((h, w, n), dtype=np.uint8)
        stack[2, 2, 1] = 255

        with tempfile.TemporaryDirectory() as tmpdir:
            mat_path = Path(tmpdir) / "1_label.mat"
            savemat(str(mat_path), {"volLabel": stack})

            labels = load_frame_labels_from_mat(mat_path)

            self.assertEqual(list(labels), [0, 1, 0])

    def test_ignores_scipy_metadata_keys(self):
        h, w, n = 2, 2, 2
        stack = np.ones((h, w, n), dtype=np.uint8)  # every frame anomalous

        with tempfile.TemporaryDirectory() as tmpdir:
            mat_path = Path(tmpdir) / "2_label.mat"
            savemat(str(mat_path), {"gt_mask": stack})
            # savemat always adds __header__/__version__/__globals__ itself;
            # this test just confirms load_frame_labels_from_mat doesn't choke
            # on an arbitrary (non-"volLabel") key name, since the real
            # ground_truth_demo files' exact key name isn't independently
            # confirmed (see module docstring).
            labels = load_frame_labels_from_mat(mat_path)

            self.assertEqual(list(labels), [1, 1])


class ComputeAucRocTests(unittest.TestCase):
    def test_perfect_scores_yield_auc_one(self):
        labels = np.array([0, 0, 1, 1])
        scores = np.array([0.1, 0.2, 0.8, 0.9])
        self.assertAlmostEqual(compute_auc_roc(labels, scores), 1.0)

    def test_mismatched_lengths_are_truncated_not_errored(self):
        labels = np.array([0, 0, 1, 1, 1])
        scores = np.array([0.1, 0.2, 0.8, 0.9])  # one shorter
        # Should not raise; truncates to the shorter length (4) and computes
        # a valid AUC over that overlap rather than crashing on shape mismatch.
        auc = compute_auc_roc(labels, scores)
        self.assertGreaterEqual(auc, 0.0)
        self.assertLessEqual(auc, 1.0)

    def test_single_class_ground_truth_raises_value_error(self):
        labels = np.array([0, 0, 0, 0])
        scores = np.array([0.1, 0.2, 0.3, 0.4])
        with self.assertRaises(ValueError):
            compute_auc_roc(labels, scores)


if __name__ == "__main__":
    unittest.main()
