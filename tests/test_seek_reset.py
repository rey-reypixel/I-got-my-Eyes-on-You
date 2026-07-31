import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock

from src.detection import MultiModelDetectionEngine
from src.state_machine import reset_state, tracked_bags, tracked_people


class SeekResetTests(unittest.TestCase):
    """A manual rewind/skip (see main.py's 'r'/'f' hotkeys, src/detection.py's
    run() loop) jumps the video capture position discontinuously. Both
    ByteTrack's own internal state (model.track(..., persist=True) assumes
    continuous, monotonically-increasing frames) and our own tracking layer
    (stale_counter, last_known_* ghost positions, accumulated WARNING/ABANDONED
    timers) are built on that same continuity assumption. Without a reset,
    bounding boxes and their persistence go visibly incoherent after any
    rewind/skip: ghost boxes frozen at positions from the wrong point in time,
    IDs and states left over from a segment the video no longer shows.
    """

    def setUp(self):
        reset_state()
        self.engine = MultiModelDetectionEngine(source="dummy.mp4", model_path="dummy.pt")

    def test_seek_clears_tracked_people_and_bags(self):
        tracked_people[1] = {"state": "NORMAL", "stale_counter": 0}
        tracked_bags[5] = {"state": "ABANDONED", "stale_counter": 0}

        self.engine._reset_after_seek()

        self.assertEqual(tracked_people, {})
        self.assertEqual(tracked_bags, {})

    def test_seek_resets_the_underlying_bytetrack_instance(self):
        mock_tracker = MagicMock()
        self.engine.model = SimpleNamespace(predictor=SimpleNamespace(trackers=[mock_tracker]))

        self.engine._reset_after_seek()

        mock_tracker.reset.assert_called_once()

    def test_seek_is_safe_before_any_model_has_been_loaded(self):
        self.assertIsNone(self.engine.model)
        self.engine._reset_after_seek()  # must not raise

    def test_seek_is_safe_before_the_predictor_has_any_trackers_yet(self):
        self.engine.model = SimpleNamespace(predictor=SimpleNamespace())
        self.engine._reset_after_seek()  # must not raise (no .track() call has run yet)


if __name__ == "__main__":
    unittest.main()
