import unittest
from types import SimpleNamespace

from src.state_machine import recently_evicted_bags, reset_state, tracked_bags
from src.tracking import find_recoverable_bag_id, update_tracking_states


def _bag_result(track_id, x, y, w=42.0, h=42.0):
    return [
        SimpleNamespace(
            boxes=SimpleNamespace(
                xywh=[[x, y, w, h]],
                conf=[0.9],
                cls=[24],
                id=[track_id],
            ),
            names={24: "backpack"},
        )
    ]


class BagIdentityRecoveryTests(unittest.TestCase):
    """Bags are stationary, so unlike people, a brand-new track ID appearing
    right where a recently-vanished one was is overwhelmingly likely to be the
    same physical bag reacquired under a new ID (ByteTrack has no Re-ID). This
    covers recovering its accumulated state instead of resetting to
    ATTENDED/timer=0 every time the underlying detection churns.
    """

    def test_new_id_near_a_vanished_bag_recovers_its_state_and_timer(self):
        reset_state()
        tracked_bags[5] = {
            "owner_id": 1,
            "center_coords": (100.0, 100.0),
            "width": 40.0,
            "height": 40.0,
            "state": "WARNING",
            "timer": 7,
            "stale_counter": 3,
            "is_carried": False,
        }

        update_tracking_states(_bag_result(9, 105.0, 98.0), polygon_vertices=[])

        self.assertNotIn(5, tracked_bags)
        self.assertIn(9, tracked_bags)
        self.assertEqual(tracked_bags[9]["state"], "WARNING")
        self.assertEqual(tracked_bags[9]["timer"], 8)  # 7 recovered + 1 more WARNING tick this frame
        self.assertEqual(tracked_bags[9]["owner_id"], 1)

    def test_new_id_far_from_any_vanished_bag_starts_fresh(self):
        reset_state()
        tracked_bags[5] = {
            "owner_id": 1,
            "center_coords": (100.0, 100.0),
            "width": 40.0,
            "height": 40.0,
            "state": "ABANDONED",
            "timer": 50,
            "stale_counter": 3,
            "is_carried": False,
        }

        update_tracking_states(_bag_result(9, 500.0, 500.0), polygon_vertices=[])

        self.assertIn(5, tracked_bags)  # untouched, not merged
        self.assertIn(9, tracked_bags)
        self.assertEqual(tracked_bags[9]["state"], "WARNING")  # fresh ATTENDED, immediately ticks to WARNING (no owner present)
        self.assertEqual(tracked_bags[9]["timer"], 1)

    def test_a_still_active_nearby_bag_is_not_treated_as_recoverable(self):
        # Isolates find_recoverable_bag_id directly: going through the full
        # update_tracking_states pipeline for this scenario is confounded by
        # the pre-existing (unrelated) duplicate-bag eviction, which also
        # merges any two bags within 80px regardless of active status --
        # that's a separate, already-existing mechanism, not what's being
        # tested here.
        reset_state()
        tracked_bags[5] = {
            "owner_id": 1,
            "center_coords": (100.0, 100.0),
            "width": 40.0,
            "height": 40.0,
            "state": "ABANDONED",
            "timer": 50,
            "stale_counter": 0,
            "is_carried": False,
        }

        recovered = find_recoverable_bag_id(105.0, 98.0, active_bag_ids={5, 9})

        self.assertIsNone(recovered)


class CarriedBagEchoTests(unittest.TestCase):
    """A bag classified as 'carried' is deleted outright the instant it drops
    out of detection (to stop its ghost lingering/following the wrong
    person -- a deliberate, pre-existing choice). That path bypassed
    find_recoverable_bag_id entirely, because by the time a new/reused ID
    reacquired the same physical bag, there was nothing left to recover from
    (see bugs_and_debugs.txt #29's follow-up). This covers the short-lived
    echo buffer that closes that gap.
    """

    def test_carried_bag_dropout_stashes_an_echo_instead_of_vanishing_outright(self):
        reset_state()
        tracked_bags[5] = {
            "owner_id": 1,
            "center_coords": (100.0, 100.0),
            "width": 42.0,
            "height": 42.0,
            "state": "ATTENDED",
            "timer": 0,
            "stale_counter": 0,
            "is_carried": True,
        }

        # Bag 5 is missing this frame entirely; it was classified as carried,
        # so it's evicted immediately -- but should leave a recoverable echo.
        update_tracking_states([], polygon_vertices=[])

        self.assertNotIn(5, tracked_bags)
        self.assertIn(5, recently_evicted_bags)
        self.assertEqual(recently_evicted_bags[5]["state"], "ATTENDED")
        self.assertEqual(recently_evicted_bags[5]["center_coords"], (100.0, 100.0))

    def test_new_bag_id_recovers_state_from_the_echo_buffer(self):
        reset_state()
        recently_evicted_bags[5] = {
            "center_coords": (100.0, 100.0),
            "state": "WARNING",
            "timer": 12,
            "owner_id": 2,
            "ttl": 20,
        }

        update_tracking_states(_bag_result(9, 103.0, 101.0), polygon_vertices=[])

        self.assertNotIn(5, recently_evicted_bags)
        self.assertIn(9, tracked_bags)
        self.assertEqual(tracked_bags[9]["state"], "WARNING")
        self.assertEqual(tracked_bags[9]["timer"], 13)  # 12 recovered + 1 more WARNING tick (no owner present)
        self.assertEqual(tracked_bags[9]["owner_id"], 2)

    def test_echo_expires_after_its_ttl(self):
        reset_state()
        recently_evicted_bags[5] = {
            "center_coords": (100.0, 100.0),
            "state": "WARNING",
            "timer": 5,
            "owner_id": None,
            "ttl": 1,
        }

        update_tracking_states([], polygon_vertices=[])

        self.assertNotIn(5, recently_evicted_bags)


if __name__ == "__main__":
    unittest.main()
