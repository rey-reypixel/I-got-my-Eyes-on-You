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
    """A bag classified as 'carried' gets a grace window (config
    bag_carried_dropout_grace_frames, shipped default 900 -- see
    bugs_and_debugs.txt #35) before eviction on dropout, then is deleted
    outright once that grace is exhausted (to stop its ghost lingering/
    following the wrong person -- a deliberate, pre-existing choice, see
    bugs_and_debugs.txt #32/#33 for why the grace window was added on top of
    it, and #35 for why the default grew from 1 frame to 900). That eviction
    path bypassed find_recoverable_bag_id entirely, because by the time a
    new/reused ID reacquired the same physical bag, there was nothing left to
    recover from (see bugs_and_debugs.txt #29's follow-up). These tests pin
    the grace window to a small explicit value via state_machine_config so
    the mechanism itself -- absorb within grace, evict once exhausted, reset
    on reacquisition -- stays fast and precise regardless of what the shipped
    default happens to be.
    """

    def test_carried_bag_survives_a_single_frame_miss_within_the_grace_window(self):
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

        # Bag 5 is missing this frame entirely, but this is only its first
        # consecutive miss -- within a 1-frame grace window, so it must not
        # be evicted yet.
        update_tracking_states(
            [], polygon_vertices=[], state_machine_config={"bag_carried_dropout_grace_frames": 1}
        )

        self.assertIn(5, tracked_bags)
        self.assertNotIn(5, recently_evicted_bags)
        self.assertEqual(tracked_bags[5]["carried_miss_streak"], 1)

    def test_carried_bag_dropout_stashes_an_echo_once_grace_is_exhausted(self):
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
        config = {"bag_carried_dropout_grace_frames": 1}

        # First miss: absorbed by the grace window (see test above). No owner
        # in frame, so process_abandoned_logic still ticks ATTENDED -> WARNING
        # against the frozen position -- the grace window holds the box, it
        # doesn't freeze the state machine.
        update_tracking_states([], polygon_vertices=[], state_machine_config=config)
        self.assertIn(5, tracked_bags)
        self.assertEqual(tracked_bags[5]["state"], "WARNING")

        # Second consecutive miss: grace exhausted -- now it's evicted, but
        # should leave a recoverable echo (carrying over the WARNING/timer it
        # had already accumulated) instead of vanishing without a trace.
        update_tracking_states([], polygon_vertices=[], state_machine_config=config)

        self.assertNotIn(5, tracked_bags)
        self.assertIn(5, recently_evicted_bags)
        self.assertEqual(recently_evicted_bags[5]["state"], "WARNING")
        self.assertEqual(recently_evicted_bags[5]["timer"], 1)
        self.assertEqual(recently_evicted_bags[5]["center_coords"], (100.0, 100.0))

    def test_carried_bag_reacquired_mid_grace_resets_the_miss_streak(self):
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
        config = {"bag_carried_dropout_grace_frames": 1}

        update_tracking_states([], polygon_vertices=[], state_machine_config=config)  # 1 miss, within grace
        # Redetected again (still riding on the owner) before grace ran out.
        update_tracking_states(_bag_result(5, 101.0, 99.0), polygon_vertices=[], state_machine_config=config)

        self.assertIn(5, tracked_bags)
        self.assertEqual(tracked_bags[5]["carried_miss_streak"], 0)

        # Now it can absorb another full single-frame miss again, proving the
        # streak actually reset instead of continuing to accumulate.
        update_tracking_states([], polygon_vertices=[], state_machine_config=config)
        self.assertIn(5, tracked_bags)
        self.assertNotIn(5, recently_evicted_bags)

    def test_shipped_default_evicts_a_genuinely_carried_bag_quickly(self):
        # Regression guard for bugs_and_debugs.txt #36: a bag genuinely being
        # carried, that then leaves with its owner (both vanish together), must
        # not linger as a stale "ABANDONED (PERSISTED)" ghost for hundreds of
        # frames -- reported from real footage (AVSS 2007 AVSSS07_EASY.mpg).
        # The shipped grace default is small specifically so this resolves in a
        # handful of frames, not the 900-frame window used for genuinely placed
        # bags (see test_bag_persistence.py and the motion-stability tests below
        # for why placed bags don't need this path at all anymore).
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

        for _ in range(20):  # comfortably past the small shipped grace default
            update_tracking_states([], polygon_vertices=[])

        self.assertNotIn(5, tracked_bags)
        self.assertIn(5, recently_evicted_bags)

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


def _mixed_result(people, bags):
    """One frame's worth of person + bag detections, batched the way a real
    ultralytics Results object batches multiple boxes under one Boxes instance.
    """
    xywh, conf, cls, ids = [], [], [], []
    for (tid, x, y, w, h) in people:
        xywh.append([x, y, w, h])
        conf.append(0.9)
        cls.append(0)
        ids.append(tid)
    for (tid, x, y, w, h) in bags:
        xywh.append([x, y, w, h])
        conf.append(0.9)
        cls.append(24)
        ids.append(tid)
    return [
        SimpleNamespace(
            boxes=SimpleNamespace(xywh=xywh, conf=conf, cls=cls, id=ids),
            names={0: "person", 24: "backpack"},
        )
    ]


class DistantPersonCarriedMisfireTests(unittest.TestCase):
    """A bag's carried-vs-placed classification (`is_carried`) must require its
    reference person to actually be near the bag. Without that gate, whichever
    person is globally nearest -- even one far across the frame, well outside
    any plausible carrying distance -- still gets run through the vertical
    dy_centroid-vs-dy_feet check, and incidental perspective alignment (a bag
    placed farther back can sit level with a nearby person's torso rather than
    their feet) can flip `is_carried` to True for a bag nobody is touching.
    That misfire is fatal down the line: a bag believed "carried" is evicted
    outright (no persisted box, see CarriedBagEchoTests above) on its very
    next detection dropout -- exactly the left-behind scenario this check
    exists to protect (see bugs_and_debugs.txt #29's follow-up note flagging
    this as an unconfirmed suspicion, confirmed and fixed as #32).
    """

    def test_bag_far_from_the_only_people_in_frame_is_not_classified_as_carried(self):
        reset_state()
        # Two people together, well off to one side; a bag placed on the floor,
        # far from both -- nobody is anywhere near it, let alone carrying it.
        people = [(11, 330.0, 270.0, 90.0, 180.0), (12, 400.0, 270.0, 90.0, 180.0)]
        bags = [(50, 100.0, 260.0, 40.0, 50.0)]

        update_tracking_states(_mixed_result(people, bags), polygon_vertices=[])

        self.assertFalse(tracked_bags[50]["is_carried"])

    def test_left_behind_bag_survives_a_detection_dropout_instead_of_vanishing(self):
        reset_state()
        people = [(11, 330.0, 270.0, 90.0, 180.0), (12, 400.0, 270.0, 90.0, 180.0)]
        bags = [(50, 100.0, 260.0, 40.0, 50.0)]

        update_tracking_states(_mixed_result(people, bags), polygon_vertices=[])
        # Bag drops out of YOLO detection for a frame -- extremely common for a
        # small stationary object (bugs_and_debugs.txt #13/#19).
        update_tracking_states(_mixed_result(people, []), polygon_vertices=[])

        self.assertIn(50, tracked_bags)
        self.assertNotIn(50, recently_evicted_bags)

    def test_genuinely_carried_bag_close_to_its_owner_is_unaffected(self):
        reset_state()
        people = [(1, 300.0, 300.0, 60.0, 160.0)]
        bags = [(60, 310.0, 290.0, 35.0, 45.0)]  # riding at the owner's torso

        update_tracking_states(_mixed_result(people, bags), polygon_vertices=[])

        self.assertTrue(tracked_bags[60]["is_carried"])

    def test_person_bent_over_a_placed_bag_is_still_not_classified_as_carried(self):
        # Regression guard for bugs_and_debugs.txt #20: standing right next to
        # (or bending over) a placed bag must not flip is_carried, even though
        # the person is well within the new proximity gate.
        reset_state()
        people = [(2, 105.0, 250.0, 60.0, 100.0)]  # bent over: feet(ground)=300, centroid=250
        bags = [(61, 100.0, 280.0, 40.0, 40.0)]  # floor bag: own box center 20px above ground

        update_tracking_states(_mixed_result(people, bags), polygon_vertices=[])

        self.assertFalse(tracked_bags[61]["is_carried"])


class BagMotionStabilityTests(unittest.TestCase):
    """A bag riding along with a moving person displaces frame to frame; a bag
    resting on the floor, even right next to someone (passing the proximity
    gate above), does not. Reported from real footage (AVSS 2007
    AVSSS07_EASY.mpg): a bag that was genuinely just sitting there stayed
    misclassified 'carried' from vertical/perspective coincidence alone, which
    is fatal down the line -- a bag believed carried gets a much shorter
    dropout grace than a placed one (see CarriedBagEchoTests), so a
    stationary bag misclassified this way could be evicted outright instead
    of getting the long placed-bag persistence window it actually needs. See
    bugs_and_debugs.txt #36.
    """

    def test_bag_that_never_moves_gets_reclassified_as_placed_after_enough_stationary_frames(self):
        reset_state()
        people = [(1, 300.0, 300.0, 60.0, 160.0)]
        bags = [(60, 310.0, 290.0, 35.0, 45.0)]  # passes the vertical carried heuristic every frame

        for _ in range(5):  # bag_stationary_window (6) not yet full
            update_tracking_states(_mixed_result(people, bags), polygon_vertices=[])
        self.assertTrue(tracked_bags[60]["is_carried"])

        update_tracking_states(_mixed_result(people, bags), polygon_vertices=[])  # 6th identical frame
        self.assertFalse(tracked_bags[60]["is_carried"])

    def test_bag_that_moves_with_its_carrier_stays_classified_as_carried(self):
        reset_state()
        for step in range(6):
            people = [(1, 300.0 + step * 5.0, 300.0, 60.0, 160.0)]
            bags = [(60, 310.0 + step * 5.0, 290.0, 35.0, 45.0)]
            update_tracking_states(_mixed_result(people, bags), polygon_vertices=[])

        self.assertTrue(tracked_bags[60]["is_carried"])

    def test_carried_bag_that_leaves_frame_with_its_owner_is_evicted_quickly(self):
        # Regression guard for the AVSS report: a bag genuinely carried out of
        # frame together with its owner must not linger as a stale "ABANDONED
        # (PERSISTED)" ghost -- it should disappear within the small shipped
        # grace window, not survive for hundreds of frames.
        reset_state()
        for step in range(6):
            people = [(1, 300.0 + step * 5.0, 300.0, 60.0, 160.0)]
            bags = [(60, 310.0 + step * 5.0, 290.0, 35.0, 45.0)]
            update_tracking_states(_mixed_result(people, bags), polygon_vertices=[])
        self.assertTrue(tracked_bags[60]["is_carried"])

        # Both person and bag leave the frame together (genuinely carried away).
        for _ in range(10):  # comfortably past the small shipped grace default
            update_tracking_states([], polygon_vertices=[])

        self.assertNotIn(60, tracked_bags)


if __name__ == "__main__":
    unittest.main()
