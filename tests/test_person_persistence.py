import unittest
from types import SimpleNamespace

from src.state_machine import reset_state, tracked_people
from src.tracking import update_tracking_states


def _person_result(track_id, x, y, w=50.0, h=120.0):
    return [
        SimpleNamespace(
            boxes=SimpleNamespace(
                xywh=[[x, y, w, h]],
                conf=[0.9],
                cls=[0],
                id=[track_id],
            ),
            names={0: "person"},
        )
    ]


class PersonPersistenceTests(unittest.TestCase):
    """A brief YOLO detection dropout should not erase the last-known box a
    person's overlay needs to keep rendering (see bugs_and_debugs.txt: people
    never got the same persistence treatment bags did, so a single missed
    frame made an otherwise-still-tracked person's box vanish outright).
    """

    def test_last_known_box_survives_a_detection_dropout(self):
        reset_state()
        update_tracking_states(_person_result(42, 100.0, 100.0, w=55.0, h=130.0), polygon_vertices=[])

        self.assertEqual(tracked_people[42]["last_known_center"], (100.0, 100.0))
        self.assertEqual(tracked_people[42]["last_known_width"], 55.0)
        self.assertEqual(tracked_people[42]["last_known_height"], 130.0)

        # Frame where YOLO fails to detect her at all -- last-known box must remain.
        update_tracking_states([], polygon_vertices=[])

        self.assertIn(42, tracked_people)
        self.assertEqual(tracked_people[42]["last_known_center"], (100.0, 100.0))
        self.assertEqual(tracked_people[42]["last_known_width"], 55.0)
        self.assertEqual(tracked_people[42]["last_known_height"], 130.0)
        self.assertEqual(tracked_people[42]["stale_counter"], 1)

    def test_last_known_box_updates_when_redetected(self):
        reset_state()
        update_tracking_states(_person_result(7, 100.0, 100.0), polygon_vertices=[])
        update_tracking_states([], polygon_vertices=[])  # one missed frame
        update_tracking_states(_person_result(7, 140.0, 110.0, w=60.0, h=140.0), polygon_vertices=[])

        self.assertEqual(tracked_people[7]["last_known_center"], (140.0, 110.0))
        self.assertEqual(tracked_people[7]["last_known_width"], 60.0)
        self.assertEqual(tracked_people[7]["stale_counter"], 0)


if __name__ == "__main__":
    unittest.main()
