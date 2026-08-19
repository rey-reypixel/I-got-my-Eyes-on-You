import tempfile
import unittest
from pathlib import Path

from benchmark.avss_ground_truth import _discover_avss_clips, load_ground_truth

# Minimal but schema-faithful ViPER-GT XML, matching the real AVSS 2007 EASY
# clip's structure (data/raw/abandoned_objects/AVSS 2007/AVSSS07_EASY.txt):
# one Information file descriptor plus PutObject/AbandonedObject events, each
# carrying a single static (dynamic="false") bounding box for its whole
# framespan.
SAMPLE_VIPER_XML = """<?xml version="1.0" encoding="UTF-8"?>
<viper xmlns="http://lamp.cfar.umd.edu/viper#" xmlns:data="http://lamp.cfar.umd.edu/viperdata#">
    <config>
        <descriptor name="Information" type="FILE">
            <attribute dynamic="false" name="NUMFRAMES" type="http://lamp.cfar.umd.edu/viperdata#dvalue"/>
            <attribute dynamic="false" name="FRAMERATE" type="http://lamp.cfar.umd.edu/viperdata#fvalue"/>
            <attribute dynamic="false" name="H-FRAME-SIZE" type="http://lamp.cfar.umd.edu/viperdata#dvalue"/>
            <attribute dynamic="false" name="V-FRAME-SIZE" type="http://lamp.cfar.umd.edu/viperdata#dvalue"/>
        </descriptor>
        <descriptor name="PutObject" type="OBJECT">
            <attribute dynamic="false" name="BoundingBox" type="http://lamp.cfar.umd.edu/viperdata#bbox"/>
        </descriptor>
        <descriptor name="AbandonedObject" type="OBJECT">
            <attribute dynamic="false" name="BoundingBox" type="http://lamp.cfar.umd.edu/viperdata#bbox"/>
        </descriptor>
    </config>
    <data>
        <sourcefile filename="SAMPLE.mpg">
            <file id="0" name="Information">
                <attribute name="NUMFRAMES"><data:dvalue value="1000"/></attribute>
                <attribute name="FRAMERATE"><data:fvalue value="25.0"/></attribute>
                <attribute name="H-FRAME-SIZE"><data:dvalue value="640"/></attribute>
                <attribute name="V-FRAME-SIZE"><data:dvalue value="480"/></attribute>
            </file>
            <object framespan="100:900" id="0" name="PutObject">
                <attribute name="BoundingBox">
                    <data:bbox height="50" width="30" x="200" y="300"/>
                </attribute>
            </object>
            <object framespan="250:900" id="0" name="AbandonedObject">
                <attribute name="BoundingBox">
                    <data:bbox height="50" width="30" x="200" y="300"/>
                </attribute>
            </object>
        </sourcefile>
    </data>
</viper>
"""


class AvssGroundTruthTests(unittest.TestCase):
    def test_parses_information_and_events_from_sample_xml(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            xml_path = Path(tmpdir) / "sample.txt"
            xml_path.write_text(SAMPLE_VIPER_XML, encoding="utf-8")

            gt = load_ground_truth(xml_path)

            self.assertEqual(gt.source_filename, "SAMPLE.mpg")
            self.assertEqual(gt.num_frames, 1000)
            self.assertEqual(gt.frame_rate, 25.0)
            self.assertEqual(gt.frame_size, (640, 480))
            self.assertEqual(len(gt.events), 2)

    def test_abandoned_object_event_has_correct_framespan_and_bbox(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            xml_path = Path(tmpdir) / "sample.txt"
            xml_path.write_text(SAMPLE_VIPER_XML, encoding="utf-8")

            gt = load_ground_truth(xml_path)
            abandoned = gt.events_named("AbandonedObject")

            self.assertEqual(len(abandoned), 1)
            event = abandoned[0]
            self.assertEqual(event.start_frame, 250)
            self.assertEqual(event.end_frame, 900)
            # x=200,y=300,width=30,height=50 -> (x1,y1,x2,y2)
            self.assertEqual(event.bbox, (200.0, 300.0, 230.0, 350.0))

    def test_put_object_event_precedes_abandoned_object_in_time(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            xml_path = Path(tmpdir) / "sample.txt"
            xml_path.write_text(SAMPLE_VIPER_XML, encoding="utf-8")

            gt = load_ground_truth(xml_path)
            put = gt.events_named("PutObject")[0]
            abandoned = gt.events_named("AbandonedObject")[0]

            self.assertLess(put.start_frame, abandoned.start_frame)

    def test_discover_avss_clips_pairs_same_stem_video_and_ground_truth(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            avss_dir = Path(tmpdir)
            (avss_dir / "CLIP_A.txt").write_text(SAMPLE_VIPER_XML, encoding="utf-8")
            (avss_dir / "CLIP_A.mpg").write_bytes(b"")
            (avss_dir / "CLIP_B.txt").write_text(SAMPLE_VIPER_XML, encoding="utf-8")
            # CLIP_B has no matching video file -- should be skipped, not crash.

            pairs = _discover_avss_clips(avss_dir)

            self.assertEqual(len(pairs), 1)
            video_path, xml_path = pairs[0]
            self.assertEqual(video_path.name, "CLIP_A.mpg")
            self.assertEqual(xml_path.name, "CLIP_A.txt")


class AvssRealDataSmokeTest(unittest.TestCase):
    """Optional smoke check against the real AVSS 2007 dataset, if present.

    data/raw/ is gitignored (external data, dropped in manually per
    data/READ.md) -- skipped gracefully on a fresh clone before the dataset
    is downloaded, rather than failing the suite.
    """

    AVSS_DIR = Path(__file__).resolve().parent.parent / "data" / "raw" / "abandoned_objects" / "AVSS 2007"

    @unittest.skipUnless(AVSS_DIR.is_dir(), "AVSS 2007 dataset not present under data/raw")
    def test_all_three_real_clips_parse_without_error(self):
        pairs = _discover_avss_clips(self.AVSS_DIR)
        self.assertEqual(len(pairs), 3, "expected EASY/MEDIUM/HARD clip+ground-truth pairs")
        for _video_path, xml_path in pairs:
            gt = load_ground_truth(xml_path)
            self.assertGreaterEqual(len(gt.events_named("AbandonedObject")), 1)


if __name__ == "__main__":
    unittest.main()
