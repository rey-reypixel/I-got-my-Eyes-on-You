import unittest

import numpy as np

from src.detection import letterbox_resize


class LetterboxResizeTests(unittest.TestCase):
    """Regression coverage for letterbox_resize (src/detection.py), which has
    had no test coverage despite being the fix for bugs_and_debugs.txt #25
    (naive stretch-resize anisotropically distorting human proportions on
    every non-square source video the pipeline had processed). Uses a
    synthetic circle rather than a real frame -- a circle staying a circle
    (vs. becoming an ellipse) is a direct, unambiguous test of aspect-ratio
    preservation, the exact property #25 was about.
    """

    def _circle_frame(self, w: int, h: int) -> np.ndarray:
        frame = np.zeros((h, w, 3), dtype=np.uint8)
        cx, cy, r = w // 2, h // 2, min(w, h) // 4
        yy, xx = np.ogrid[:h, :w]
        mask = (xx - cx) ** 2 + (yy - cy) ** 2 <= r * r
        frame[mask] = (0, 0, 255)
        return frame

    def _circle_bbox(self, gray_value: int, channel_frame: np.ndarray) -> tuple:
        """Returns (min_x, min_y, max_x, max_y) of red pixels in a BGR frame."""
        red_mask = (channel_frame[:, :, 2] > 200) & (channel_frame[:, :, 0] < 50)
        ys, xs = np.where(red_mask)
        return xs.min(), ys.min(), xs.max(), ys.max()

    def test_output_is_exactly_imgsz_square(self):
        frame = self._circle_frame(320, 240)
        out = letterbox_resize(frame, 640)
        self.assertEqual(out.shape[:2], (640, 640))

    def test_square_input_passed_through_unchanged(self):
        frame = self._circle_frame(640, 640)
        out = letterbox_resize(frame, 640)
        np.testing.assert_array_equal(out, frame)

    def test_padding_uses_neutral_gray_114(self):
        # 320x240 -> 640x640: scale=min(640/320,640/240)=2.0, new_w=640,
        # new_h=480, so there must be padding on the vertical axis.
        frame = self._circle_frame(320, 240)
        out = letterbox_resize(frame, 640)
        pad_row = out[0]  # top row is guaranteed to be padding for this input
        self.assertTrue((pad_row == 114).all())

    def test_aspect_ratio_preserved_circle_stays_circle(self):
        # This is the actual regression guard for #25: a circle must stay a
        # circle (bounding box width == height) under letterbox_resize, where
        # a naive cv2.resize(frame, (imgsz, imgsz)) would turn it into an
        # ellipse whenever the source isn't already square.
        frame = self._circle_frame(320, 240)
        out = letterbox_resize(frame, 640)
        x0, y0, x1, y1 = self._circle_bbox(200, out)
        width, height = x1 - x0, y1 - y0
        self.assertAlmostEqual(width, height, delta=2)

    def test_naive_stretch_would_have_distorted_the_same_input(self):
        # Contrast case: confirms the synthetic fixture actually exercises a
        # real distortion risk (i.e. this isn't a fixture that would pass
        # either way). A plain cv2.resize to (imgsz, imgsz) on the same
        # 320x240 input must turn the circle into a visibly non-square
        # (taller-than-wide) ellipse, matching the documented 2.0x/2.667x
        # distortion factors for this exact source size.
        import cv2

        frame = self._circle_frame(320, 240)
        stretched = cv2.resize(frame, (640, 640))
        x0, y0, x1, y1 = self._circle_bbox(200, stretched)
        width, height = x1 - x0, y1 - y0
        self.assertGreater(height, width * 1.2)


if __name__ == "__main__":
    unittest.main()
