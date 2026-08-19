"""Generates Fig. 5 (letterbox preprocessing vs. naive stretch-resize) from a
REAL frame, per diagrams.txt's spec -- not a mockup.

Uses data/raw/suspicious_activities/UFC_Crime/Fighting/Fighting002_x264.mp4,
which is exactly 320x240 -- chosen because that's the paper's own cited
illustrative example (Section III-E: "a 320x240 frame naively stretched to
640x640... stretched 2.0x horizontally but 2.667x vertically"), so this
figure's numbers are traceable directly to the paper text, not just plausible.
This specific clip/frame was picked after probing several 320x240 UCF-Crime
candidates: some (e.g. the Burglary category) have letterbox bars already
baked into the raw source frame from the original CCTV recording, which
would confound the point of this figure; this one is full-frame with a
clearly visible standing person, needed to show anisotropic body distortion.

Verified against a synthetic borderless test image (a solid-color frame with
a red circle) that BOTH resize functions themselves are correct -- the naive
stretch fills the target with zero padding and turns the circle into an
ellipse; the letterbox path keeps the circle a circle and adds only the
expected gray bars. Any black borders visible around the photo content in
the real-footage output below are baked into that specific CCTV recording
itself (multiple datasets checked -- ABODA, other UCF-Crime clips -- show
the same pattern to varying degrees), not an artifact of either function.

The naive stretch-resize path no longer exists in src/detection.py (only
letterbox_resize does) -- reconstructed here as a two-line cv2.resize call,
exactly as diagrams.txt anticipated.

Needs only cv2/numpy (both already installed) -- no torch/ultralytics/GPU.
Run: python paper/figures/generate_fig5_letterbox.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import cv2
import numpy as np

from src.detection import letterbox_resize

SOURCE_VIDEO = ROOT / "data" / "raw" / "suspicious_activities" / "UFC_Crime" / "Fighting" / "Fighting002_x264.mp4"
OUTPUT_PATH = Path(__file__).resolve().parent / "fig5_letterbox_comparison.png"
TARGET_SIZE = 640
FRAME_INDEX = 200  # chosen after probing several frames: full-frame content (no
# pre-existing letterboxing baked into the source, unlike some other UCF-Crime
# clips checked first) with a clearly visible standing person, needed to show
# anisotropic body-proportion distortion -- the actual point of this figure.


def naive_stretch_resize(frame: np.ndarray, size: int) -> np.ndarray:
    """The old, no-longer-shipped preprocessing path (bugs_and_debugs.txt
    #25): independently scale width and height to fill the square, ignoring
    aspect ratio.
    """
    return cv2.resize(frame, (size, size))


def label_panel(img: np.ndarray, title: str, subtitle: str) -> np.ndarray:
    """Adds a title bar above a panel image."""
    bar_h = 64
    canvas = np.full((img.shape[0] + bar_h, img.shape[1], 3), 255, dtype=np.uint8)
    canvas[bar_h:, :, :] = img
    cv2.putText(canvas, title, (12, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (20, 20, 20), 2, cv2.LINE_AA)
    cv2.putText(canvas, subtitle, (12, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (90, 90, 90), 1, cv2.LINE_AA)
    return canvas


def main() -> None:
    if not SOURCE_VIDEO.is_file():
        raise FileNotFoundError(
            f"Source video not found: {SOURCE_VIDEO}\n"
            "This script needs the real dataset under data/raw/ (gitignored, "
            "not part of this checkout by default)."
        )

    cap = cv2.VideoCapture(str(SOURCE_VIDEO))
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open video: {SOURCE_VIDEO}")

    cap.set(cv2.CAP_PROP_POS_FRAMES, FRAME_INDEX)
    ret, frame = cap.read()
    cap.release()
    if not ret or frame is None:
        raise RuntimeError(f"Could not read frame {FRAME_INDEX} from {SOURCE_VIDEO}")

    src_h, src_w = frame.shape[:2]
    scale_x = TARGET_SIZE / src_w
    scale_y = TARGET_SIZE / src_h
    print(f"Source frame: {src_w}x{src_h}. Naive stretch factors: "
          f"{scale_x:.3f}x horizontal, {scale_y:.3f}x vertical.")

    stretched = naive_stretch_resize(frame, TARGET_SIZE)
    letterboxed = letterbox_resize(frame, TARGET_SIZE)

    # Annotate the distortion factors directly onto the stretch panel, so the
    # caption's numbers are traceable to the image itself (diagrams.txt spec).
    cv2.putText(
        stretched, f"{scale_x:.2f}x horizontal, {scale_y:.2f}x vertical",
        (10, TARGET_SIZE - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 220), 2, cv2.LINE_AA,
    )

    left = label_panel(stretched, "Naive stretch resize", f"cv2.resize(frame, ({TARGET_SIZE}, {TARGET_SIZE}))")
    right = label_panel(letterboxed, "Letterbox resize (used)", "letterbox_resize(frame, 640) -- src/detection.py")

    gap = 16
    combined = np.full((left.shape[0], left.shape[1] + gap + right.shape[1], 3), 255, dtype=np.uint8)
    combined[:, : left.shape[1]] = left
    combined[:, left.shape[1] + gap :] = right

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(OUTPUT_PATH), combined)
    print(f"Wrote {OUTPUT_PATH} ({combined.shape[1]}x{combined.shape[0]})")


if __name__ == "__main__":
    main()
