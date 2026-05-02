#!/usr/bin/env python3
"""ROI overlay tool for the CH27 taping counter.

Reads the ROI constants from ``taping_counter.py`` and draws each rectangle on
top of a sample frame so the implementer can confirm placement before running
the full pipeline.

Usage:
    python3 taping_roi_calibrator.py /path/to/frame.png
    python3 taping_roi_calibrator.py /path/to/frame.png --output annotated.png
"""

import argparse
import os
import sys

import cv2

# Import the live ROI constants so the calibrator never drifts from production
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from taping_counter import (
    LEFT_TABLE_ROI,
    RIGHT_TABLE_ROI,
    HEAP_ROI,
    TAPE_DISPENSER_LEFT_ROI,
    TAPE_DISPENSER_RIGHT_ROI,
)

ROIS = [
    ("LEFT TABLE",  LEFT_TABLE_ROI,            (60, 220, 255)),   # orange
    ("RIGHT TABLE", RIGHT_TABLE_ROI,           (255, 180, 60)),   # cyan-ish
    ("HEAP",        HEAP_ROI,                  (180, 180, 180)),  # gray
    ("TAPE-L",      TAPE_DISPENSER_LEFT_ROI,   (180, 255, 180)),  # green
    ("TAPE-R",      TAPE_DISPENSER_RIGHT_ROI,  (180, 255, 180)),  # green
]


def annotate(frame_path, output_path):
    img = cv2.imread(frame_path)
    if img is None:
        sys.exit(f"ERROR: could not read {frame_path}")
    h, w = img.shape[:2]
    print(f"Frame: {w}x{h}")

    for name, roi, color in ROIS:
        x1, y1, x2, y2 = roi
        # Clip to frame
        x1c, y1c = max(0, x1), max(0, y1)
        x2c, y2c = min(w, x2), min(h, y2)
        cv2.rectangle(img, (x1c, y1c), (x2c, y2c), color, 3)
        label = f"{name} ({x1},{y1})-({x2},{y2})"
        # Background box for readability
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(img, (x1c, max(0, y1c - th - 6)), (x1c + tw + 6, y1c), color, -1)
        cv2.putText(img, label, (x1c + 3, y1c - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        print(f"  {name:12s}  ({x1:4d},{y1:4d}) - ({x2:4d},{y2:4d})  "
              f"size {x2-x1}x{y2-y1}")

    cv2.imwrite(output_path, img)
    print(f"\nWrote {output_path}")


def main():
    parser = argparse.ArgumentParser(description="CH27 ROI overlay tool")
    parser.add_argument("frame", help="Path to a sample frame (PNG/JPG)")
    parser.add_argument("--output", "-o", help="Output path (default: alongside input)")
    args = parser.parse_args()

    if args.output:
        out = args.output
    else:
        base, ext = os.path.splitext(args.frame)
        out = f"{base}_rois{ext or '.png'}"

    annotate(args.frame, out)


if __name__ == "__main__":
    main()
