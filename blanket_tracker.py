"""
blanket_tracker.py — Blanket Factory Production Counter
========================================================
Uses OpenCV background subtraction to count blanket processing events
across defined zones in your factory cameras.

USAGE:
  # Analyze a video file:
  python blanket_tracker.py --source path/to/video.mp4 --camera ch21

  # Connect to live RTSP stream:
  python blanket_tracker.py --source rtsp://user:pass@192.168.1.100/ch21 --camera ch21 --live

  # Run both cameras:
  python blanket_tracker.py --source rtsp://user:pass@NVR_IP/ch19 --camera ch19 --live &
  python blanket_tracker.py --source rtsp://user:pass@NVR_IP/ch21 --camera ch21 --live &

INSTALL:
  pip install opencv-python numpy
  (Optional for YOLOv8 upgrade): pip install ultralytics supervision
"""

import cv2
import numpy as np
import argparse
import json
import time
import sys
from datetime import datetime
from pathlib import Path

# ─────────────────────────────────────────────────────────────────
# ZONE CONFIGURATION
# Adjust these rectangles (x1, y1, x2, y2) to match your camera view
# Use VLC or any viewer to find pixel coordinates
# ─────────────────────────────────────────────────────────────────

ZONE_CONFIG = {
    "ch19": [
        {
            "name": "Cutting Table",
            "rect": (350, 100, 900, 380),
            "threshold": 0.04,       # fraction of zone pixels that must be active
            "cooldown_sec": 2.0,     # min seconds between counts in this zone
        },
        {
            "name": "Left Sorting Worker",
            "rect": (0, 580, 450, 800),
            "threshold": 0.03,
            "cooldown_sec": 2.0,
        },
        {
            "name": "Right Pile Area",
            "rect": (800, 300, 1300, 700),
            "threshold": 0.03,
            "cooldown_sec": 2.0,
        },
    ],
    "ch21": [
        {
            "name": "Weighing Station",     # ← BEST chokepoint for counting
            "rect": (900, 280, 1300, 680),
            "threshold": 0.04,
            "cooldown_sec": 2.0,
        },
        {
            "name": "Folding Table",
            "rect": (600, 200, 1000, 500),
            "threshold": 0.04,
            "cooldown_sec": 2.0,
        },
        {
            "name": "Sewing Machines",
            "rect": (0, 0, 350, 400),
            "threshold": 0.03,
            "cooldown_sec": 1.5,
        },
    ],
}


class BlanketTracker:
    def __init__(self, source, camera="ch21", live=False, output_log="counts.json"):
        self.source = source
        self.camera = camera.lower()
        self.live = live
        self.output_log = output_log
        self.zones = ZONE_CONFIG.get(self.camera, ZONE_CONFIG["ch21"])

        # State
        self.counts = {z["name"]: 0 for z in self.zones}
        self.cooldown = {z["name"]: 0.0 for z in self.zones}
        self.prev_activity = {z["name"]: 0.0 for z in self.zones}
        self.events = []
        self.start_time = None
        self.frame_idx = 0

        # Background subtractor — MOG2 works well for indoor factory scenes
        self.bg_sub = cv2.createBackgroundSubtractorMOG2(
            history=100,
            varThreshold=40,
            detectShadows=False
        )

        self.morph_kernel = np.ones((5, 5), np.uint8)

    def process_frame(self, frame, timestamp):
        """Process a single frame and update zone counts."""
        fg = self.bg_sub.apply(frame)
        fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, self.morph_kernel)
        fg = cv2.dilate(fg, self.morph_kernel, iterations=2)

        results = {}
        for zone in self.zones:
            name = zone["name"]
            x1, y1, x2, y2 = zone["rect"]
            zone_fg = fg[y1:y2, x1:x2]
            area = (x2 - x1) * (y2 - y1)
            activity = np.sum(zone_fg > 0) / area if area > 0 else 0.0

            triggered = False
            now = timestamp

            if (activity > zone["threshold"] and
                    now >= self.cooldown[name] and
                    self.prev_activity[name] < zone["threshold"]):

                self.counts[name] += 1
                self.cooldown[name] = now + zone["cooldown_sec"]
                event = {
                    "time": round(timestamp, 2),
                    "datetime": datetime.now().isoformat(),
                    "camera": self.camera,
                    "zone": name,
                    "activity": round(float(activity), 4),
                    "cumulative": self.counts[name],
                }
                self.events.append(event)
                triggered = True

                total = sum(self.counts.values())
                print(
                    f"  [{datetime.now().strftime('%H:%M:%S')}] "
                    f"EVENT  {self.camera.upper()} | {name:25s} | "
                    f"Activity: {activity:.3f} | "
                    f"Zone total: {self.counts[name]:3d} | "
                    f"Grand total: {total}"
                )

            self.prev_activity[name] = activity
            results[name] = {
                "activity": round(float(activity), 4),
                "count": self.counts[name],
                "triggered": triggered,
            }

        return fg, results

    def draw_overlay(self, frame, zone_results):
        """Draw zone rectangles and counts on frame."""
        overlay = frame.copy()
        for zone in self.zones:
            name = zone["name"]
            x1, y1, x2, y2 = zone["rect"]
            result = zone_results.get(name, {})
            activity = result.get("activity", 0)
            count = result.get("count", 0)
            triggered = result.get("triggered", False)

            # Color: green if active, amber if triggered, grey otherwise
            if triggered:
                color = (0, 165, 255)  # orange
                thickness = 3
            elif activity > zone["threshold"]:
                color = (0, 255, 120)  # green
                thickness = 2
            else:
                color = (80, 80, 120)   # muted
                thickness = 1

            cv2.rectangle(overlay, (x1, y1), (x2, y2), color, thickness)
            # Label
            label = f"{name}: {count}"
            cv2.rectangle(overlay, (x1, y1 - 22), (x1 + len(label)*9 + 6, y1), color, -1)
            cv2.putText(overlay, label, (x1 + 3, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)

        # Grand total
        total = sum(self.counts.values())
        cv2.rectangle(overlay, (10, 10), (300, 50), (20, 20, 40), -1)
        cv2.putText(overlay, f"BLANKETS: {total}  |  {self.camera.upper()}",
                    (20, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 255), 2, cv2.LINE_AA)

        return overlay

    def run(self):
        print(f"\n{'='*60}")
        print(f"  Blanket Tracker — Camera: {self.camera.upper()}")
        print(f"  Source: {self.source}")
        print(f"  Mode: {'LIVE STREAM' if self.live else 'VIDEO FILE'}")
        print(f"  Zones: {[z['name'] for z in self.zones]}")
        print(f"{'='*60}\n")

        cap = cv2.VideoCapture(self.source)
        if not cap.isOpened():
            print(f"ERROR: Cannot open source: {self.source}")
            sys.exit(1)

        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        self.start_time = time.time()

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    if self.live:
                        print("Stream lost — reconnecting in 3s...")
                        time.sleep(3)
                        cap = cv2.VideoCapture(self.source)
                        continue
                    else:
                        break

                timestamp = self.frame_idx / fps
                fg_mask, zone_results = self.process_frame(frame, timestamp)

                if not self.live:
                    # Show annotated frame for file analysis
                    annotated = self.draw_overlay(frame, zone_results)
                    cv2.imshow(f"Blanket Tracker — {self.camera.upper()}", annotated)
                    cv2.imshow(f"Motion Mask — {self.camera.upper()}", fg_mask)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        break
                else:
                    # Live mode: just print, no display needed
                    pass

                self.frame_idx += 1

        except KeyboardInterrupt:
            print("\nStopped by user.")

        finally:
            cap.release()
            cv2.destroyAllWindows()
            self.save_results()
            self.print_summary()

    def save_results(self):
        output = {
            "camera": self.camera,
            "source": str(self.source),
            "session_start": datetime.fromtimestamp(self.start_time).isoformat() if self.start_time else None,
            "session_end": datetime.now().isoformat(),
            "frames_processed": self.frame_idx,
            "blankets_counted": self.counts,
            "total": sum(self.counts.values()),
            "events": self.events,
        }
        Path(self.output_log).write_text(json.dumps(output, indent=2))
        print(f"\nResults saved to: {self.output_log}")

    def print_summary(self):
        print(f"\n{'='*60}")
        print(f"  FINAL COUNTS — {self.camera.upper()}")
        print(f"{'='*60}")
        for zone, count in self.counts.items():
            print(f"  {zone:30s} {count:5d}")
        print(f"  {'TOTAL':30s} {sum(self.counts.values()):5d}")
        print(f"{'='*60}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Blanket Factory Production Counter")
    parser.add_argument("--source", required=True, help="Video file path or RTSP URL")
    parser.add_argument("--camera", default="ch21", choices=["ch19", "ch21"],
                        help="Camera ID for zone config")
    parser.add_argument("--live", action="store_true", help="Live stream mode (no display)")
    parser.add_argument("--output", default="counts.json", help="Output JSON log file")
    args = parser.parse_args()

    tracker = BlanketTracker(
        source=args.source,
        camera=args.camera,
        live=args.live,
        output_log=args.output,
    )
    tracker.run()
