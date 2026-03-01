"""
blanket_tracker.py — Blanket Factory Production Counter
========================================================
Uses OpenCV background subtraction to extract per-frame motion data
across defined zones in factory camera footage.

USAGE:
  # Analyze one or more video files (camera auto-detected from filename):
  python blanket_tracker.py video1.mp4 video2.mp4

  # Override camera, specify output:
  python blanket_tracker.py --source path/to/video.mp4 --camera ch21 --output results.json

  # Connect to live RTSP stream:
  python blanket_tracker.py --source rtsp://user:pass@192.168.1.100/ch21 --camera ch21 --live

INSTALL:
  pip install opencv-python numpy
"""

import cv2
import numpy as np
import argparse
import json
import re
import sys
import time
from datetime import datetime
from pathlib import Path

# ─────────────────────────────────────────────────────────────────
# ZONE CONFIGURATION
# Rectangles are (x1, y1, x2, y2) in pixels for 1920x1080 frames.
# ─────────────────────────────────────────────────────────────────

ZONE_CONFIG = {
    "ch19": [
        {
            "name": "Cutting Table",
            "rect": (350, 100, 900, 380),
            "threshold": 0.04,
            "cooldown_sec": 2.0,
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
            "name": "Weighing Station",
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


def detect_camera_from_filename(filepath):
    """Try to detect camera ID (ch19/ch21) from the filename."""
    name = Path(filepath).name.lower()
    match = re.search(r'ch(\d+)', name)
    if match:
        cam_id = f"ch{match.group(1)}"
        if cam_id in ZONE_CONFIG:
            return cam_id
    return None


class BlanketTracker:
    def __init__(self, source, camera="ch21", live=False):
        self.source = source
        self.camera = camera.lower()
        self.live = live
        self.zones = ZONE_CONFIG.get(self.camera, ZONE_CONFIG["ch21"])

        # State
        self.counts = {z["name"]: 0 for z in self.zones}
        self.cooldown = {z["name"]: 0.0 for z in self.zones}
        self.prev_activity = {z["name"]: 0.0 for z in self.zones}
        self.events = []
        self.frame_data = []  # per-frame activity records
        self.frame_idx = 0

        # Background subtractor — MOG2 works well for indoor factory scenes
        self.bg_sub = cv2.createBackgroundSubtractorMOG2(
            history=100,
            varThreshold=40,
            detectShadows=False,
        )
        self.morph_kernel = np.ones((5, 5), np.uint8)

    def process_frame(self, frame, timestamp):
        """Process a single frame: compute foreground mask and per-zone activity."""
        fg = self.bg_sub.apply(frame)
        fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, self.morph_kernel)
        fg = cv2.dilate(fg, self.morph_kernel, iterations=2)

        # Global motion: fraction of all pixels with detected motion
        total_pixels = fg.shape[0] * fg.shape[1]
        global_activity = float(np.sum(fg > 0) / total_pixels) if total_pixels > 0 else 0.0

        zone_activities = {}
        for zone in self.zones:
            name = zone["name"]
            x1, y1, x2, y2 = zone["rect"]
            # Clamp to frame bounds
            x1c = max(0, min(x1, frame.shape[1]))
            y1c = max(0, min(y1, frame.shape[0]))
            x2c = max(0, min(x2, frame.shape[1]))
            y2c = max(0, min(y2, frame.shape[0]))

            zone_fg = fg[y1c:y2c, x1c:x2c]
            area = zone_fg.shape[0] * zone_fg.shape[1]
            activity = float(np.sum(zone_fg > 0) / area) if area > 0 else 0.0

            # Event detection (rising edge with cooldown)
            triggered = False
            if (activity > zone["threshold"]
                    and timestamp >= self.cooldown[name]
                    and self.prev_activity[name] < zone["threshold"]):

                self.counts[name] += 1
                self.cooldown[name] = timestamp + zone["cooldown_sec"]
                self.events.append({
                    "time_sec": round(timestamp, 3),
                    "frame": self.frame_idx,
                    "camera": self.camera,
                    "zone": name,
                    "activity": round(activity, 5),
                    "cumulative_count": self.counts[name],
                })
                triggered = True

                total = sum(self.counts.values())
                print(
                    f"  [frame {self.frame_idx:4d} t={timestamp:6.2f}s] "
                    f"EVENT  {self.camera.upper()} | {name:25s} | "
                    f"Activity: {activity:.4f} | "
                    f"Zone count: {self.counts[name]:3d} | "
                    f"Grand total: {total}"
                )

            self.prev_activity[name] = activity
            zone_activities[name] = round(activity, 5)

        # Record per-frame data
        frame_record = {
            "frame": self.frame_idx,
            "time_sec": round(timestamp, 3),
            "global_activity": round(global_activity, 5),
            "zones": zone_activities,
        }
        self.frame_data.append(frame_record)
        return frame_record

    def run(self):
        """Process the video source and collect per-frame motion data."""
        print(f"\n{'='*60}")
        print(f"  Blanket Tracker — Camera: {self.camera.upper()}")
        print(f"  Source: {self.source}")
        print(f"  Mode: {'LIVE STREAM' if self.live else 'VIDEO FILE'}")
        print(f"  Zones: {[z['name'] for z in self.zones]}")
        print(f"{'='*60}\n")

        cap = cv2.VideoCapture(self.source)
        if not cap.isOpened():
            print(f"ERROR: Cannot open source: {self.source}")
            return None

        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        print(f"  Resolution: {width}x{height}, FPS: {fps:.2f}, Frames: {total_frames}")
        print(f"  Duration: {total_frames / fps:.2f}s\n")

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
                self.process_frame(frame, timestamp)
                self.frame_idx += 1

                # Progress indicator every 50 frames
                if self.frame_idx % 50 == 0:
                    pct = (self.frame_idx / total_frames * 100) if total_frames > 0 else 0
                    print(f"  ... processed {self.frame_idx}/{total_frames} frames ({pct:.0f}%)")

        except KeyboardInterrupt:
            print("\nStopped by user.")

        finally:
            cap.release()

        # Build results
        result = {
            "camera": self.camera,
            "source": str(self.source),
            "video_info": {
                "width": width,
                "height": height,
                "fps": round(fps, 2),
                "total_frames": total_frames,
                "duration_sec": round(total_frames / fps, 3),
            },
            "zone_config": [
                {"name": z["name"], "rect": list(z["rect"]), "threshold": z["threshold"]}
                for z in self.zones
            ],
            "frames_processed": self.frame_idx,
            "blankets_counted": self.counts,
            "total_events": sum(self.counts.values()),
            "events": self.events,
            "frames": self.frame_data,
        }

        self.print_summary()
        return result

    def print_summary(self):
        print(f"\n{'='*60}")
        print(f"  FINAL COUNTS — {self.camera.upper()}")
        print(f"{'='*60}")
        for zone, count in self.counts.items():
            print(f"  {zone:30s} {count:5d}")
        print(f"  {'TOTAL':30s} {sum(self.counts.values()):5d}")
        print(f"  Frames processed: {self.frame_idx}")
        print(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Blanket Factory Production Counter — extracts per-frame motion data"
    )
    parser.add_argument(
        "videos", nargs="*",
        help="Video file paths (camera auto-detected from filename, e.g. NVR_ch19_...mp4)",
    )
    parser.add_argument("--source", help="Single video file path or RTSP URL")
    parser.add_argument(
        "--camera", choices=["ch19", "ch21"],
        help="Camera ID for zone config (auto-detected from filename if omitted)",
    )
    parser.add_argument("--live", action="store_true", help="Live stream mode")
    parser.add_argument("--output", default="blanket_activity.json", help="Output JSON file")
    args = parser.parse_args()

    # Collect all video sources to process
    jobs = []

    if args.source:
        cam = args.camera or detect_camera_from_filename(args.source) or "ch21"
        jobs.append((args.source, cam))

    for vid in (args.videos or []):
        cam = detect_camera_from_filename(vid)
        if cam is None:
            print(f"WARNING: Cannot detect camera from '{vid}', skipping. "
                  f"Use --source and --camera instead.")
            continue
        jobs.append((vid, cam))

    if not jobs:
        parser.print_help()
        print("\nERROR: No video sources specified.")
        sys.exit(1)

    # Process each video
    all_results = []
    for source, camera in jobs:
        tracker = BlanketTracker(source=source, camera=camera, live=args.live)
        result = tracker.run()
        if result:
            all_results.append(result)

    if not all_results:
        print("ERROR: No videos could be processed.")
        sys.exit(1)

    # Build output JSON
    output = {
        "generated_at": datetime.now().isoformat(),
        "cameras": all_results,
        "summary": {
            cam_result["camera"]: {
                "frames": cam_result["frames_processed"],
                "duration_sec": cam_result["video_info"]["duration_sec"],
                "total_events": cam_result["total_events"],
                "zone_counts": cam_result["blankets_counted"],
            }
            for cam_result in all_results
        },
    }

    Path(args.output).write_text(json.dumps(output, indent=2))
    print(f"\nResults saved to: {args.output}")
    print(f"  Total cameras processed: {len(all_results)}")
    for r in all_results:
        print(f"  {r['camera']}: {r['frames_processed']} frames, "
              f"{r['total_events']} events detected")


if __name__ == "__main__":
    main()
