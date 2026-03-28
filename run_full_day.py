#!/usr/bin/env python3
"""Run CH19 + CH21 algorithms on a full day of NVR recordings.

Processes all video segments sequentially per camera, applies time offsets
so all events are relative to the start of the first file, and merges
results into a single JSON per camera.

Usage:
    python3 run_full_day.py
"""

import json
import os
import sys
import time
import re
import subprocess
from pathlib import Path
from datetime import datetime

BASE = Path(__file__).parent
DATA = BASE / "frames" / "New Long video data"
CUTTING_DIR = DATA / "Cutting"
PASSING_DIR = DATA / "Passing"

CH19_OUTPUT = BASE / "cutting_fullday.json"
CH21_OUTPUT = BASE / "blanket_fullday.json"


def sorted_videos(directory):
    """Return video files sorted by NVR timestamp in filename."""
    vids = sorted(directory.glob("NVR_*.mp4"))
    return [str(v) for v in vids]


def get_video_duration(path):
    """Get video duration in seconds using OpenCV."""
    import cv2
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        return 0
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return frames / fps


def run_ch19(videos):
    """Process all CH19 videos sequentially, merge with time offsets."""
    sys.path.insert(0, str(BASE))
    from cutting_counter import CuttingCounter

    all_events = []
    all_breaks = []
    all_frame_data = []
    total_duration = 0.0
    total_frames = 0
    total_processing = 0.0
    segment_info = []

    for i, video in enumerate(videos):
        print(f"\n{'='*70}")
        print(f"  CH19 SEGMENT {i+1}/{len(videos)}: {os.path.basename(video)}")
        print(f"  Time offset: {total_duration:.1f}s ({total_duration/60:.1f} min)")
        print(f"{'='*70}")

        counter = CuttingCounter(video)
        results = counter.run()

        seg_duration = results["metadata"]["duration_sec"]
        seg_frames = results["metadata"]["total_frames"]
        seg_processing = results["metadata"]["processing_time_sec"]

        # Offset events
        for evt in results["events"]:
            evt["time_sec"] = round(evt["time_sec"] + total_duration, 2)
            evt["frame"] = evt["frame"] + total_frames
            evt["segment"] = i
            all_events.append(evt)

        # Offset breaks
        for brk in results["breaks"]:
            brk["time_sec"] = round(brk["time_sec"] + total_duration, 2)
            brk["frame"] = brk["frame"] + total_frames
            brk["segment"] = i
            all_breaks.append(brk)

        # Offset frame_data
        for fd in results["frame_data"]:
            fd["time_sec"] = round(fd["time_sec"] + total_duration, 2)
            fd["frame"] = fd["frame"] + total_frames
            all_frame_data.append(fd)

        segment_info.append({
            "file": os.path.basename(video),
            "segment_index": i,
            "offset_sec": round(total_duration, 2),
            "duration_sec": round(seg_duration, 2),
            "frames": seg_frames,
            "cuts_detected": len(results["events"]),
            "processing_sec": round(seg_processing, 2),
        })

        total_duration += seg_duration
        total_frames += seg_frames
        total_processing += seg_processing

    # Compute merged summary
    total_cuts = len(all_events)
    break_time = sum(
        all_breaks[i+1]["time_sec"] - all_breaks[i]["time_sec"]
        for i in range(0, len(all_breaks) - 1, 2)
        if all_breaks[i]["type"] == "break_start" and i+1 < len(all_breaks)
        and all_breaks[i+1]["type"] == "break_end"
    )
    active_time = total_duration - break_time
    cuts_per_min = (total_cuts / active_time * 60) if active_time > 0 else 0

    cycle_times = []
    for i in range(1, len(all_events)):
        cycle_times.append(all_events[i]["time_sec"] - all_events[i-1]["time_sec"])

    import numpy as np
    avg_cycle = float(np.mean(cycle_times)) if cycle_times else 0

    high = sum(1 for c in all_events if c.get("confidence") == "high")
    med = sum(1 for c in all_events if c.get("confidence") == "medium")
    low = sum(1 for c in all_events if c.get("confidence") == "low")

    merged = {
        "metadata": {
            "type": "full_day",
            "total_videos": len(videos),
            "video_files": [os.path.basename(v) for v in videos],
            "fps": 25.0,
            "duration_sec": round(total_duration, 2),
            "total_frames": total_frames,
            "processing_time_sec": round(total_processing, 2),
            "version": "v5-robust",
            "generated_at": datetime.now().isoformat(),
        },
        "segments": segment_info,
        "config": {
            "note": "Same config as v5-robust, see cutting_counter.py"
        },
        "summary": {
            "total_cuts": total_cuts,
            "active_time_sec": round(active_time, 1),
            "break_time_sec": round(break_time, 1),
            "cuts_per_minute": round(cuts_per_min, 1),
            "avg_cycle_sec": round(avg_cycle, 1),
            "confidence_high": high,
            "confidence_medium": med,
            "confidence_low": low,
        },
        "events": all_events,
        "breaks": all_breaks,
        "frame_data": all_frame_data[::4],  # Sample every 4th for size
    }

    # First-hour sanity check
    first_hour_cuts = [e for e in all_events if e["time_sec"] <= 3600]
    print(f"\n{'='*70}")
    print(f"  CH19 FULL DAY RESULTS")
    print(f"{'='*70}")
    print(f"  Total duration: {total_duration:.0f}s ({total_duration/3600:.1f} hrs)")
    print(f"  Total cuts: {total_cuts}")
    print(f"  Active time: {active_time:.0f}s, Break time: {break_time:.0f}s")
    print(f"  Rate: {cuts_per_min:.1f} cuts/min")
    print(f"  Avg cycle: {avg_cycle:.1f}s")
    print(f"  Confidence: {high} high, {med} medium, {low} low")
    print(f"  Processing: {total_processing:.0f}s ({total_frames/total_processing:.0f} fps)")
    print(f"\n  SANITY CHECK — First hour: {len(first_hour_cuts)} cuts (v5 had 450)")
    print(f"{'='*70}")

    return merged


def run_ch21(videos):
    """Process all CH21 videos using the native multi-file support."""
    sys.path.insert(0, str(BASE))
    from blanket_counter import BlanketCounter

    all_results = []
    total_duration = 0.0
    total_processing = 0.0

    for i, video in enumerate(videos):
        print(f"\n{'='*70}")
        print(f"  CH21 SEGMENT {i+1}/{len(videos)}: {os.path.basename(video)}")
        print(f"  Time offset: {total_duration:.1f}s ({total_duration/60:.1f} min)")
        print(f"{'='*70}")

        start = time.time()
        counter = BlanketCounter(source=video)
        result = counter.run()
        elapsed = time.time() - start

        if result:
            # Add offset info
            result["time_offset_sec"] = round(total_duration, 2)
            result["segment_index"] = i

            # Offset all events
            for evt in result.get("events", []):
                evt["time_sec"] = round(evt["time_sec"] + total_duration, 2)
                if "frame" in evt:
                    evt["frame"] = evt["frame"]  # per-segment frame numbers

            # Offset frame data
            for fd in result.get("frames", []):
                fd["time_sec"] = round(fd["time_sec"] + total_duration, 2)

            seg_duration = result.get("video_info", {}).get("duration_sec", 0)
            total_duration += seg_duration
            total_processing += elapsed
            all_results.append(result)

    # Merge
    total_accepted = sum(r["results"]["accepted"] for r in all_results)
    total_rejected = sum(r["results"]["rejected"] for r in all_results)
    total_blankets = sum(r["results"]["total_blankets"] for r in all_results)
    total_weighed = sum(r["results"]["blankets_weighed"] for r in all_results)

    # Count table_blanket_off events
    all_events = []
    for r in all_results:
        all_events.extend(r.get("events", []))
    table_off_count = sum(1 for e in all_events if e.get("type") == "table_blanket_off")

    merged = {
        "generated_at": datetime.now().isoformat(),
        "metadata": {
            "type": "full_day",
            "total_videos": len(videos),
            "video_files": [os.path.basename(v) for v in [str(p) for p in videos]],
            "duration_sec": round(total_duration, 2),
            "processing_time_sec": round(total_processing, 2),
        },
        "videos": all_results,
        "total_blankets_weighed": total_weighed,
        "total_accepted": total_accepted,
        "total_rejected": total_rejected,
        "total_blankets": total_blankets,
        "total_table_blanket_off": table_off_count,
    }

    # First-hour sanity check
    first_hour_accepted = sum(
        1 for e in all_events
        if e.get("type") == "blanket_accepted" and e.get("time_sec", 9999) <= 3600
    )

    print(f"\n{'='*70}")
    print(f"  CH21 FULL DAY RESULTS")
    print(f"{'='*70}")
    print(f"  Total duration: {total_duration:.0f}s ({total_duration/3600:.1f} hrs)")
    print(f"  Accepted: {total_accepted}")
    print(f"  Rejected: {total_rejected}")
    print(f"  Total blankets: {total_blankets}")
    print(f"  Table blanket off: {table_off_count}")
    print(f"  Processing: {total_processing:.0f}s")
    print(f"\n  SANITY CHECK — First hour accepted: {first_hour_accepted} (v4 had 223)")
    print(f"{'='*70}")

    return merged


def main():
    import multiprocessing

    ch19_videos = sorted_videos(CUTTING_DIR)
    ch21_videos = sorted_videos(PASSING_DIR)

    print(f"CH19: {len(ch19_videos)} video files")
    for v in ch19_videos:
        print(f"  {os.path.basename(v)}")
    print(f"\nCH21: {len(ch21_videos)} video files")
    for v in ch21_videos:
        print(f"  {os.path.basename(v)}")
    print()

    ch19_result = None
    ch21_result = None

    if "--ch19-only" in sys.argv:
        ch19_result = run_ch19(ch19_videos)
    elif "--ch21-only" in sys.argv:
        ch21_result = run_ch21(ch21_videos)
    else:
        # Sequential both (run in separate terminals for parallel)
        ch19_result = run_ch19(ch19_videos)
        ch21_result = run_ch21(ch21_videos)

    # Save results
    if ch19_result:
        CH19_OUTPUT.write_text(json.dumps(ch19_result, indent=2))
        print(f"\nCH19 saved to: {CH19_OUTPUT}")
        print(f"  {ch19_result['summary']['total_cuts']} cuts over "
              f"{ch19_result['metadata']['duration_sec']/3600:.1f} hrs")

    if ch21_result:
        CH21_OUTPUT.write_text(json.dumps(ch21_result, indent=2))
        print(f"\nCH21 saved to: {CH21_OUTPUT}")
        print(f"  {ch21_result['total_accepted']} accepted, "
              f"{ch21_result['total_rejected']} rejected over "
              f"{ch21_result['metadata']['duration_sec']/3600:.1f} hrs")


if __name__ == "__main__":
    main()
