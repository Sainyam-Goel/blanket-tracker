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
TAPING_DIR  = BASE / "Taping Cam27"   # NVR files are in subdirs (Tape/, Tape 2/, ...)

CH19_OUTPUT = BASE / "cutting_fullday.json"
CH19_OUTPUT_V6 = BASE / "cutting_fullday_v6.json"
CH21_OUTPUT = BASE / "blanket_fullday.json"
CH27_OUTPUT = BASE / "taping_fullday.json"        # primary (v2 by default)
CH27_OUTPUT_V1 = BASE / "taping_fullday_v1.json"  # legacy v1 for comparison


def sorted_videos(directory, recursive=False):
    """Return video files sorted by NVR timestamp in filename.
    If ``recursive``, also looks inside subdirectories (CH27 layout).
    """
    pattern = "**/NVR_*.mp4" if recursive else "NVR_*.mp4"
    vids = sorted(directory.glob(pattern), key=lambda p: p.name)
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


def run_ch19(videos, version="v5"):
    """Process all CH19 videos sequentially, merge with time offsets."""
    sys.path.insert(0, str(BASE))
    from cutting_counter import CuttingCounter

    all_events = []
    all_breaks = []
    all_frame_data = []
    all_suppressed = []
    total_duration = 0.0
    total_frames = 0
    total_processing = 0.0
    segment_info = []

    for i, video in enumerate(videos):
        print(f"\n{'='*70}")
        print(f"  CH19 SEGMENT {i+1}/{len(videos)}: {os.path.basename(video)}")
        print(f"  Time offset: {total_duration:.1f}s ({total_duration/60:.1f} min)")
        print(f"{'='*70}")

        counter = CuttingCounter(video, version=version)
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

        # Offset suppressed candidates (v6 audit log)
        for sc in results.get("suppressed_candidates", []):
            sc["time_sec"] = round(sc.get("time_sec", 0) + total_duration, 2)
            sc["segment"] = i
            all_suppressed.append(sc)

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
            "version": "v6-permissive" if version == "v6" else "v5-robust",
            "generated_at": datetime.now().isoformat(),
        },
        "segments": segment_info,
        "config": {
            "note": f"See cutting_counter.py ({version})"
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
        "suppressed_candidates": all_suppressed,
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


def run_ch27(videos, frame_step=1, version="v2"):
    """Process all CH27 taping videos sequentially, merge with time offsets.
    ``frame_step`` advances raw frames N at a time (HEVC decode optimization).
    v2 algorithm requires consecutive frames for air-motion frame-difference,
    so frame_step=1 is the safe default. v1 tolerates frame_step=2.
    """
    sys.path.insert(0, str(BASE))
    from taping_counter import TapingCounter

    all_events = []
    all_breaks = []
    all_suppressed = []
    all_frame_data = []
    all_heap_trace = []
    total_duration = 0.0
    total_frames = 0
    total_processing = 0.0
    segment_info = []

    for i, video in enumerate(videos):
        print(f"\n{'='*70}")
        print(f"  CH27 SEGMENT {i+1}/{len(videos)}: {os.path.basename(video)}")
        print(f"  Time offset: {total_duration:.1f}s ({total_duration/60:.1f} min)")
        print(f"{'='*70}")

        counter = TapingCounter(video, frame_step=frame_step, version=version)
        results = counter.run()

        seg_duration = results["metadata"]["duration_sec"]
        seg_frames = results["metadata"]["total_frames"]
        seg_processing = results["metadata"]["processing_time_sec"]

        # Offset events
        for evt in results["events"]:
            evt["time_sec"] = round(evt["time_sec"] + total_duration, 2)
            evt["cycle_start_sec"] = round(evt.get("cycle_start_sec", 0) + total_duration, 2)
            evt["frame"] = evt["frame"] + total_frames
            evt["segment"] = i
            all_events.append(evt)

        # Offset breaks
        for brk in results.get("breaks", []):
            brk["time_sec"] = round(brk["time_sec"] + total_duration, 2)
            brk["frame"] = brk["frame"] + total_frames
            brk["segment"] = i
            all_breaks.append(brk)

        # Offset suppressed candidates
        for sc in results.get("suppressed_candidates", []):
            sc["time_sec"] = round(sc["time_sec"] + total_duration, 2)
            sc["frame"] = sc["frame"] + total_frames
            sc["segment"] = i
            all_suppressed.append(sc)

        # Offset frame_data
        for fd in results.get("frame_data", []):
            fd["time_sec"] = round(fd["time_sec"] + total_duration, 2)
            fd["frame"] = fd["frame"] + total_frames
            all_frame_data.append(fd)

        # Offset heap_trace
        for ht in results.get("heap_trace", []):
            ht["time_sec"] = round(ht["time_sec"] + total_duration, 2)
            ht["frame"] = ht["frame"] + total_frames
            all_heap_trace.append(ht)

        L = sum(1 for e in results["events"] if e.get("table") == "left")
        R = sum(1 for e in results["events"] if e.get("table") == "right")
        segment_info.append({
            "file": os.path.basename(video),
            "segment_index": i,
            "offset_sec": round(total_duration, 2),
            "duration_sec": round(seg_duration, 2),
            "frames": seg_frames,
            "left_cycles": L,
            "right_cycles": R,
            "processing_sec": round(seg_processing, 2),
        })

        total_duration += seg_duration
        total_frames += seg_frames
        total_processing += seg_processing

    # Merged summary
    L_all = [e for e in all_events if e.get("table") == "left"]
    R_all = [e for e in all_events if e.get("table") == "right"]
    durations = [e["cycle_duration_sec"] for e in all_events]
    import numpy as np
    mean_dur = float(np.mean(durations)) if durations else 0.0
    med_dur = float(np.median(durations)) if durations else 0.0
    balance = (min(len(L_all), len(R_all)) / max(len(L_all), len(R_all))) if max(len(L_all), len(R_all)) > 0 else 1.0
    overlap = sum(1 for e in all_events if e.get("via_overlap_detector"))
    long_count = sum(1 for e in all_events if e.get("long_cycle"))

    merged = {
        "metadata": {
            "type": "full_day",
            "camera": "CH27",
            "total_videos": len(videos),
            "video_files": [os.path.basename(v) for v in videos],
            "fps": 25.0,
            "duration_sec": round(total_duration, 2),
            "total_frames": total_frames,
            "processing_time_sec": round(total_processing, 2),
            "version": "v1",
            "generated_at": datetime.now().isoformat(),
        },
        "segments": segment_info,
        "config": {
            "note": "See taping_counter.py V1_CONFIG"
        },
        "summary": {
            "total_cycles": len(all_events),
            "left_cycles": len(L_all),
            "right_cycles": len(R_all),
            "mean_cycle_sec": round(mean_dur, 2),
            "median_cycle_sec": round(med_dur, 2),
            "table_balance_ratio": round(balance, 3),
            "overlap_cycles": overlap,
            "long_cycles": long_count,
            "suppressed_count": len(all_suppressed),
        },
        "events": all_events,
        "breaks": all_breaks,
        "suppressed_candidates": all_suppressed,
        "frame_data": all_frame_data[::4],   # ~5Hz × ¼ ≈ 1.25Hz, ~40k samples for 9hr
        "heap_trace": all_heap_trace,
    }

    print(f"\n{'='*70}")
    print(f"  CH27 FULL DAY RESULTS")
    print(f"{'='*70}")
    print(f"  Total duration: {total_duration:.0f}s ({total_duration/3600:.2f} hrs)")
    print(f"  Total cycles:   {len(all_events)} (L={len(L_all)}, R={len(R_all)})")
    print(f"  Mean dur:       {mean_dur:.1f}s   Median: {med_dur:.1f}s")
    print(f"  Balance ratio:  {balance:.2f}")
    print(f"  Overlap cycles: {overlap}")
    print(f"  Long cycles:    {long_count}")
    print(f"  Suppressed:     {len(all_suppressed)}")
    print(f"  Processing:     {total_processing:.0f}s ({total_frames/total_processing:.0f} fps)")
    print(f"{'='*70}")

    return merged


def main():
    import multiprocessing

    # Parse --version flag (v5 default, v6 optional)
    version = "v5"
    if "--version" in sys.argv:
        idx = sys.argv.index("--version")
        if idx + 1 < len(sys.argv):
            version = sys.argv[idx + 1]
        if version not in ("v5", "v6"):
            print(f"ERROR: --version must be v5 or v6 (got {version})")
            sys.exit(1)

    ch19_output = CH19_OUTPUT_V6 if version == "v6" else CH19_OUTPUT
    print(f"CH19 variant: {version} → {ch19_output.name}")

    ch19_videos = sorted_videos(CUTTING_DIR) if CUTTING_DIR.exists() else []
    ch21_videos = sorted_videos(PASSING_DIR) if PASSING_DIR.exists() else []
    ch27_videos = sorted_videos(TAPING_DIR, recursive=True) if TAPING_DIR.exists() else []

    print(f"CH19: {len(ch19_videos)} video files")
    for v in ch19_videos:
        print(f"  {os.path.basename(v)}")
    print(f"\nCH21: {len(ch21_videos)} video files")
    for v in ch21_videos:
        print(f"  {os.path.basename(v)}")
    print(f"\nCH27: {len(ch27_videos)} video files")
    for v in ch27_videos:
        print(f"  {os.path.basename(v)}")
    print()

    ch19_result = None
    ch21_result = None
    ch27_result = None

    if "--ch19-only" in sys.argv:
        ch19_result = run_ch19(ch19_videos, version=version)
    elif "--ch21-only" in sys.argv:
        ch21_result = run_ch21(ch21_videos)
    elif "--ch27-only" in sys.argv:
        # CH27 supports its own version flag (v1=combined activity, v2=air motion).
        # Default to v2 (precision-tuned via 5-min GT clip).
        ch27_version = "v2"
        if "--ch27-version" in sys.argv:
            idx = sys.argv.index("--ch27-version")
            if idx + 1 < len(sys.argv):
                ch27_version = sys.argv[idx + 1]
        ch27_result = run_ch27(ch27_videos, version=ch27_version)
    else:
        # Sequential all (run in separate terminals for parallel)
        if ch19_videos:
            ch19_result = run_ch19(ch19_videos, version=version)
        if ch21_videos:
            ch21_result = run_ch21(ch21_videos)
        if ch27_videos:
            ch27_result = run_ch27(ch27_videos, version="v2")

    # Save results
    if ch19_result:
        ch19_output.write_text(json.dumps(ch19_result, indent=2))
        print(f"\nCH19 saved to: {ch19_output}")
        print(f"  {ch19_result['summary']['total_cuts']} cuts over "
              f"{ch19_result['metadata']['duration_sec']/3600:.1f} hrs "
              f"(variant: {ch19_result['metadata']['version']})")

    if ch21_result:
        CH21_OUTPUT.write_text(json.dumps(ch21_result, indent=2))
        print(f"\nCH21 saved to: {CH21_OUTPUT}")
        print(f"  {ch21_result['total_accepted']} accepted, "
              f"{ch21_result['total_rejected']} rejected over "
              f"{ch21_result['metadata']['duration_sec']/3600:.1f} hrs")

    if ch27_result:
        # Route v1 to its own file so v2 (default) doesn't clobber it
        ver = ch27_result.get("metadata", {}).get("version", "v2")
        out_path = CH27_OUTPUT_V1 if ver == "v1" else CH27_OUTPUT
        out_path.write_text(json.dumps(ch27_result, indent=2))
        print(f"\nCH27 saved to: {out_path}  (variant: {ver})")
        print(f"  {ch27_result['summary']['total_cycles']} cycles "
              f"(L={ch27_result['summary']['left_cycles']}, "
              f"R={ch27_result['summary']['right_cycles']}) over "
              f"{ch27_result['metadata']['duration_sec']/3600:.1f} hrs")


if __name__ == "__main__":
    main()
