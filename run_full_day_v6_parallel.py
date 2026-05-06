#!/usr/bin/env python3
"""Parallel CH27 v6 run — processes all NVR segments in parallel, merges with offsets."""
import json, os, sys, time
from pathlib import Path
from multiprocessing import Pool, cpu_count

BASE = Path(__file__).parent

VIDEOS = [
    ("Tape/NVR_ch27_main_20260428090012_20260428100012.mp4", "09:00-10:00"),
    ("Tape 4/NVR_ch27_main_20260428100012_20260428110012.mp4", "10:00-11:00"),
    ("Tape 4/NVR_ch27_main_20260428110012_20260428120012.mp4", "11:00-12:00"),
    ("Tape 2/NVR_ch27_main_20260428120012_20260428125317.mp4", "12:00-12:53"),
    ("Tape 5/NVR_ch27_main_20260428125317_20260428140012.mp4", "12:53-14:00"),
    ("Tape 3/NVR_ch27_main_20260428140012_20260428150012.mp4", "14:00-15:00"),
    ("Tape/NVR_ch27_main_20260428150012_20260428160012.mp4", "15:00-16:00"),
    ("Tape 3/NVR_ch27_main_20260428170012_20260428180012.mp4", "17:00-18:00"),
    ("Tape 3/NVR_ch27_main_20260428180012_20260428190012.mp4", "18:00-19:00"),
]

def process_segment(args):
    i, rel_path, label = args
    video_path = BASE / "Taping Cam27" / rel_path
    if not video_path.exists():
        print(f"[{label}] SKIP — file not found: {video_path}")
        return None

    sys.path.insert(0, str(BASE))
    from taping_counter import TapingCounter

    t0 = time.time()
    c = TapingCounter(str(video_path), version="v4", fast_mode=True)
    result = c.run()
    elapsed = time.time() - t0

    events = result["events"]
    suppressed = result.get("suppressed_candidates", [])
    breaks = result.get("breaks", [])
    dur = result["metadata"]["duration_sec"]
    fps = result["metadata"]["total_frames"] / elapsed if elapsed > 0 else 0

    print(f"[{label}] {len(events)} events "
          f"(L={sum(1 for e in events if e.get('table')=='left')}, "
          f"R={sum(1 for e in events if e.get('table')=='right')}) "
          f"{elapsed:.0f}s ({fps:.0f}fps)")

    return {
        "segment": i,
        "label": label,
        "file": video_path.name,
        "duration_sec": dur,
        "total_frames": result["metadata"]["total_frames"],
        "events": events,
        "suppressed": suppressed,
        "breaks": breaks,
    }


def main():
    existing = {}
    for i, (rel_path, label) in enumerate(VIDEOS):
        video_path = BASE / "Taping Cam27" / rel_path
        if video_path.exists():
            existing[i] = (i, rel_path, label)
            print(f"[{label}] queued")
        else:
            print(f"[{label}] MISSING")

    if not existing:
        print("No videos found!")
        return

    workers = min(cpu_count(), len(existing))
    print(f"\nRunning {len(existing)} segments with {workers} parallel workers...\n")

    with Pool(workers) as pool:
        results = pool.map(process_segment, existing.values())

    results = [r for r in results if r is not None]
    results.sort(key=lambda r: r["segment"])

    # Merge with time offsets
    all_events = []
    all_suppressed = []
    total_dur = 0.0
    total_frames = 0
    total_proc = 0.0
    segments = []

    for r in results:
        offset = total_dur
        for e in r["events"]:
            e["time_sec"] = round(e["time_sec"] + offset, 2)
            e["segment"] = r["segment"]
        for s in r["suppressed"]:
            s["time_sec"] = round(s.get("time_sec", 0) + offset, 2)
            s["segment"] = r["segment"]

        all_events.extend(r["events"])
        all_suppressed.extend(r["suppressed"])
        total_dur += r["duration_sec"]
        total_frames += r["total_frames"]

        L = sum(1 for e in r["events"] if e.get("table") == "left")
        R = sum(1 for e in r["events"] if e.get("table") == "right")

        segments.append({
            "label": r["label"],
            "segment_index": r["segment"],
            "offset_sec": round(offset, 2),
            "duration_sec": round(r["duration_sec"], 2),
            "frames": r["total_frames"],
            "left_cycles": L,
            "right_cycles": R,
        })

    # Summary
    import numpy as np

    # Post-merge temporal NMS — catches double-fires across segment boundaries
    nms_suppressed = 0
    if all_events:
        per_tbl = {"left": [], "right": []}
        for e in all_events:
            tbl = e.get("table", "")
            if tbl in per_tbl:
                per_tbl[tbl].append(e)
        kept_all = []
        for tbl in ["left", "right"]:
            evts = sorted(per_tbl[tbl],
                          key=lambda e: (e["time_sec"], -e.get("v4_prob", 0)))
            kept = []
            for e in evts:
                suppressed = False
                for k in kept:
                    if abs(e["time_sec"] - k["time_sec"]) <= 5.0:
                        if e.get("v4_prob", 0) <= k.get("v4_prob", 0):
                            suppressed = True
                            break
                if not suppressed:
                    kept.append(e)
                else:
                    all_suppressed.append({**e, "reason": "temporal_nms"})
                    nms_suppressed += 1
            kept_all.extend(kept)
        all_events = kept_all
        print(f"\n  [post-merge NMS] suppressed {nms_suppressed} boundary-crossing events")

    L_all = [e for e in all_events if e.get("table") == "left"]
    R_all = [e for e in all_events if e.get("table") == "right"]

    merged = {
        "metadata": {
            "type": "full_day_v6",
            "camera": "CH27",
            "total_videos": len(results),
            "duration_sec": round(total_dur, 2),
            "total_frames": total_frames,
            "version": "v8 (30 features, table_motion + heap_std + cooldown 5s + post-merge NMS)",
            "generated_at": __import__("datetime").datetime.now().isoformat(),
        },
        "segments": segments,
        "summary": {
            "total_cycles": len(all_events),
            "left_cycles": len(L_all),
            "right_cycles": len(R_all),
            "suppressed_count": len(all_suppressed),
        },
        "events": all_events[:10000],
        "suppressed_candidates": all_suppressed[:5000],
    }

    out_path = BASE / "taping_fullday_v8.json"
    out_path.write_text(json.dumps(merged, indent=2))
    print(f"\nSaved to {out_path}")
    print(f"Total: {len(all_events)} cycles (L={len(L_all)}, R={len(R_all)}) over {total_dur/3600:.1f} hrs")


if __name__ == "__main__":
    main()
