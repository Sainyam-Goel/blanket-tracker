#!/usr/bin/env python3
"""Phase 1: Data-driven ROI discovery via motion heatmaps.

Computes per-pixel motion from labeled toss frames vs idle frames
to discover which regions of the frame uniquely signal a toss event.

Output:
  gt_clips/heatmap_toss.jpg     — average motion during toss events
  gt_clips/heatmap_idle.jpg     — average motion during idle periods
  gt_clips/heatmap_contrast.jpg — toss minus idle (bright = toss-unique)
  gt_clips/heatmap_regions.json — extracted ROI candidates
"""

import json
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))

from train_taping_classifier import cluster_labels, CLIPS_FOR_TRAINING

CLIPS_DIR = HERE / "gt_clips"
FRAMES_AROUND = 5   # ±5 frames around each labeled toss frame
IDLE_CLIPS = ["gt_clip5_endofday", "gt_clip7_postlunch_return"]


def load_frame_patches(clip_name, labels, clip_type="toss", max_frames=200):
    """For each labeled toss frame, grab the frame and compute motion.

    Returns accumulated motion, brightness change, and pixel count heatmaps
    at 1920×1080 resolution.
    """
    video_path = CLIPS_DIR / f"{clip_name}.mp4"
    if not video_path.exists():
        print(f"  [skip] {video_path} not found")
        return None

    cap = cv2.VideoCapture(str(video_path))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Separate labels: toss vs idle
    toss_labels = [l for l in labels if l["type"] == "toss"]
    load_labels = [l for l in labels if l["type"] == "load"]

    # 1) Toss clusters — frame-diff motion at the toss peak
    clusters = cluster_labels(labels)
    toss_clusters = [c for c in clusters if c["type"] == "toss"]

    motion_acc = np.zeros((1080, 1920), dtype=np.float64)
    color_acc = {"B": np.zeros((1080, 1920), dtype=np.float64),
                 "G": np.zeros((1080, 1920), dtype=np.float64),
                 "R": np.zeros((1080, 1920), dtype=np.float64)}
    count = 0

    for cl in toss_clusters:
        # Use the FIRST labeled frame of each cluster (best signal)
        peak_f = cl["peak_frame"]
        start_f = max(0, peak_f - 3)
        end_f = min(total_frames - 1, peak_f + 2)

        cap.set(cv2.CAP_PROP_POS_FRAMES, start_f)
        prev_gray = None
        for f in range(start_f, end_f + 1):
            ret, frame = cap.read()
            if not ret:
                break
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            if prev_gray is not None:
                # Motion = abs frame-to-frame difference
                diff = np.abs(gray.astype(np.float64) - prev_gray.astype(np.float64))
                motion_acc += diff
                # Per-channel motion
                for ch, idx in [("B", 0), ("G", 1), ("R", 2)]:
                    prev_ch = prev_frame[:, :, idx].astype(np.float64)
                    curr_ch = frame[:, :, idx].astype(np.float64)
                    color_acc[ch] += np.abs(curr_ch - prev_ch)

                count += 1

            prev_gray = gray
            prev_frame = frame

        if count >= max_frames:
            break

    cap.release()

    if count == 0:
        return None

    motion_avg = motion_acc / count
    for ch in color_acc:
        color_acc[ch] /= count

    print(f"  {clip_name}: {len(toss_clusters)} toss clusters → {count} motion patches")
    return {"motion": motion_avg, "color": color_acc, "count": count}


def sample_idle_motion(cap, total_frames, max_frames=200):
    """Sample frames evenly across an idle clip to capture background motion."""
    step = max(1, (total_frames - 10) // (max_frames // 5))
    motion_acc = np.zeros((1080, 1920), dtype=np.float64)
    count = 0

    for start_f in range(5, total_frames - 5, step):
        if count >= max_frames:
            break
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_f)
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Get next frame for diff
        ret2, frame2 = cap.read()
        if not ret2:
            break
        gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

        diff = np.abs(gray.astype(np.float64) - gray2.astype(np.float64))
        motion_acc += diff
        count += 1

    if count == 0:
        return None
    return {"motion": motion_acc / count, "count": count}


def main():
    print("=" * 60)
    print("  PHASE 1 — DATA-DRIVEN ROI DISCOVERY")
    print("=" * 60)

    # Accumulate heatmaps across all clips
    toss_motion = np.zeros((1080, 1920), dtype=np.float64)
    toss_count = 0

    idle_motion = np.zeros((1080, 1920), dtype=np.float64)
    idle_count = 0

    # Process training clips (toss + load)
    for clip in CLIPS_FOR_TRAINING:
        if clip in IDLE_CLIPS:
            continue
        try:
            d = json.loads((CLIPS_DIR / f"{clip}.labels.json").read_text())
        except Exception:
            print(f"  [skip] {clip} — no labels")
            continue

        result = load_frame_patches(clip, d["labels"])
        if result:
            toss_motion += result["motion"] * result["count"]
            toss_count += result["count"]

    # Process idle clips separately — sample evenly (no labels to use)
    for clip in IDLE_CLIPS:
        video_path = CLIPS_DIR / f"{clip}.mp4"
        if not video_path.exists():
            continue
        cap = cv2.VideoCapture(str(video_path))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        result = sample_idle_motion(cap, total_frames, max_frames=200)
        cap.release()
        if result:
            idle_motion += result["motion"] * result["count"]
            idle_count += result["count"]
            print(f"  {clip}: {result['count']} idle patches")

    if toss_count == 0:
        print("ERROR: no toss frames found")
        return

    toss_avg = toss_motion / toss_count
    idle_avg = idle_motion / max(1, idle_count)
    contrast = np.clip(toss_avg - idle_avg, 0, None)

    # Normalize to 0-255 for saving
    print(f"\n  Total toss patches: {toss_count}")
    print(f"  Total idle patches: {idle_count}")

    # Save heatmaps
    for name, data, cmap in [
        ("toss", toss_avg, cv2.COLORMAP_HOT),
        ("idle", idle_avg, cv2.COLORMAP_HOT),
        ("contrast", contrast, cv2.COLORMAP_JET),
    ]:
        norm = np.clip(data / max(1.0, data.max()), 0, 1)
        img = (norm * 255).astype(np.uint8)
        colored = cv2.applyColorMap(img, cmap)
        out = str(CLIPS_DIR / f"heatmap_{name}.jpg")
        cv2.imwrite(out, colored)
        print(f"  Saved: {out}")

    # Also save raw numpy for Phase 2
    np.savez_compressed(str(CLIPS_DIR / "heatmaps.npz"),
                        toss=toss_avg, idle=idle_avg, contrast=contrast)
    print(f"  Saved: {CLIPS_DIR}/heatmaps.npz")

    # Find hotspots in contrast map
    print(f"\n=== ROI CANDIDATES (top 5 contrast regions) ===")
    # Downsample to 8×8 grid to find rough regions
    grid_h, grid_w = 12, 20  # 90×96 pixel cells
    cell_h, cell_w = 1080 // grid_h, 1920 // grid_w
    scores = []
    for gy in range(grid_h):
        for gx in range(grid_w):
            y1, y2 = gy * cell_h, min(1080, (gy + 1) * cell_h)
            x1, x2 = gx * cell_w, min(1920, (gx + 1) * cell_w)
            score = float(contrast[y1:y2, x1:x2].mean())
            scores.append((score, gx, gy, x1, y1, x2, y2))

    scores.sort(key=lambda s: -s[0])
    for rank, (score, gx, gy, x1, y1, x2, y2) in enumerate(scores[:10]):
        print(f"  {rank+1}. ({x1},{y1})→({x2},{y2})  score={score:.3f}  "
              f"grid({gx},{gy})")

    # Save as JSON
    regions = {
        "description": "Top 10 high-contrast regions (toss - idle motion)",
        "frame_size": [1920, 1080],
        "toss_patches": toss_count,
        "idle_patches": idle_count,
        "regions": [(int(x1), int(y1), int(x2), int(y2), float(score))
                    for score, gx, gy, x1, y1, x2, y2 in scores[:10]],
    }
    (CLIPS_DIR / "heatmap_regions.json").write_text(json.dumps(regions, indent=2))


if __name__ == "__main__":
    main()
