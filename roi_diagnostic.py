"""
ROI Diagnostic — Find the best acceptance landing zone and reject motion zone.

For each known accepted/rejected timestamp, compute frame differences in the
area around the scale to see WHERE blankets go after being processed.
Outputs heatmaps and candidate ROI signal profiles.

Usage: python3 roi_diagnostic.py /path/to/1hr_video.mp4
"""

import cv2
import numpy as np
import json
import sys
from pathlib import Path

# Known ground truth timestamps (seconds)
ACCEPTED_TIMES = [
    5, 12, 20, 27, 33, 41, 49, 55, 61, 68, 75, 83, 89, 98,
    115, 122, 130, 137, 145, 169, 179, 186, 194, 201, 231, 269,
    290, 321, 441, 484, 506, 515, 523, 529, 536, 541, 562, 707, 718,
    3167, 3174, 3202, 3212, 3224, 3234, 3243, 3251, 3258, 3269, 3280, 3290,
    3304, 3333, 3343, 3362, 3370, 3377, 3388, 3398, 3408,
    3421, 3429, 3436, 3451, 3460, 3468, 3476, 3483, 3509, 3530, 3540, 3548,
]

REJECTED_TIMES = [
    107, 153, 281, 299, 312, 356, 376, 392, 416, 431, 467, 491,
    623, 633, 647, 662, 671, 699,
    3193, 3355, 3494, 3522,
]

# Candidate ROIs to test (x1, y1, x2, y2)
CANDIDATE_ROIS = {
    'scale':           (1440, 440, 1520, 500),   # existing scale ROI
    'table':           (980,  340, 1240, 450),   # existing table ROI
    'alz_tight':       (1280, 520, 1440, 660),   # acceptance landing zone (tight)
    'alz_wide':        (1200, 500, 1460, 700),   # acceptance landing zone (wide)
    'below_scale':     (1380, 510, 1540, 660),   # directly below scale
    'left_of_scale':   (1250, 440, 1430, 560),   # left of scale, same height
    'far_right':       (1550, 380, 1750, 560),   # far right (reject direction?)
    'table_right':     (1240, 340, 1440, 480),   # between table and scale
    'floor_center':    (1100, 550, 1350, 720),   # floor center-right
}

FPS = 25.0


def get_frame_at(cap, time_sec, fps):
    """Seek to a specific time and return grayscale frame."""
    frame_num = int(time_sec * fps)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
    ret, frame = cap.read()
    if not ret:
        return None
    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


def measure_roi_change(cap, event_time, fps, roi, before_offset=-1, after_offset=4):
    """Measure change in a ROI between before and after an event.

    Returns the mean absolute diff between frame at (event_time + before_offset)
    and frame at (event_time + after_offset).
    """
    x1, y1, x2, y2 = roi

    frame_before = get_frame_at(cap, event_time + before_offset, fps)
    frame_after = get_frame_at(cap, event_time + after_offset, fps)

    if frame_before is None or frame_after is None:
        return None

    patch_before = frame_before[y1:y2, x1:x2].astype(float)
    patch_after = frame_after[y1:y2, x1:x2].astype(float)

    return float(np.mean(np.abs(patch_after - patch_before)))


def measure_motion_window(cap, event_time, fps, roi, window_start=0, window_end=5, step=0.5):
    """Measure motion (frame-to-frame diff) in a ROI over a time window after event.

    Returns list of (time_offset, motion_value) pairs.
    """
    x1, y1, x2, y2 = roi
    results = []
    prev_patch = None

    t = window_start
    while t <= window_end:
        frame = get_frame_at(cap, event_time + t, fps)
        if frame is None:
            t += step
            continue

        patch = frame[y1:y2, x1:x2].astype(float)
        if prev_patch is not None:
            motion = float(np.mean(np.abs(patch - prev_patch)))
            results.append((t, motion))

        prev_patch = patch
        t += step

    return results


def compute_motion_heatmap(cap, event_times, fps, before_offset=-1, after_offset=4):
    """Compute average motion heatmap for a set of events.

    For each event, computes |frame_after - frame_before| and accumulates.
    Returns the average heatmap.
    """
    heatmap = None
    count = 0

    for t in event_times:
        frame_before = get_frame_at(cap, t + before_offset, fps)
        frame_after = get_frame_at(cap, t + after_offset, fps)

        if frame_before is None or frame_after is None:
            continue

        diff = np.abs(frame_after.astype(float) - frame_before.astype(float))

        if heatmap is None:
            heatmap = diff
        else:
            heatmap += diff
        count += 1

    if count > 0:
        heatmap /= count

    return heatmap, count


def main():
    if len(sys.argv) < 2:
        print("Usage: python3 roi_diagnostic.py /path/to/video.mp4")
        sys.exit(1)

    video_path = sys.argv[1]
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"ERROR: Cannot open {video_path}")
        sys.exit(1)

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps
    print(f"Video: {video_path}")
    print(f"FPS: {fps:.1f}, Duration: {duration:.1f}s, Frames: {total_frames}")

    outdir = Path(video_path).parent / "frames" / "diagnostic"
    outdir.mkdir(parents=True, exist_ok=True)

    # ═══════════════════════════════════════════════════════════════
    # 1. MOTION HEATMAPS: Where do accepted vs rejected blankets go?
    # ═══════════════════════════════════════════════════════════════
    print("\n" + "="*60)
    print("  MOTION HEATMAPS")
    print("="*60)

    # Use first 12 minutes only (most GT data there)
    acc_early = [t for t in ACCEPTED_TIMES if t < 720]
    rej_early = [t for t in REJECTED_TIMES if t < 720]

    print(f"\n  Computing accepted heatmap ({len(acc_early)} events)...")
    heatmap_acc, n_acc = compute_motion_heatmap(cap, acc_early, fps, before_offset=-1, after_offset=4)
    print(f"  Used {n_acc}/{len(acc_early)} events")

    print(f"  Computing rejected heatmap ({len(rej_early)} events)...")
    heatmap_rej, n_rej = compute_motion_heatmap(cap, rej_early, fps, before_offset=-1, after_offset=4)
    print(f"  Used {n_rej}/{len(rej_early)} events")

    # Save heatmaps as images
    if heatmap_acc is not None:
        # Normalize to 0-255 for visualization
        h_norm = (heatmap_acc / max(heatmap_acc.max(), 1) * 255).astype(np.uint8)
        h_color = cv2.applyColorMap(h_norm, cv2.COLORMAP_JET)
        cv2.imwrite(str(outdir / "heatmap_accepted.jpg"), h_color)
        print(f"  Saved: {outdir / 'heatmap_accepted.jpg'}")

        # Also save with ROI overlays
        for name, (x1, y1, x2, y2) in CANDIDATE_ROIS.items():
            cv2.rectangle(h_color, (x1, y1), (x2, y2), (255, 255, 255), 2)
            cv2.putText(h_color, name, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1)
        cv2.imwrite(str(outdir / "heatmap_accepted_rois.jpg"), h_color)

    if heatmap_rej is not None:
        h_norm = (heatmap_rej / max(heatmap_rej.max(), 1) * 255).astype(np.uint8)
        h_color = cv2.applyColorMap(h_norm, cv2.COLORMAP_JET)
        cv2.imwrite(str(outdir / "heatmap_rejected.jpg"), h_color)
        print(f"  Saved: {outdir / 'heatmap_rejected.jpg'}")

        for name, (x1, y1, x2, y2) in CANDIDATE_ROIS.items():
            cv2.rectangle(h_color, (x1, y1), (x2, y2), (255, 255, 255), 2)
            cv2.putText(h_color, name, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1)
        cv2.imwrite(str(outdir / "heatmap_rejected_rois.jpg"), h_color)

    # Difference heatmap: where accepted motion is HIGH and rejected is LOW
    if heatmap_acc is not None and heatmap_rej is not None:
        diff_map = heatmap_acc - heatmap_rej
        # Positive = more motion during accepts (acceptance landing zone)
        # Negative = more motion during rejects (reject zone)
        pos = np.clip(diff_map, 0, None)
        neg = np.clip(-diff_map, 0, None)

        # Green = acceptance zone, Red = reject zone
        diff_color = np.zeros((*diff_map.shape, 3), dtype=np.uint8)
        diff_color[:,:,1] = (pos / max(pos.max(), 1) * 255).astype(np.uint8)  # green
        diff_color[:,:,2] = (neg / max(neg.max(), 1) * 255).astype(np.uint8)  # red

        for name, (x1, y1, x2, y2) in CANDIDATE_ROIS.items():
            cv2.rectangle(diff_color, (x1, y1), (x2, y2), (255, 255, 255), 2)
            cv2.putText(diff_color, name, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1)
        cv2.imwrite(str(outdir / "heatmap_diff_green_acc_red_rej.jpg"), diff_color)
        print(f"  Saved: {outdir / 'heatmap_diff_green_acc_red_rej.jpg'}")

    # ═══════════════════════════════════════════════════════════════
    # 2. ROI SIGNAL ANALYSIS: Which ROI best discriminates?
    # ═══════════════════════════════════════════════════════════════
    print("\n" + "="*60)
    print("  ROI SIGNAL ANALYSIS")
    print("="*60)

    # Test each candidate ROI: measure change for every accepted and rejected event
    for roi_name, roi in CANDIDATE_ROIS.items():
        acc_signals = []
        rej_signals = []

        # Test with multiple time offsets
        for after_t in [3, 5, 7]:
            for t in acc_early[:20]:  # first 20 accepted events
                val = measure_roi_change(cap, t, fps, roi, before_offset=-1, after_offset=after_t)
                if val is not None:
                    acc_signals.append(val)

            for t in rej_early[:15]:  # first 15 rejected events
                val = measure_roi_change(cap, t, fps, roi, before_offset=-1, after_offset=after_t)
                if val is not None:
                    rej_signals.append(val)

        if acc_signals and rej_signals:
            acc_mean = np.mean(acc_signals)
            acc_std = np.std(acc_signals)
            rej_mean = np.mean(rej_signals)
            rej_std = np.std(rej_signals)
            # Discrimination ratio: how separated are the distributions?
            separation = abs(acc_mean - rej_mean) / max((acc_std + rej_std) / 2, 0.1)

            print(f"\n  {roi_name:20s} | Accept: {acc_mean:5.1f} +/- {acc_std:4.1f} "
                  f"| Reject: {rej_mean:5.1f} +/- {rej_std:4.1f} "
                  f"| Separation: {separation:.2f}")
        else:
            print(f"\n  {roi_name:20s} | No data")

    # ═══════════════════════════════════════════════════════════════
    # 3. TABLE TEXTURE PROFILE: Final fold signature
    # ═══════════════════════════════════════════════════════════════
    print("\n" + "="*60)
    print("  TABLE TEXTURE PROFILES (Final Fold Detection)")
    print("="*60)

    tx1, ty1, tx2, ty2 = CANDIDATE_ROIS['table']

    # For accepted events: measure texture in the last 3 seconds before the event
    print("\n  Accepted events - texture in final 3 seconds before scale event:")
    for t in acc_early[:15]:
        profile = []
        for offset in [-3.0, -2.5, -2.0, -1.5, -1.0, -0.5, 0.0]:
            frame = get_frame_at(cap, t + offset, fps)
            if frame is not None:
                texture = float(np.std(frame[ty1:ty2, tx1:tx2]))
                profile.append(texture)
        if profile:
            trend = profile[-1] - profile[0] if len(profile) > 1 else 0
            print(f"    t={t:4d}s | texture: {' → '.join(f'{v:.0f}' for v in profile)}"
                  f" | trend: {trend:+.0f}")

    print("\n  Rejected events - texture in final 3 seconds before reject:")
    for t in rej_early[:15]:
        profile = []
        for offset in [-3.0, -2.5, -2.0, -1.5, -1.0, -0.5, 0.0]:
            frame = get_frame_at(cap, t + offset, fps)
            if frame is not None:
                texture = float(np.std(frame[ty1:ty2, tx1:tx2]))
                profile.append(texture)
        if profile:
            trend = profile[-1] - profile[0] if len(profile) > 1 else 0
            print(f"    t={t:4d}s | texture: {' → '.join(f'{v:.0f}' for v in profile)}"
                  f" | trend: {trend:+.0f}")

    # ═══════════════════════════════════════════════════════════════
    # 4. SAVE ANNOTATED FRAMES at key moments
    # ═══════════════════════════════════════════════════════════════
    print("\n" + "="*60)
    print("  SAVING ANNOTATED FRAMES")
    print("="*60)

    # Save frames at a few accepted and rejected moments with all ROIs drawn
    sample_times = [
        (5, 'accept_t5'),
        (12, 'accept_t12'),
        (107, 'reject_t107'),
        (153, 'reject_t153'),
        (281, 'reject_t281'),
    ]

    for t, label in sample_times:
        # Frame at event time
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(t * fps))
        ret, frame = cap.read()
        if not ret:
            continue

        # Draw all candidate ROIs
        colors = {
            'scale': (0, 0, 255),       # red
            'table': (0, 255, 0),        # green
            'alz_tight': (255, 255, 0),  # cyan
            'alz_wide': (255, 200, 0),   # light cyan
            'below_scale': (0, 165, 255),# orange
            'left_of_scale': (0, 255, 255),# yellow
            'far_right': (128, 0, 255),  # purple
            'table_right': (255, 0, 255),# magenta
            'floor_center': (255, 128, 0),# blue-ish
        }

        for name, (x1, y1, x2, y2) in CANDIDATE_ROIS.items():
            color = colors.get(name, (255, 255, 255))
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, name, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        # Also draw frames at t+3s (after blanket has been tossed)
        cap.set(cv2.CAP_PROP_POS_FRAMES, int((t + 3) * fps))
        ret2, frame_after = cap.read()

        if ret2:
            for name, (x1, y1, x2, y2) in CANDIDATE_ROIS.items():
                color = colors.get(name, (255, 255, 255))
                cv2.rectangle(frame_after, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame_after, name, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

            # Stack side by side: before | after
            combined = np.hstack([frame, frame_after])
            cv2.putText(combined, f"t={t}s (event)", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
            cv2.putText(combined, f"t={t+3}s (after)", (frame.shape[1]+20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
        else:
            combined = frame

        out_path = outdir / f"annotated_{label}.jpg"
        cv2.imwrite(str(out_path), combined)
        print(f"  Saved: {out_path}")

    cap.release()

    print("\n" + "="*60)
    print("  DIAGNOSTIC COMPLETE")
    print("="*60)
    print(f"\n  All outputs in: {outdir}")
    print("  Key files:")
    print("    heatmap_accepted.jpg — average motion after accepted events")
    print("    heatmap_rejected.jpg — average motion after rejected events")
    print("    heatmap_diff_green_acc_red_rej.jpg — green=accept zone, red=reject zone")
    print("    annotated_*.jpg — side-by-side frames at event time vs 3s later")


if __name__ == "__main__":
    main()
