"""
Focused diagnostic on the undetected and misclassified rejected blankets.

Analyzes:
1. The 7 completely undetected rejects (no table or scale signal at all)
2. The 4 weight-rejected blankets (went on scale, then rejected)
3. All 22 rejected blankets — table texture profile and duration analysis

Usage: python3 reject_diagnostic.py /path/to/1hr_video.mp4
"""

import cv2
import numpy as np
import sys

TABLE_ROI = (980, 340, 1240, 450)
SCALE_ROI = (1440, 440, 1520, 500)

# All 22 rejected blankets with categories
REJECTED = {
    # Correctly detected as rejected (table cycle, no scale) — 11 events
    'detected': [107, 153, 299, 356, 376, 392, 416, 431, 491, 671, 3355],

    # Detected as ACCEPTED (went on scale, weight-rejected) — 4 events
    'weight_rejected': [153, 281, 416, 3494],  # Wait, let me use the correct ones
    # Actually from PROJECT_NOTES: 2:33, 6:56, 10:33, 58:14
    'weight_rejected_corrected': [153, 416, 633, 3494],
}

# Correct categorization from PROJECT_NOTES.md:
# Correctly detected as rejected: 11/22
# Falsely counted as accepted (weight-rejected): 4/22 → times 2:33=153, 6:56=416, 10:33=633, 58:14=3494
# Not detected at all: 7/22

ALL_REJECTED = [
    107, 153, 281, 299, 312, 356, 376, 392, 416, 431, 467, 491,
    623, 633, 647, 662, 671, 699,
    3193, 3355, 3494, 3522,
]

# Weight-rejected (from PROJECT_NOTES: 2:33, 6:56, 10:33, 58:14)
WEIGHT_REJECTED = [153, 416, 633, 3494]

ALL_ACCEPTED = [
    5, 12, 20, 27, 33, 41, 49, 55, 61, 68, 75, 83, 89, 98,
    115, 122, 130, 137, 145, 169, 179, 186, 194, 201, 231, 269,
    290, 321, 441, 484, 506, 515, 523, 529, 536, 541, 562, 707, 718,
    3167, 3174, 3202, 3212, 3224, 3234, 3243, 3251, 3258, 3269, 3280, 3290,
    3304, 3333, 3343, 3362, 3370, 3377, 3388, 3398, 3408,
    3421, 3429, 3436, 3451, 3460, 3468, 3476, 3483, 3509, 3530, 3540, 3548,
]


def fmt(secs):
    return f"{int(secs)//60}:{int(secs)%60:02d}"


def get_frame_at(cap, time_sec, fps):
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(time_sec * fps))
    ret, frame = cap.read()
    if not ret:
        return None
    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


def analyze_table_profile(cap, event_time, fps, window_before=6, window_after=3, step=0.2):
    """Detailed texture profile around an event.

    Returns list of (time_offset, texture_std, scale_diff) tuples.
    """
    tx1, ty1, tx2, ty2 = TABLE_ROI
    sx1, sy1, sx2, sy2 = SCALE_ROI

    # Get scale reference from 10s before event
    ref_frame = get_frame_at(cap, max(0, event_time - 10), fps)
    scale_ref = None
    if ref_frame is not None:
        scale_ref = ref_frame[sy1:sy2, sx1:sx2].astype(float)

    profile = []
    t = -window_before
    while t <= window_after:
        frame = get_frame_at(cap, event_time + t, fps)
        if frame is not None:
            texture = float(np.std(frame[ty1:ty2, tx1:tx2]))
            if scale_ref is not None:
                scale_diff = float(np.mean(np.abs(frame[sy1:sy2, sx1:sx2].astype(float) - scale_ref)))
            else:
                scale_diff = 0
            profile.append((t, texture, scale_diff))
        t += step

    return profile


def compute_table_cycle_features(profile, texture_threshold=75):
    """Extract features from a texture profile.

    Returns dict with:
    - above_duration: total time texture > threshold (seconds)
    - peak_texture: max texture value
    - final_texture: texture at t=0 (event time)
    - texture_at_minus1: texture at t=-1s
    - texture_slope: slope of texture in final 2s
    - above_segments: number of continuous above-threshold segments
    """
    if not profile:
        return None

    above_duration = 0
    peak_texture = 0
    final_texture = 0
    tex_at_minus1 = 0
    tex_at_minus2 = 0

    for t, tex, _ in profile:
        if tex > texture_threshold:
            above_duration += 0.2  # step size
        peak_texture = max(peak_texture, tex)
        if abs(t) < 0.15:
            final_texture = tex
        if abs(t + 1.0) < 0.15:
            tex_at_minus1 = tex
        if abs(t + 2.0) < 0.15:
            tex_at_minus2 = tex

    # Texture slope in final 2 seconds
    slope = final_texture - tex_at_minus2 if tex_at_minus2 > 0 else 0

    # Count above-threshold segments
    above = False
    segments = 0
    for _, tex, _ in profile:
        if tex > texture_threshold and not above:
            segments += 1
            above = True
        elif tex <= texture_threshold:
            above = False

    return {
        'above_duration': round(above_duration, 1),
        'peak_texture': round(peak_texture, 1),
        'final_texture': round(final_texture, 1),
        'texture_slope_2s': round(slope, 1),
        'above_segments': segments,
    }


def compute_scale_features(profile):
    """Extract scale-related features from the profile."""
    peak_scale = 0
    scale_above_25_duration = 0

    for t, _, scale_diff in profile:
        peak_scale = max(peak_scale, scale_diff)
        if scale_diff > 25:
            scale_above_25_duration += 0.2

    return {
        'peak_scale_diff': round(peak_scale, 1),
        'scale_above25_duration': round(scale_above_25_duration, 1),
    }


def main():
    if len(sys.argv) < 2:
        print("Usage: python3 reject_diagnostic.py /path/to/video.mp4")
        sys.exit(1)

    video_path = sys.argv[1]
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"ERROR: Cannot open {video_path}")
        sys.exit(1)

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0

    # ═══════════════════════════════════════════════════════════════
    # 1. ALL REJECTED BLANKETS — detailed profile
    # ═══════════════════════════════════════════════════════════════
    print("="*80)
    print("  ALL REJECTED BLANKETS — Table & Scale Profile")
    print("="*80)

    rej_features = []
    for t in ALL_REJECTED:
        profile = analyze_table_profile(cap, t, fps, window_before=6, window_after=3)
        table_feat = compute_table_cycle_features(profile)
        scale_feat = compute_scale_features(profile)

        is_weight_rej = t in WEIGHT_REJECTED
        label = "WEIGHT-REJ" if is_weight_rej else "REJECTED  "

        if table_feat and scale_feat:
            feat = {**table_feat, **scale_feat, 'time': t, 'is_weight_rej': is_weight_rej}
            rej_features.append(feat)
            print(f"  {fmt(t):>6s} [{label}] | "
                  f"table: above={table_feat['above_duration']:3.1f}s "
                  f"peak={table_feat['peak_texture']:5.1f} "
                  f"final={table_feat['final_texture']:5.1f} "
                  f"slope={table_feat['texture_slope_2s']:+5.1f} "
                  f"segs={table_feat['above_segments']} | "
                  f"scale: peak={scale_feat['peak_scale_diff']:5.1f} "
                  f"dur25={scale_feat['scale_above25_duration']:3.1f}s")

    # ═══════════════════════════════════════════════════════════════
    # 2. SAMPLE ACCEPTED BLANKETS — for comparison
    # ═══════════════════════════════════════════════════════════════
    print("\n" + "="*80)
    print("  SAMPLE ACCEPTED BLANKETS — Table & Scale Profile")
    print("="*80)

    acc_features = []
    for t in ALL_ACCEPTED[:25]:  # first 25 for comparison
        profile = analyze_table_profile(cap, t, fps, window_before=6, window_after=3)
        table_feat = compute_table_cycle_features(profile)
        scale_feat = compute_scale_features(profile)

        if table_feat and scale_feat:
            feat = {**table_feat, **scale_feat, 'time': t}
            acc_features.append(feat)
            print(f"  {fmt(t):>6s} [ACCEPTED  ] | "
                  f"table: above={table_feat['above_duration']:3.1f}s "
                  f"peak={table_feat['peak_texture']:5.1f} "
                  f"final={table_feat['final_texture']:5.1f} "
                  f"slope={table_feat['texture_slope_2s']:+5.1f} "
                  f"segs={table_feat['above_segments']} | "
                  f"scale: peak={scale_feat['peak_scale_diff']:5.1f} "
                  f"dur25={scale_feat['scale_above25_duration']:3.1f}s")

    # ═══════════════════════════════════════════════════════════════
    # 3. STATISTICAL COMPARISON
    # ═══════════════════════════════════════════════════════════════
    print("\n" + "="*80)
    print("  STATISTICAL COMPARISON")
    print("="*80)

    def stats(vals, label):
        if not vals:
            return f"{label}: no data"
        return (f"{label}: mean={np.mean(vals):5.1f} "
                f"median={np.median(vals):5.1f} "
                f"std={np.std(vals):4.1f} "
                f"range=[{min(vals):.1f}, {max(vals):.1f}]")

    features_to_compare = ['above_duration', 'peak_texture', 'final_texture',
                           'texture_slope_2s', 'above_segments', 'peak_scale_diff']

    for feat_name in features_to_compare:
        acc_vals = [f[feat_name] for f in acc_features]
        rej_vals = [f[feat_name] for f in rej_features if not f.get('is_weight_rej')]
        wr_vals = [f[feat_name] for f in rej_features if f.get('is_weight_rej')]

        print(f"\n  {feat_name}:")
        print(f"    {stats(acc_vals, 'Accepted     ')}")
        print(f"    {stats(rej_vals, 'Rejected     ')}")
        print(f"    {stats(wr_vals, 'Weight-rej   ')}")

        # Separation score
        if acc_vals and rej_vals:
            acc_m, rej_m = np.mean(acc_vals), np.mean(rej_vals)
            acc_s, rej_s = np.std(acc_vals), np.std(rej_vals)
            sep = abs(acc_m - rej_m) / max((acc_s + rej_s) / 2, 0.1)
            print(f"    Separation (acc vs rej): {sep:.2f}")

    # ═══════════════════════════════════════════════════════════════
    # 4. THRESHOLDING ANALYSIS
    # ═══════════════════════════════════════════════════════════════
    print("\n" + "="*80)
    print("  THRESHOLDING ANALYSIS")
    print("="*80)

    # For each potential feature, test various thresholds
    print("\n  Can we classify based on above_duration?")
    for thresh in [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]:
        acc_correct = sum(1 for f in acc_features if f['above_duration'] >= thresh)
        rej_correct = sum(1 for f in rej_features if f['above_duration'] < thresh)
        print(f"    thresh={thresh:.1f}s: "
              f"acc_correct={acc_correct}/{len(acc_features)} "
              f"rej_correct={rej_correct}/{len(rej_features)} "
              f"total={acc_correct+rej_correct}/{len(acc_features)+len(rej_features)}")

    print("\n  Can we classify based on peak_scale_diff?")
    for thresh in [10, 15, 20, 25, 30]:
        acc_correct = sum(1 for f in acc_features if f['peak_scale_diff'] >= thresh)
        rej_correct = sum(1 for f in rej_features if not f.get('is_weight_rej')
                          and f['peak_scale_diff'] < thresh)
        wr_caught = sum(1 for f in rej_features if f.get('is_weight_rej')
                        and f['peak_scale_diff'] >= thresh)
        print(f"    thresh={thresh}: "
              f"acc_correct={acc_correct}/{len(acc_features)} "
              f"rej_no_scale={rej_correct}/{sum(1 for f in rej_features if not f.get('is_weight_rej'))} "
              f"wr_trigger={wr_caught}/{sum(1 for f in rej_features if f.get('is_weight_rej'))}")

    print("\n  Can we classify based on peak_texture?")
    for thresh in [75, 80, 85, 90, 95, 100]:
        acc_above = sum(1 for f in acc_features if f['peak_texture'] >= thresh)
        rej_above = sum(1 for f in rej_features if f['peak_texture'] >= thresh)
        print(f"    thresh={thresh}: "
              f"acc_above={acc_above}/{len(acc_features)} ({acc_above/max(len(acc_features),1)*100:.0f}%) "
              f"rej_above={rej_above}/{len(rej_features)} ({rej_above/max(len(rej_features),1)*100:.0f}%)")

    cap.release()
    print("\n" + "="*80)
    print("  DIAGNOSTIC COMPLETE")
    print("="*80)


if __name__ == "__main__":
    main()
