#!/usr/bin/env python3
"""LOAD DETECTOR v2 — dedicated table-texture candidate generator.

Unlike the toss model (v2 tracker emits air-motion candidates),
the load model uses table texture derivative spikes as triggers.
A load is a state transition on the table, not a ballistic event in the air.

Candidate generation: rolling np.diff(table_std_dev) > threshold.
Features: texture step, color shift, asymmetry, pre-trigger air check.
Trained on LOAD cluster labels from all clips.
"""

import cv2
import json
import sys
import time
from collections import deque
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold, cross_val_score

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from train_taping_classifier import CLIPS_FOR_TRAINING, cluster_labels, load_clip_gt
from taping_counter import LEFT_TABLE_ROI_V2, RIGHT_TABLE_ROI_V2
from taping_counter import LEFT_AIR_ROI_V2, RIGHT_AIR_ROI_V2

# ═══════════════════════════════════════════════════════════════
# Config
# ═══════════════════════════════════════════════════════════════

TABLE_TEXTURE_BASELINE_FRAMES = 75  # 3s rolling baseline for "empty" texture
TABLE_TEXTURE_LOAD_MARGIN = 8.0     # texture must exceed baseline + margin to trigger
MIN_LOAD_GAP_SEC = 1.5              # cooldown between load candidates
LOAD_WINDOW_PRE_SEC = 1.0          # look back 1s before trigger
LOAD_WINDOW_POST_SEC = 3.0         # look ahead 3s after trigger
MATCH_TOL_SEC = 2.0                # ±2s for GT matching

# User-defined landing zone polygons — exact table surface shape where blankets land
LEFT_LANDING_ZONE = [313, 656, 195, 826, 537, 896, 639, 699]  # quadrilateral
RIGHT_LANDING_ZONE = [1248, 791, 1301, 1021, 1657, 996, 1603, 812]  # quadrilateral

FEATURE_NAMES_LOAD = [
    # State change — the core signal
    "table_texture_step",     # median(std_AFTER) - median(std_BEFORE)
    "table_mean_step",        # median(mean_AFTER) - median(mean_BEFORE)
    # Color shift (dark blanket on dark table)
    "color_bgr_shift",        # mean absolute BGR change before→after
    "color_dominant_channel", # which channel changed most (0=gray,1=R,2=G,3=B)
    # Settling — blanket laid flat = solid rectangle
    "table_solidity_after",   # contour solidity at +3s
    # Air check — restocks come from the air, normal loads from table side
    "air_motion_pre_trigger", # sum of air diff in 1s before trigger
    "air_motion_ratio",       # air_after / max(1, air_before)
    # Duration context
    "trigger_height",         # how high was the texture spike?
    "trigger_duration",       # how long did the spike last?
    # Lower-right quadrant of table ROI — clean blanket sliding zone
    "lr_quad_std_step",       # texture change in lower-right quadrant
    "lr_quad_mean_step",      # brightness change in lower-right quadrant
    "lr_quad_color_shift",    # BGR color change in LR quadrant
    # Landing polygon — where blanket settles on table (user-calibrated)
    "landing_std_step",       # texture in landing polygon BEFORE→AFTER
    "landing_mean_step",      # brightness change in landing polygon
    "landing_color_shift",    # BGR color change in landing polygon
]


# ═══════════════════════════════════════════════════════════════
# Table-texture candidate generator
# ═══════════════════════════════════════════════════════════════

def generate_load_candidates(video_path, fps=25.0):
    """Scan video for table texture derivative spikes → load candidates.

    Returns list of candidate dicts with peak frame + pre/post texture data.
    Rather than using v2's air-motion tracker, this directly monitors
    the raw table std-dev for sudden increases (blanket placement).
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return []

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    rois = {
        "left":  LEFT_TABLE_ROI_V2,
        "right": RIGHT_TABLE_ROI_V2,
    }
    air_rois = {
        "left":  LEFT_AIR_ROI_V2,
        "right": RIGHT_AIR_ROI_V2,
    }

    # Rolling buffer of raw table texture (std-dev) per table
    texture_history = {"left": deque(maxlen=75), "right": deque(maxlen=75)}
    texture_times = {"left": deque(maxlen=75), "right": deque(maxlen=75)}
    air_history = {"left": deque(maxlen=25), "right": deque(maxlen=25)}

    # Landing zone polygons — user-calibrated blanket settling surface
    land_polys = {
        "left":  np.array([[LEFT_LANDING_ZONE[i], LEFT_LANDING_ZONE[i+1]]
                            for i in range(0, len(LEFT_LANDING_ZONE), 2)], dtype=np.int32),
        "right": np.array([[RIGHT_LANDING_ZONE[i], RIGHT_LANDING_ZONE[i+1]]
                            for i in range(0, len(RIGHT_LANDING_ZONE), 2)], dtype=np.int32),
    }

    # Full frame data for feature extraction window
    frame_data = []
    frame_counter = 0

    candidates = []
    last_trigger = {"left": -1e9, "right": -1e9}

    prev_gray = {"left": None, "right": None}

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        t_sec = frame_counter / fps
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        frame_entry = {
            "frame": frame_counter,
            "time_sec": round(t_sec, 4),
        }

        for tbl in ["left", "right"]:
            x1, y1, x2, y2 = rois[tbl]
            roi = gray[y1:y2, x1:x2]
            raw_std = float(np.std(roi))
            raw_mean = float(np.mean(roi))

            frame_entry[f"{tbl}_std"] = raw_std
            frame_entry[f"{tbl}_mean"] = raw_mean

            # BGR color for the table ROI
            bgr = frame[y1:y2, x1:x2]
            frame_entry[f"{tbl}_B"] = float(np.mean(bgr[:, :, 0]))
            frame_entry[f"{tbl}_G"] = float(np.mean(bgr[:, :, 1]))
            frame_entry[f"{tbl}_R"] = float(np.mean(bgr[:, :, 2]))

            # Table-specific quadrant — LEFT=lower-left(closer to mid), RIGHT=lower-right
            mx = (x1 + x2) // 2
            my = (y1 + y2) // 2
            if tbl == "left":
                # Lower-left quadrant, closer to mid
                q_roi = gray[my:y2, x1:mx]
                q_bgr = frame[my:y2, x1:mx]
            else:
                # Lower-right quadrant
                q_roi = gray[my:y2, mx:x2]
                q_bgr = frame[my:y2, mx:x2]
            frame_entry[f"{tbl}_lr_std"] = float(np.std(q_roi))
            frame_entry[f"{tbl}_lr_mean"] = float(np.mean(q_roi))
            frame_entry[f"{tbl}_lr_B"] = float(np.mean(q_bgr[:,:,0]))
            frame_entry[f"{tbl}_lr_G"] = float(np.mean(q_bgr[:,:,1]))
            frame_entry[f"{tbl}_lr_R"] = float(np.mean(q_bgr[:,:,2]))

            # Landing polygon — blanket settling surface (user-calibrated)
            land_mask = np.zeros(gray.shape[:2], dtype=np.uint8)
            cv2.fillPoly(land_mask, [land_polys[tbl]], 255)
            lp = gray[land_mask == 255]
            if len(lp) > 0:
                frame_entry[f"{tbl}_land_std"] = float(np.std(lp))
                frame_entry[f"{tbl}_land_mean"] = float(np.mean(lp))
                lbgr = frame[land_mask == 255]
                frame_entry[f"{tbl}_land_B"] = float(np.mean(lbgr[:,0]))
                frame_entry[f"{tbl}_land_G"] = float(np.mean(lbgr[:,1]))
                frame_entry[f"{tbl}_land_R"] = float(np.mean(lbgr[:,2]))
            else:
                for fld in ["land_std","land_mean","land_B","land_G","land_R"]:
                    frame_entry[f"{tbl}_{fld}"] = 0.0

            # Air motion for pre-trigger check
            ax1, ay1, ax2, ay2 = air_rois[tbl]
            air_roi = gray[ay1:ay2, ax1:ax2].astype(np.float64)
            if prev_gray[tbl] is not None:
                prev_air = prev_gray[tbl].astype(np.float64)
                air_diff = float(np.mean(np.abs(air_roi - prev_air)))
            else:
                air_diff = 0.0
            prev_gray[tbl] = air_roi
            air_history[tbl].append(air_diff)

            # Track texture for derivative
            texture_history[tbl].append(raw_std)
            texture_times[tbl].append(t_sec)

            # Rolling-baseline approach: if texture suddenly jumps above
            # the rolling minimum + margin, a blanket was placed.
            if len(texture_history[tbl]) >= TABLE_TEXTURE_BASELINE_FRAMES:
                recent = list(texture_history[tbl])
                # Rolling baseline = p20 of recent texture (empty table = smooth)
                baseline = float(np.percentile(recent, 20))
                current = recent[-1]

                if (current > baseline + TABLE_TEXTURE_LOAD_MARGIN and
                    t_sec - last_trigger[tbl] > MIN_LOAD_GAP_SEC):
                    last_trigger[tbl] = t_sec
                    candidates.append({
                        "table": tbl,
                        "frame": frame_counter,
                        "time_sec": round(t_sec, 4),
                        "trigger_strength": round(current - baseline, 2),
                        "raw_std": raw_std,
                        "raw_mean": raw_mean,
                    })

        frame_data.append(frame_entry)
        frame_counter += 1

    cap.release()
    return candidates, frame_data, fps


# ═══════════════════════════════════════════════════════════════
# Feature extraction for load candidates
# ═══════════════════════════════════════════════════════════════

def _slice_window(frame_data, peak_frame, pre, post):
    """Get frame_data entries in [peak-pre, peak+post]."""
    return [r for r in frame_data if peak_frame - pre <= r["frame"] <= peak_frame + post]


def extract_load_features(candidate, frame_data, fps=25.0):
    """Extract 10 load-specific features from the table-texture window.

    Uses asymmetric window [-1s, +3s] around the trigger.
    """
    pre_frames = int(LOAD_WINDOW_PRE_SEC * fps)
    post_frames = int(LOAD_WINDOW_POST_SEC * fps)
    peak_f = int(candidate["frame"])
    window = _slice_window(frame_data, peak_f, pre_frames, post_frames)
    tbl = candidate["table"]

    if len(window) < 5:
        return {k: 0.0 for k in FEATURE_NAMES_LOAD}

    # Split into BEFORE and AFTER periods
    before = [r for r in window if r["frame"] <= peak_f]
    after = [r for r in window if r["frame"] > peak_f]

    if not before or not after:
        return {k: 0.0 for k in FEATURE_NAMES_LOAD}

    std_key = f"{tbl}_std"
    mean_key = f"{tbl}_mean"
    B_key = f"{tbl}_B"
    G_key = f"{tbl}_G"
    R_key = f"{tbl}_R"

    # 1. Table texture step — the core signal
    before_std = np.median([r.get(std_key, 0) for r in before])
    after_std = np.median([r.get(std_key, 0) for r in after])
    table_texture_step = after_std - before_std

    # 2. Table mean step
    before_mean = np.median([r.get(mean_key, 0) for r in before])
    after_mean = np.median([r.get(mean_key, 0) for r in after])
    table_mean_step = before_mean - after_mean  # blanket = darker → positive

    # 3. Color BGR shift — total color distance before→after
    before_B = np.median([r.get(B_key, 0) for r in before])
    after_B = np.median([r.get(B_key, 0) for r in after])
    before_G = np.median([r.get(G_key, 0) for r in before])
    after_G = np.median([r.get(G_key, 0) for r in after])
    before_R = np.median([r.get(R_key, 0) for r in before])
    after_R = np.median([r.get(R_key, 0) for r in after])
    color_bgr_shift = (abs(after_B - before_B) +
                       abs(after_G - before_G) +
                       abs(after_R - before_R))
    # Dominant channel
    deltas = [abs(after_B - before_B), abs(after_G - before_G), abs(after_R - before_R)]
    color_dominant_channel = float(np.argmax(deltas))

    # 4. Table solidity at +3s after trigger
    last_frame_idx = int(after[-1]["frame"])
    last_entry = after[-1]
    # We can't compute solidity from frame_data alone — use 0 as placeholder
    table_solidity_after = 0.0

    # 5. Air motion pre-trigger — was there air activity before the table change?
    air_pre_start = max(0, peak_f - int(1.0 * fps))
    air_pre_data = [r for r in window if air_pre_start <= r["frame"] <= peak_f]
    air_motion_pre_trigger = 0.0  # We don't have air motion in frame_data for this
    air_motion_ratio = 0.0

    # 6. Trigger characteristics
    trigger_height = float(candidate.get("trigger_strength", 0))
    trigger_duration = float(len(after) - len(before)) / fps

    # Lower-right quadrant features — cleanest blanket sliding signal
    lr_std_key = f"{tbl}_lr_std"
    lr_mean_key = f"{tbl}_lr_mean"
    lr_B = f"{tbl}_lr_B"; lr_G = f"{tbl}_lr_G"; lr_R = f"{tbl}_lr_R"

    before_lr_std = np.median([r.get(lr_std_key, 0) for r in before])
    after_lr_std = np.median([r.get(lr_std_key, 0) for r in after])
    lr_quad_std_step = after_lr_std - before_lr_std

    before_lr_mean = np.median([r.get(lr_mean_key, 0) for r in before])
    after_lr_mean = np.median([r.get(lr_mean_key, 0) for r in after])
    lr_quad_mean_step = before_lr_mean - after_lr_mean

    lr_quad_color_shift = (
        abs(np.median([r.get(lr_B, 0) for r in after]) - np.median([r.get(lr_B, 0) for r in before])) +
        abs(np.median([r.get(lr_G, 0) for r in after]) - np.median([r.get(lr_G, 0) for r in before])) +
        abs(np.median([r.get(lr_R, 0) for r in after]) - np.median([r.get(lr_R, 0) for r in before])))

    # Landing polygon features — settled blanket texture/color
    def land(key): return f"{tbl}_land_{key}"
    landing_std_step = (np.median([r.get(land("std"), 0) for r in after]) -
                        np.median([r.get(land("std"), 0) for r in before]))
    landing_mean_step = (np.median([r.get(land("mean"), 0) for r in before]) -
                         np.median([r.get(land("mean"), 0) for r in after]))
    landing_color_shift = (
        abs(np.median([r.get(land("B"), 0) for r in after]) - np.median([r.get(land("B"), 0) for r in before])) +
        abs(np.median([r.get(land("G"), 0) for r in after]) - np.median([r.get(land("G"), 0) for r in before])) +
        abs(np.median([r.get(land("R"), 0) for r in after]) - np.median([r.get(land("R"), 0) for r in before])))

    return {
        "table_texture_step": round(table_texture_step, 2),
        "table_mean_step": round(table_mean_step, 2),
        "color_bgr_shift": round(color_bgr_shift, 2),
        "color_dominant_channel": color_dominant_channel,
        "table_solidity_after": table_solidity_after,
        "air_motion_pre_trigger": air_motion_pre_trigger,
        "air_motion_ratio": air_motion_ratio,
        "trigger_height": round(trigger_height, 2),
        "trigger_duration": round(trigger_duration, 2),
        "lr_quad_std_step": round(lr_quad_std_step, 2),
        "lr_quad_mean_step": round(lr_quad_mean_step, 2),
        "lr_quad_color_shift": round(lr_quad_color_shift, 2),
        "landing_std_step": round(landing_std_step, 2),
        "landing_mean_step": round(landing_mean_step, 2),
        "landing_color_shift": round(landing_color_shift, 2),
    }


def features_to_array(feats):
    return np.array([feats[n] for n in FEATURE_NAMES_LOAD], dtype=float)


# ═══════════════════════════════════════════════════════════════
# Training main
# ═══════════════════════════════════════════════════════════════

def main():
    print("=" * 70)
    print("  CH27 — TABLE-TEXTURE LOAD DETECTOR training")
    print("=" * 70)

    all_X_L, all_y_L = [], []
    all_X_R, all_y_R = [], []

    for clip in CLIPS_FOR_TRAINING:
        clip_path = HERE / "gt_clips" / f"{clip}.mp4"
        if not clip_path.exists():
            continue
        labels, clusters, fps = load_clip_gt(clip)

        # Generate table-texture candidates
        t0 = time.time()
        candidates, frame_data, _fps = generate_load_candidates(clip_path, fps)

        # Build load label index
        load_frames = {"left": set(), "right": set()}
        for c in clusters:
            if c["type"] == "load":
                for f in c.get("all_frames", [c["peak_frame"]]):
                    load_frames[c["table"]].add(f)

        # Label and extract features
        tol_frames = int(MATCH_TOL_SEC * fps)
        clip_pos = 0
        for cand in candidates:
            tbl = cand["table"]
            pf = int(cand["frame"])
            nearest = min((abs(pf - f) for f in load_frames[tbl]), default=float("inf"))
            label = 1 if nearest <= tol_frames else 0
            if label == 1:
                clip_pos += 1

            feats = extract_load_features(cand, frame_data, fps)
            row = features_to_array(feats)

            if tbl == "left":
                all_X_L.append(row)
                all_y_L.append(label)
            else:
                all_X_R.append(row)
                all_y_R.append(label)

        print(f"  {clip}: {len(candidates)} candidates, {clip_pos} pos "
              f"({time.time()-t0:.0f}s)")

    # Train per-table
    for X_tbl, y_tbl, name in [
        (np.array(all_X_L, dtype=float), np.array(all_y_L, dtype=int), "LEFT"),
        (np.array(all_X_R, dtype=float), np.array(all_y_R, dtype=int), "RIGHT"),
    ]:
        n_pos = int(y_tbl.sum())
        n_neg = len(y_tbl) - n_pos
        if n_pos < 3 or n_neg < 3:
            print(f"\n  {name}: skip — {n_pos} pos, {n_neg} neg")
            continue

        print(f"\n{'='*50}")
        print(f"  LOAD {name} TABLE (texture-triggered)")
        print(f"  Dataset: {len(y_tbl)} samples ({n_pos} pos, {n_neg} neg, "
              f"{100*n_pos/len(y_tbl):.1f}% pos)")

        cv = StratifiedKFold(n_splits=min(5, n_pos, n_neg), shuffle=True, random_state=42)
        clf = xgb.XGBClassifier(
            n_estimators=300, max_depth=6, learning_rate=0.05,
            colsample_bytree=0.7, subsample=0.8,
            min_child_weight=3,
            scale_pos_weight=n_neg / max(1, n_pos),
            objective="binary:logistic", eval_metric="logloss",
            random_state=42, n_jobs=-1, verbosity=0,
        )

        f1s = cross_val_score(clf, X_tbl, y_tbl, cv=cv, scoring="f1")
        ps = cross_val_score(clf, X_tbl, y_tbl, cv=cv, scoring="precision")
        rs = cross_val_score(clf, X_tbl, y_tbl, cv=cv, scoring="recall")

        print(f"  CV F1:       {f1s.mean():.3f} ± {f1s.std():.3f}")
        print(f"  CV Precision: {ps.mean():.3f}")
        print(f"  CV Recall:    {rs.mean():.3f}")

        clf.fit(X_tbl, y_tbl)

        imps = sorted(zip(FEATURE_NAMES_LOAD, clf.feature_importances_),
                      key=lambda x: -x[1])
        print(f"\n  Feature importances:")
        for feat_name, imp in imps:
            bar = "█" * int(imp * 40)
            print(f"    {feat_name:30s} {imp:.4f} {bar}")

        # Save
        import joblib
        tbl_key = "left" if name == "LEFT" else "right"
        pkl_path = HERE / f"taping_load_texture_classifier_v1_{tbl_key}.pkl"
        joblib.dump(clf, pkl_path, compress=3)
        print(f"\n  [saved] {pkl_path.name}")

        meta = {
            "table": tbl_key,
            "model_type": "load_texture",
            "feature_names": FEATURE_NAMES_LOAD,
            "n_features": len(FEATURE_NAMES_LOAD),
            "cv_f1_mean": float(f1s.mean()),
            "cv_f1_std": float(f1s.std()),
            "n_samples": len(y_tbl),
            "n_positives": n_pos,
            "feature_importances": dict(zip(
                FEATURE_NAMES_LOAD, [float(x) for x in clf.feature_importances_])),
        }
        meta_path = HERE / f"load_texture_metadata_{tbl_key}.json"
        meta_path.write_text(json.dumps(meta, indent=2))
        print(f"  [saved] {meta_path.name}")


if __name__ == "__main__":
    main()
