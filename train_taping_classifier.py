#!/usr/bin/env python3
"""CH27 v4 classifier training pipeline.

Reads `gt_clips/*.labels.json` (manual GT from gt_labeler / gt_labeler_web),
clusters adjacent frame-labels into action windows, runs the v2 detector in
candidate-collection mode on each clip, matches candidates against GT, extracts
features for each candidate, trains a RandomForest, and saves the model.

Run:
    python3 train_taping_classifier.py
    python3 train_taping_classifier.py --coverage-only   # diagnostic step
    python3 train_taping_classifier.py --hold-out gt_clip4_afternoon_dark
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

# ─────────────────────────────────────────────────────────────────
#  Configuration
# ─────────────────────────────────────────────────────────────────

GT_DIR = HERE / "gt_clips"
CLIPS_FOR_TRAINING = [
    "gt_clip1_morning",
    "gt_clip2_prelunch",
    "gt_clip3_postlunch",
    "gt_clip4_afternoon_dark",
]
CLIP_SKIP = ["gt_clip5_endofday"]  # only 2 stray labels — not enough to train on

CLUSTER_GAP_FRAMES = 25       # ≤1s gap → same action window
COVERAGE_TOL_SEC = 2.0        # GT cluster covered if v2 candidate within ±2s
MATCH_TOL_SEC = 1.5           # candidate is "positive" if within ±1.5s of GT cluster

# Feature extraction window (frames before/after peak)
FEATURE_PRE = 50    # 2.0 s pre-peak  (allows pre_peak_table_max + baseline)
FEATURE_POST = 38   # 1.5 s post-peak (allows post_peak_table_min + decay)
AIR_THRESH_FOR_SHAPE = 3.0  # threshold used for duration / AUC calculations

FEATURE_NAMES = [
    # Air-motion shape
    "peak_height",
    "peak_above_baseline",
    "duration_above_thresh_sec",
    "rise_time_sec",
    "decay_time_sec",
    "skewness",
    "auc_above_thresh",
    # Table-state context
    "pre_peak_table_max",
    "post_peak_table_min",
    "table_drop",
    "table_signal_at_peak",
    # Cross-table artifact rejection
    "simultaneous_air_other_table",
    "air_diff_to_other",
]


# ─────────────────────────────────────────────────────────────────
#  Phase 1: GT clustering
# ─────────────────────────────────────────────────────────────────

def cluster_labels(labels, gap_frames=CLUSTER_GAP_FRAMES):
    """Cluster a list of label dicts into action windows.

    Groups by (table, type), sorts by frame, merges adjacent labels whose
    frame gap ≤ gap_frames.

    Returns list of dicts: {table, type, peak_frame, start_frame, end_frame,
    n_labels}
    """
    by_key = {}
    for l in labels:
        by_key.setdefault((l["table"], l["type"]), []).append(l)

    clusters = []
    for (tbl, typ), arr in by_key.items():
        arr = sorted(arr, key=lambda l: l["frame"])
        cur = [arr[0]]
        for l in arr[1:]:
            if l["frame"] - cur[-1]["frame"] <= gap_frames:
                cur.append(l)
            else:
                clusters.append(_finalize_cluster(cur, tbl, typ))
                cur = [l]
        clusters.append(_finalize_cluster(cur, tbl, typ))
    return clusters


def _finalize_cluster(arr, table, typ):
    frames = [l["frame"] for l in arr]
    return {
        "table": table,
        "type": typ,
        "start_frame": int(min(frames)),
        "end_frame": int(max(frames)),
        "peak_frame": int(np.median(frames)),  # center-of-mass (median frame)
        "n_labels": len(arr),
        "span_sec": (max(frames) - min(frames)) / 25.0,
    }


def load_clip_gt(clip_name):
    """Load and cluster a single clip's GT. Returns (labels, clusters, fps)."""
    path = GT_DIR / f"{clip_name}.labels.json"
    data = json.loads(path.read_text())
    labels = data["labels"]
    fps = data.get("fps", 25.0)
    clusters = cluster_labels(labels)
    return labels, clusters, fps


def print_cluster_summary(clip_name, clusters):
    print(f"\n=== {clip_name} ===")
    by_kt = {}
    for c in clusters:
        by_kt.setdefault((c["table"], c["type"]), []).append(c)
    for (tbl, typ) in [("left", "load"), ("left", "toss"),
                       ("right", "load"), ("right", "toss")]:
        arr = by_kt.get((tbl, typ), [])
        if not arr:
            print(f"  {tbl:5s} {typ:5s}  0 clusters")
            continue
        spans = [c["span_sec"] for c in arr]
        long_spans = sum(1 for s in spans if s > 1.5)
        print(f"  {tbl:5s} {typ:5s}  {len(arr):3d} clusters · "
              f"avg_size={np.mean([c['n_labels'] for c in arr]):.1f} · "
              f"median_span={np.median(spans):.2f}s · "
              f"max_span={max(spans):.1f}s · "
              f"{long_spans} clusters >1.5s span (review)")


# ─────────────────────────────────────────────────────────────────
#  Phase 2: Candidate-collection (lazy import — depends on cv2 etc.)
# ─────────────────────────────────────────────────────────────────

def collect_candidates(clip_path, fps=25.0):
    """Run v2 detector with VERY lowered thresholds and NO gating.
    Returns list of candidate dicts: {table, peak_frame, peak_t, peak_air,
    peak_signal, frame_window_data} where frame_window_data is a dict of
    per-frame signal arrays (mean, std, signal, air_motion) for peak±25 frames.
    """
    from taping_counter import TapingCounter

    # Override: lowered thresholds, no cooldown/min_cycle/strong gates,
    # context still permissive
    counter = TapingCounter(
        str(clip_path),
        version="v2",
        # AIR thresholds — very low so we catch every plausible pulse
        air_toss_thresh_left=3.0,
        air_toss_thresh_right=2.0,
        air_toss_thresh_left_low=3.0,
        air_toss_thresh_right_low=2.0,
        # Disable gates
        load_strong=0.0,
        context_signal_thresh=0.0,
        min_gap_sec=0.5,
        min_cycle_sec=0.5,
        idle_to_conservative_sec=99999,  # never enter conservative mode
        # Frame data sampling at full rate so we have signals for feature extraction
        frame_data_every=1,
    )
    results = counter.run()
    # The events list is what passed through v2's pulse-end logic; we want
    # ALL pulse-ends regardless of gates. Achieved here by setting all gates
    # to permissive values above. Verify by counting.
    return results


# ─────────────────────────────────────────────────────────────────
#  Phase 3: Feature extraction
# ─────────────────────────────────────────────────────────────────

def _slice_frame_data(frame_data, peak_frame, pre, post):
    """Return the slice of frame_data covering [peak-pre, peak+post] (inclusive).
    frame_data is a list of dicts with a 'frame' key. Caller must ensure
    frame_data was collected at full rate (frame_data_every=1).
    """
    if not frame_data:
        return []
    # frame_data is ordered by frame; binary search would be faster but this
    # runs ~150 times per clip and we have ~7500 frames, so linear is fine
    start = peak_frame - pre
    end = peak_frame + post
    return [r for r in frame_data if start <= r["frame"] <= end]


def extract_features(candidate, frame_data):
    """Compute the 13 feature dict for one candidate.

    candidate:  {table, frame, time_sec, ...}
    frame_data: full per-frame signals from candidate-collection mode
                (each row has left_/right_ {mean, std, signal, air_motion})
    """
    table = candidate["table"]
    other = "right" if table == "left" else "left"
    peak_frame = int(candidate["frame"])
    peak_t = float(candidate["time_sec"])

    # Slice the signal window
    window = _slice_frame_data(frame_data, peak_frame, FEATURE_PRE, FEATURE_POST)
    if not window:
        # Should not happen with full-rate frame_data
        return {k: 0.0 for k in FEATURE_NAMES}

    air_key = f"{table}_air_motion"
    other_air_key = f"{other}_air_motion"
    sig_key = f"{table}_signal"

    air_series = np.array([r[air_key] for r in window], dtype=float)
    other_air_series = np.array([r[other_air_key] for r in window], dtype=float)
    sig_series = np.array([r[sig_key] for r in window], dtype=float)
    times = np.array([r["time_sec"] for r in window], dtype=float)

    # Locate the peak within the window (could differ slightly from candidate
    # due to smoothing / sampling). Use the actual max in window.
    peak_idx = int(np.argmax(air_series))
    peak_height = float(air_series[peak_idx])

    # ── Air shape features ──
    # Baseline = mean of air motion in [-2.0s, -0.8s] before peak
    pre_baseline_start = max(0, peak_idx - int(2.0 * 25))
    pre_baseline_end = max(0, peak_idx - int(0.8 * 25))
    if pre_baseline_end > pre_baseline_start:
        air_baseline = float(np.mean(air_series[pre_baseline_start:pre_baseline_end]))
    else:
        air_baseline = float(np.mean(air_series[: max(1, peak_idx)]))
    peak_above_baseline = peak_height - air_baseline

    above = air_series > AIR_THRESH_FOR_SHAPE
    duration_above_thresh_sec = float(above.sum()) / 25.0
    auc_above_thresh = float(np.maximum(air_series - AIR_THRESH_FOR_SHAPE, 0).sum()) / 25.0

    # Rise time: from last threshold-crossing BEFORE peak to peak
    rise_time_sec = 0.0
    for i in range(peak_idx, -1, -1):
        if air_series[i] < AIR_THRESH_FOR_SHAPE:
            rise_time_sec = (peak_idx - i) / 25.0
            break
    else:
        rise_time_sec = peak_idx / 25.0

    # Decay time: from peak to first time air drops to half-peak after the peak
    half_peak = peak_height * 0.5
    decay_time_sec = 0.0
    for i in range(peak_idx + 1, len(air_series)):
        if air_series[i] <= half_peak:
            decay_time_sec = (i - peak_idx) / 25.0
            break
    else:
        decay_time_sec = (len(air_series) - 1 - peak_idx) / 25.0

    skewness = decay_time_sec / max(0.04, rise_time_sec)  # 0.04s = 1 frame floor

    # ── Table-state context ──
    # Pre-peak table signal max in [-2.0s, peak]
    pre_peak_start = max(0, peak_idx - int(2.0 * 25))
    pre_peak_table_max = float(np.max(sig_series[pre_peak_start: peak_idx + 1])) if peak_idx > 0 else float(sig_series[0])
    # Post-peak table signal min in [peak, +1.5s]
    post_peak_end = min(len(sig_series), peak_idx + int(1.5 * 25))
    post_peak_table_min = float(np.min(sig_series[peak_idx: post_peak_end])) if post_peak_end > peak_idx else float(sig_series[peak_idx])
    table_drop = pre_peak_table_max - post_peak_table_min
    table_signal_at_peak = float(sig_series[peak_idx])

    # ── Cross-table (artifact rejection) ──
    # Simultaneous air motion on the OTHER table at peak ± 3 frames
    sim_lo = max(0, peak_idx - 3)
    sim_hi = min(len(other_air_series), peak_idx + 4)
    simultaneous_air_other_table = float(np.max(other_air_series[sim_lo: sim_hi]))
    air_diff_to_other = peak_height - simultaneous_air_other_table

    return {
        "peak_height": peak_height,
        "peak_above_baseline": peak_above_baseline,
        "duration_above_thresh_sec": duration_above_thresh_sec,
        "rise_time_sec": rise_time_sec,
        "decay_time_sec": decay_time_sec,
        "skewness": skewness,
        "auc_above_thresh": auc_above_thresh,
        "pre_peak_table_max": pre_peak_table_max,
        "post_peak_table_min": post_peak_table_min,
        "table_drop": table_drop,
        "table_signal_at_peak": table_signal_at_peak,
        "simultaneous_air_other_table": simultaneous_air_other_table,
        "air_diff_to_other": air_diff_to_other,
    }


def features_to_array(feat_dict):
    """Convert a feature dict to a row in the canonical feature order.
    Returns np.ndarray of shape (n_features,).
    """
    return np.array([feat_dict[name] for name in FEATURE_NAMES], dtype=float)


# ─────────────────────────────────────────────────────────────────
#  Phase 3b: Match candidates to GT clusters
# ─────────────────────────────────────────────────────────────────

def label_candidates(candidates, gt_clusters, tol_sec=MATCH_TOL_SEC, fps=25.0):
    """Assign each candidate a binary label.

    A candidate is POSITIVE (label=1) if there is a GT cluster of the same
    table with peak_frame within ±tol_sec * fps frames of the candidate.
    Otherwise label=0.

    Returns list of (candidate, label) tuples.
    """
    # Pre-index GT clusters by table for fast lookup. Only TOSS clusters
    # are positives for the toss classifier (loads are out of scope for v4).
    toss_clusters_by_table = {"left": [], "right": []}
    for c in gt_clusters:
        if c["type"] == "toss":
            toss_clusters_by_table[c["table"]].append(c["peak_frame"])
    for tbl in toss_clusters_by_table:
        toss_clusters_by_table[tbl].sort()

    tol_frames = int(tol_sec * fps)
    out = []
    for cand in candidates:
        tbl = cand["table"]
        peak_f = int(cand["frame"])
        nearest = min(
            (abs(peak_f - g) for g in toss_clusters_by_table[tbl]),
            default=float("inf"),
        )
        label = 1 if nearest <= tol_frames else 0
        out.append((cand, label))
    return out


# ─────────────────────────────────────────────────────────────────
#  Phase 0c: Coverage diagnostic
# ─────────────────────────────────────────────────────────────────

def coverage_check(clip_name):
    """For each GT cluster, find nearest v2 candidate (same table+type) and
    check whether it's within COVERAGE_TOL_SEC of the cluster's peak.
    """
    labels, clusters, fps = load_clip_gt(clip_name)
    print_cluster_summary(clip_name, clusters)

    clip_path = GT_DIR / f"{clip_name}.mp4"
    if not clip_path.exists():
        print(f"  [skip] {clip_path} not found")
        return None

    print(f"\n  Running candidate collection on {clip_path.name}…")
    t0 = time.time()
    results = collect_candidates(clip_path, fps=fps)
    elapsed = time.time() - t0
    print(f"  → {len(results['events'])} candidates in {elapsed:.0f}s")

    # Match each TOSS-only GT cluster to nearest candidate of same table
    toss_clusters = [c for c in clusters if c["type"] == "toss"]
    candidates = results["events"]  # each has {table, time_sec, frame, ...}

    cov_per_table = {"left": [], "right": []}
    for c in toss_clusters:
        tbl = c["table"]
        peak_t = c["peak_frame"] / fps
        # Find nearest same-table candidate
        same_tbl = [e for e in candidates if e["table"] == tbl]
        if not same_tbl:
            cov_per_table[tbl].append(None)
            continue
        nearest = min(same_tbl, key=lambda e: abs(e["time_sec"] - peak_t))
        delta = abs(nearest["time_sec"] - peak_t)
        cov_per_table[tbl].append(delta)

    print("  Coverage (TOSS clusters → nearest candidate):")
    for tbl in ["left", "right"]:
        deltas = cov_per_table[tbl]
        if not deltas:
            continue
        n = len(deltas)
        n_covered = sum(1 for d in deltas if d is not None and d <= COVERAGE_TOL_SEC)
        n_missing = sum(1 for d in deltas if d is None)
        n_far = sum(1 for d in deltas if d is not None and d > COVERAGE_TOL_SEC)
        cov_pct = 100 * n_covered / n if n else 0
        print(f"    {tbl:5s}  GT={n:3d}  covered={n_covered:3d} ({cov_pct:5.1f}%)  "
              f"miss={n_missing}  far(>2s)={n_far}")

    return cov_per_table


# ─────────────────────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────────────────────

def build_dataset_for_clip(clip_name, verbose=True):
    """Run candidate-collection on a clip, extract features for each candidate,
    label vs GT clusters. Returns (X, y, candidates) where:
      X : np.ndarray (N, n_features)
      y : np.ndarray (N,) of {0, 1}
      candidates : list of candidate dicts (with 'features' attached)
    """
    labels, clusters, fps = load_clip_gt(clip_name)
    clip_path = GT_DIR / f"{clip_name}.mp4"
    if not clip_path.exists():
        if verbose: print(f"[skip] {clip_path} not found")
        return None

    if verbose: print(f"\n[{clip_name}] candidate collection…")
    t0 = time.time()
    results = collect_candidates(clip_path, fps=fps)
    candidates = results.get("events", [])
    frame_data = results.get("frame_data", [])
    if verbose:
        print(f"  → {len(candidates)} candidates · "
              f"{len(frame_data)} frame_data rows · "
              f"{time.time()-t0:.0f}s")

    if verbose: print(f"[{clip_name}] feature extraction…")
    X_rows, y_rows, enriched = [], [], []
    labelled = label_candidates(candidates, clusters, fps=fps)
    for cand, y in labelled:
        feats = extract_features(cand, frame_data)
        cand_with_feats = dict(cand)
        cand_with_feats["features"] = feats
        cand_with_feats["label"] = y
        cand_with_feats["clip"] = clip_name
        enriched.append(cand_with_feats)
        X_rows.append(features_to_array(feats))
        y_rows.append(y)

    X = np.array(X_rows, dtype=float)
    y = np.array(y_rows, dtype=int)
    if verbose:
        n_pos = int(y.sum())
        print(f"  → X={X.shape}  positives={n_pos}  negatives={len(y)-n_pos}  "
              f"pos_rate={100*n_pos/max(1,len(y)):.1f}%")
    return X, y, enriched


# ─────────────────────────────────────────────────────────────────
#  Phase 4: Train RandomForest, hold out clip 4
# ─────────────────────────────────────────────────────────────────

def train_classifier(hold_out="gt_clip4_afternoon_dark", out_dir=HERE):
    """Build dataset across all training clips, train RandomForest, save .pkl."""
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import StratifiedKFold, cross_val_score
    from sklearn.metrics import (
        precision_score, recall_score, f1_score, classification_report,
        confusion_matrix,
    )
    import joblib

    # Build per-clip datasets (also used for held-out scoring)
    print("=" * 70)
    print(f"  CH27 v4 — TRAIN CLASSIFIER  (hold-out: {hold_out})")
    print("=" * 70)
    per_clip = {}
    for clip in CLIPS_FOR_TRAINING:
        res = build_dataset_for_clip(clip)
        if res is None: continue
        per_clip[clip] = res

    # Concatenate train clips
    train_clips = [c for c in CLIPS_FOR_TRAINING if c != hold_out and c in per_clip]
    print(f"\n[train] clips: {train_clips}")
    print(f"[hold-out] clip: {hold_out}")

    X_train = np.concatenate([per_clip[c][0] for c in train_clips], axis=0)
    y_train = np.concatenate([per_clip[c][1] for c in train_clips], axis=0)
    X_test = per_clip[hold_out][0] if hold_out in per_clip else None
    y_test = per_clip[hold_out][1] if hold_out in per_clip else None

    print(f"\n[dataset]")
    print(f"  TRAIN: X={X_train.shape}  positives={int(y_train.sum())}  "
          f"negatives={len(y_train)-int(y_train.sum())}  "
          f"pos_rate={100*y_train.sum()/len(y_train):.1f}%")
    if X_test is not None:
        print(f"  TEST:  X={X_test.shape}  positives={int(y_test.sum())}  "
              f"negatives={len(y_test)-int(y_test.sum())}  "
              f"pos_rate={100*y_test.sum()/len(y_test):.1f}%")

    # 5-fold stratified CV on the train set
    print("\n[5-fold cross-validation on train clips]")
    base_clf = RandomForestClassifier(
        n_estimators=300, max_depth=8,
        class_weight="balanced", random_state=42, n_jobs=-1,
        min_samples_leaf=3,
    )
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_f1 = cross_val_score(base_clf, X_train, y_train, cv=cv, scoring="f1")
    cv_p = cross_val_score(base_clf, X_train, y_train, cv=cv, scoring="precision")
    cv_r = cross_val_score(base_clf, X_train, y_train, cv=cv, scoring="recall")
    print(f"  F1        : {cv_f1.mean():.3f} ± {cv_f1.std():.3f}  "
          f"(folds: {[f'{x:.2f}' for x in cv_f1]})")
    print(f"  Precision : {cv_p.mean():.3f} ± {cv_p.std():.3f}")
    print(f"  Recall    : {cv_r.mean():.3f} ± {cv_r.std():.3f}")

    # Find best threshold via CV (NOT using held-out, to avoid leakage)
    print("\n[threshold tuning] via OOF predictions on train")
    from sklearn.model_selection import cross_val_predict
    oof_prob = cross_val_predict(base_clf, X_train, y_train, cv=cv,
                                   method="predict_proba", n_jobs=-1)[:, 1]
    best_thresh, best_f1, best_p, best_r = 0.5, 0.0, 0.0, 0.0
    for th in np.arange(0.30, 0.81, 0.02):
        yp = (oof_prob > th).astype(int)
        if yp.sum() == 0: continue
        p_ = precision_score(y_train, yp, zero_division=0)
        r_ = recall_score(y_train, yp, zero_division=0)
        f_ = f1_score(y_train, yp, zero_division=0)
        if f_ > best_f1:
            best_thresh, best_f1, best_p, best_r = th, f_, p_, r_
    print(f"  Best threshold = {best_thresh:.2f}  "
          f"→ CV F1={best_f1:.3f} (P={best_p:.3f}, R={best_r:.3f})")
    decision_threshold = float(best_thresh)

    # Final fit on ALL training data
    print("\n[final fit] on full train set")
    clf = RandomForestClassifier(
        n_estimators=300, max_depth=8,
        class_weight="balanced", random_state=42, n_jobs=-1,
        min_samples_leaf=3,
    )
    clf.fit(X_train, y_train)

    # Held-out scoring (with both default 0.5 and tuned threshold)
    p, r, f1 = None, None, None
    if X_test is not None:
        print(f"\n[held-out: {hold_out}]  (default threshold 0.5)")
        y_pred = clf.predict(X_test)
        y_prob = clf.predict_proba(X_test)[:, 1]
        p = precision_score(y_test, y_pred, zero_division=0)
        r = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        cm = confusion_matrix(y_test, y_pred)
        print(f"  Precision = {p:.3f}  Recall = {r:.3f}  F1 = {f1:.3f}")
        print(f"  CM: TN={cm[0,0]} FP={cm[0,1]} FN={cm[1,0]} TP={cm[1,1]}")

        print(f"\n[held-out]  (tuned threshold {decision_threshold:.2f})")
        y_pred_t = (y_prob > decision_threshold).astype(int)
        p_t = precision_score(y_test, y_pred_t, zero_division=0)
        r_t = recall_score(y_test, y_pred_t, zero_division=0)
        f1_t = f1_score(y_test, y_pred_t, zero_division=0)
        cm_t = confusion_matrix(y_test, y_pred_t)
        print(f"  Precision = {p_t:.3f}  Recall = {r_t:.3f}  F1 = {f1_t:.3f}")
        print(f"  CM: TN={cm_t[0,0]} FP={cm_t[0,1]} FN={cm_t[1,0]} TP={cm_t[1,1]}")
        # Use the tuned numbers in metadata
        p, r, f1 = p_t, r_t, f1_t

    # Feature importances
    print("\n[feature importances]")
    importances = sorted(
        zip(FEATURE_NAMES, clf.feature_importances_), key=lambda x: -x[1])
    for name, imp in importances:
        bar = "█" * int(imp * 60)
        print(f"  {name:32s}  {imp:.3f}  {bar}")

    # Save
    pkl_path = Path(out_dir) / "taping_pulse_classifier_toss_v4.pkl"
    joblib.dump(clf, pkl_path, compress=3)
    print(f"\n[saved] {pkl_path}")

    meta_path = Path(out_dir) / "classifier_metadata.json"
    metadata = {
        "feature_names": FEATURE_NAMES,
        "n_features": len(FEATURE_NAMES),
        "trained_at": datetime.now().isoformat(),
        "training_clips": train_clips,
        "held_out_clip": hold_out,
        "n_train_samples": int(X_train.shape[0]),
        "n_train_positives": int(y_train.sum()),
        "n_test_samples": int(X_test.shape[0]) if X_test is not None else 0,
        "n_test_positives": int(y_test.sum()) if X_test is not None else 0,
        "cv_f1_mean": float(cv_f1.mean()),
        "cv_f1_std": float(cv_f1.std()),
        "cv_f1_per_fold": [float(x) for x in cv_f1],
        "cv_precision_mean": float(cv_p.mean()),
        "cv_recall_mean": float(cv_r.mean()),
        "test_precision": float(p) if X_test is not None else None,
        "test_recall": float(r) if X_test is not None else None,
        "test_f1": float(f1) if X_test is not None else None,
        "feature_importances": dict(zip(
            FEATURE_NAMES, [float(x) for x in clf.feature_importances_])),
        "model": "RandomForestClassifier(n_estimators=300, max_depth=8, "
                 "min_samples_leaf=3, class_weight='balanced', random_state=42)",
        "decision_threshold": decision_threshold,
        "match_tol_sec": MATCH_TOL_SEC,
        "feature_pre_frames": FEATURE_PRE,
        "feature_post_frames": FEATURE_POST,
        "air_thresh_for_shape": AIR_THRESH_FOR_SHAPE,
    }
    meta_path.write_text(json.dumps(metadata, indent=2))
    print(f"[saved] {meta_path}")

    return clf, metadata


# ─────────────────────────────────────────────────────────────────
#  Phase 4 helpers: v2 baseline + leave-one-clip-out CV
# ─────────────────────────────────────────────────────────────────

def _score_predictions(times_pred, gt_clusters, table, fps=25.0,
                        tol_sec=MATCH_TOL_SEC):
    """Score a set of predicted toss timestamps against GT clusters.
    Returns precision, recall, F1, n_tp, n_fp, n_fn for ONE table.
    """
    gt_peaks = sorted(c["peak_frame"] / fps for c in gt_clusters
                      if c["table"] == table and c["type"] == "toss")
    preds = sorted(times_pred)
    used_gt = [False] * len(gt_peaks)
    used_pred = [False] * len(preds)
    # Greedy matching
    for i, p in enumerate(preds):
        best_j, best_d = -1, tol_sec
        for j, g in enumerate(gt_peaks):
            if used_gt[j]: continue
            d = abs(p - g)
            if d < best_d:
                best_d, best_j = d, j
        if best_j >= 0:
            used_gt[best_j] = True
            used_pred[i] = True
    tp = sum(used_pred)
    fp = len(preds) - tp
    fn = len(gt_peaks) - sum(used_gt)
    p = tp / max(1, tp + fp)
    r = tp / max(1, tp + fn)
    f = 2 * p * r / max(1e-9, p + r)
    return p, r, f, tp, fp, fn


def run_v2_baseline():
    """For each labeled clip, run v2 (production thresholds) and score against
    GT clusters. Establishes the baseline that v4 must beat per-clip."""
    from taping_counter import TapingCounter
    print("=" * 70)
    print("  CH27 v2 BASELINE — per-clip F1 vs GT clusters")
    print("=" * 70)
    overall = {"tp": 0, "fp": 0, "fn": 0}
    for clip in CLIPS_FOR_TRAINING:
        labels, clusters, fps = load_clip_gt(clip)
        clip_path = GT_DIR / f"{clip}.mp4"
        if not clip_path.exists(): continue
        print(f"\n[{clip}]")
        # Run v2 with PRODUCTION thresholds (defaults — no overrides)
        counter = TapingCounter(str(clip_path), version="v2", debug=False)
        results = counter.run()
        events = results.get("events", [])
        for tbl in ["left", "right"]:
            preds = [e["time_sec"] for e in events if e["table"] == tbl]
            p, r, f, tp, fp, fn = _score_predictions(preds, clusters, tbl, fps)
            overall["tp"] += tp; overall["fp"] += fp; overall["fn"] += fn
            print(f"  {tbl:5s}  pred={len(preds):3d}  GT={tp+fn:3d}  "
                  f"TP={tp:3d} FP={fp:3d} FN={fn:3d}  "
                  f"P={p:.2f} R={r:.2f} F1={f:.2f}")
    op = overall["tp"] / max(1, overall["tp"] + overall["fp"])
    or_ = overall["tp"] / max(1, overall["tp"] + overall["fn"])
    of = 2 * op * or_ / max(1e-9, op + or_)
    print(f"\nOVERALL  TP={overall['tp']}  FP={overall['fp']}  FN={overall['fn']}  "
          f"P={op:.2f} R={or_:.2f} F1={of:.2f}")


def run_leave_one_clip_out():
    """For each clip, train on the OTHER 3 and evaluate on this clip.
    Honest per-clip generalization estimate."""
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import precision_score, recall_score, f1_score
    print("=" * 70)
    print("  CH27 v4 — LEAVE-ONE-CLIP-OUT (honest per-clip generalization)")
    print("=" * 70)
    per_clip = {}
    for clip in CLIPS_FOR_TRAINING:
        res = build_dataset_for_clip(clip, verbose=False)
        if res is None: continue
        per_clip[clip] = res

    print()
    for hold in CLIPS_FOR_TRAINING:
        if hold not in per_clip: continue
        train = [c for c in CLIPS_FOR_TRAINING if c != hold and c in per_clip]
        X_tr = np.concatenate([per_clip[c][0] for c in train])
        y_tr = np.concatenate([per_clip[c][1] for c in train])
        X_te, y_te, _ = per_clip[hold]
        clf = RandomForestClassifier(
            n_estimators=300, max_depth=8, min_samples_leaf=3,
            class_weight="balanced", random_state=42, n_jobs=-1)
        clf.fit(X_tr, y_tr)
        y_pred = clf.predict(X_te)
        p = precision_score(y_te, y_pred, zero_division=0)
        r = recall_score(y_te, y_pred, zero_division=0)
        f1 = f1_score(y_te, y_pred, zero_division=0)
        print(f"  hold={hold:30s}  P={p:.3f}  R={r:.3f}  F1={f1:.3f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--coverage-only", action="store_true",
                        help="Run coverage diagnostic only (no training)")
    parser.add_argument("--features-only", action="store_true",
                        help="Run candidate collection + feature extraction "
                             "and print summary (no training)")
    parser.add_argument("--hold-out", default="gt_clip4_afternoon_dark",
                        help="Clip to hold out for final test")
    parser.add_argument("--clip", default=None,
                        help="Run on a single clip (default: all)")
    parser.add_argument("--v2-baseline", action="store_true",
                        help="Run v2 (production thresholds) on each clip and "
                             "report F1 vs GT clusters")
    parser.add_argument("--loco", action="store_true",
                        help="Leave-one-clip-out CV (honest per-clip generalization)")
    args = parser.parse_args()

    clips = [args.clip] if args.clip else CLIPS_FOR_TRAINING

    if args.coverage_only:
        print("=" * 70)
        print("  CH27 v4 — CANDIDATE-COLLECTION COVERAGE DIAGNOSTIC")
        print("=" * 70)
        all_cov = {}
        for clip in clips:
            all_cov[clip] = coverage_check(clip)
        print("\n" + "=" * 70)
        print("  SUMMARY")
        print("=" * 70)
        for clip, cov in all_cov.items():
            if cov is None: continue
            for tbl, deltas in cov.items():
                if not deltas: continue
                n = len(deltas)
                n_cov = sum(1 for d in deltas if d is not None and d <= COVERAGE_TOL_SEC)
                print(f"  {clip:30s} {tbl:5s}  {n_cov}/{n}  ({100*n_cov/n:.0f}%)")
        return

    if args.features_only:
        print("=" * 70)
        print("  CH27 v4 — FEATURE EXTRACTION (Phase 3 sanity)")
        print("=" * 70)
        for clip in clips:
            res = build_dataset_for_clip(clip)
            if res is None: continue
            X, y, enriched = res
            # Print feature summary stats for positives vs negatives
            pos_idx = y == 1
            neg_idx = y == 0
            print(f"\n  Per-feature: pos median (n={int(pos_idx.sum())}) "
                  f"vs neg median (n={int(neg_idx.sum())})")
            for i, name in enumerate(FEATURE_NAMES):
                pmed = np.median(X[pos_idx, i]) if pos_idx.any() else 0.0
                nmed = np.median(X[neg_idx, i]) if neg_idx.any() else 0.0
                ratio = (pmed - nmed) / max(0.01, abs(nmed) + abs(pmed))
                print(f"    {name:32s}  pos={pmed:7.2f}  neg={nmed:7.2f}  "
                      f"sep={ratio:+.2f}")
        return

    if args.v2_baseline:
        run_v2_baseline()
        return

    if args.loco:
        run_leave_one_clip_out()
        return

    # Default: train the classifier
    train_classifier(hold_out=args.hold_out)

    # Then retrain on ALL clips for production
    print("\n" + "=" * 70)
    print("  RETRAIN on ALL clips (production model)")
    print("=" * 70)
    train_classifier(hold_out="__none__")  # no held-out


if __name__ == "__main__":
    main()
