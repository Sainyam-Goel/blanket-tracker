#!/usr/bin/env python3
"""Sweep decision threshold across all 4 GT clips using LOCO evaluation.

Finds the optimal threshold that maximizes per-cluster (GT windows) F1,
then projects the impact on the full-day v4 run.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from train_taping_classifier import (
    CLIPS_FOR_TRAINING, load_clip_gt, collect_candidates,
    extract_features, label_candidates, features_to_array,
    FEATURE_NAMES, MATCH_TOL_SEC,
)

def build_dataset(clip_name):
    """Build (X, y, candidates) for one clip."""
    labels, clusters, fps = load_clip_gt(clip_name)
    clip_path = HERE / "gt_clips" / f"{clip_name}.mp4"
    if not clip_path.exists():
        print(f"  [skip] {clip_path} not found")
        return None

    results = collect_candidates(clip_path, fps=fps)
    candidates = results.get("events", [])
    frame_data = results.get("frame_data", [])

    X_rows, y_rows, enriched = [], [], []
    labelled = label_candidates(candidates, clusters, fps=fps)
    for cand, y in labelled:
        feats = extract_features(cand, frame_data)
        cand_with_feats = dict(cand)
        cand_with_feats["features"] = feats
        cand_with_feats["label"] = y
        cand_with_feats["clip"] = clip_name
        cand_with_feats["table"] = cand.get("table", "")
        enriched.append(cand_with_feats)
        X_rows.append(features_to_array(feats))
        y_rows.append(y)

    X = np.array(X_rows, dtype=float)
    y = np.array(y_rows, dtype=int)
    return X, y, enriched, clusters, fps


def score_clusters(pred_times, gt_clusters, table, fps, tol_sec=MATCH_TOL_SEC):
    """Greedy match predicted times to GT clusters. Returns (P, R, F1, TP, FP, FN)."""
    gt_peaks = sorted(c["peak_frame"] / fps
                      for c in gt_clusters
                      if c["table"] == table and c["type"] == "toss")
    preds = sorted(pred_times)
    used_gt = [False] * len(gt_peaks)
    used_pred = [False] * len(preds)

    for i, p in enumerate(preds):
        best_j, best_d = -1, tol_sec
        for j, g in enumerate(gt_peaks):
            if used_gt[j]:
                continue
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


def evaluate_threshold(clf, X, y, candidates, gt_clusters, fps, threshold):
    """Evaluate one threshold: predict candidates, then match to GT clusters."""
    prob = clf.predict_proba(X)[:, 1]
    keep = prob > threshold

    # Per-candidate metrics
    y_pred = keep.astype(int)
    from sklearn.metrics import precision_score, recall_score, f1_score
    cand_p = precision_score(y, y_pred, zero_division=0)
    cand_r = recall_score(y, y_pred, zero_division=0)
    cand_f1 = f1_score(y, y_pred, zero_division=0)

    # Per-cluster metrics: keep only accepted candidates, match timestamps to GT
    accepted = [c for i, c in enumerate(candidates) if keep[i]]
    cluster_metrics = {}
    overall_tp, overall_fp, overall_fn = 0, 0, 0
    for tbl in ["left", "right"]:
        preds = [c["time_sec"] for c in accepted if c.get("table") == tbl]
        p, r, f, tp, fp, fn = score_clusters(preds, gt_clusters, tbl, fps)
        cluster_metrics[tbl] = {"P": p, "R": r, "F1": f, "TP": tp, "FP": fp, "FN": fn}
        overall_tp += tp
        overall_fp += fp
        overall_fn += fn

    op = overall_tp / max(1, overall_tp + overall_fp)
    or_ = overall_tp / max(1, overall_tp + overall_fn)
    of = 2 * op * or_ / max(1e-9, op + or_)

    return {
        "threshold": threshold,
        "candidate_P": cand_p, "candidate_R": cand_r, "candidate_F1": cand_f1,
        "cluster_P": op, "cluster_R": or_, "cluster_F1": of,
        "cluster_TP": overall_tp, "cluster_FP": overall_fp, "cluster_FN": overall_fn,
        "per_table": cluster_metrics,
        "n_accepted": int(keep.sum()),
        "n_total": len(y),
    }


def main():
    print("=" * 70)
    print("  THRESHOLD OPTIMIZATION — LOCO evaluation on 4 GT clips")
    print("=" * 70)

    # Build per-clip datasets
    print("\n[1/3] Building per-clip datasets (candidate collection)...")
    per_clip = {}
    for clip in CLIPS_FOR_TRAINING:
        t0 = time.time()
        res = build_dataset(clip)
        if res is None:
            continue
        X, y, enriched, clusters, fps = res
        per_clip[clip] = (X, y, enriched, clusters, fps)
        print(f"  {clip}: X={X.shape} pos={int(y.sum())} neg={len(y)-int(y.sum())} "
              f"({time.time()-t0:.0f}s)")

    # LOCO evaluation: for each clip, train on the other 3, test on this one
    print("\n[2/3] LOCO evaluation sweeping thresholds...")
    thresholds = np.arange(0.40, 0.95, 0.02)
    all_results = {th: [] for th in thresholds}

    for hold in CLIPS_FOR_TRAINING:
        if hold not in per_clip:
            continue
        X_te, y_te, cands_te, clusters_te, fps_te = per_clip[hold]
        train = [c for c in CLIPS_FOR_TRAINING if c != hold and c in per_clip]
        X_tr = np.concatenate([per_clip[c][0] for c in train])
        y_tr = np.concatenate([per_clip[c][1] for c in train])

        clf = RandomForestClassifier(
            n_estimators=300, max_depth=8, min_samples_leaf=3,
            class_weight="balanced", random_state=42, n_jobs=-1)
        clf.fit(X_tr, y_tr)

        for th in thresholds:
            r = evaluate_threshold(clf, X_te, y_te, cands_te, clusters_te, fps_te, th)
            r["hold_out"] = hold
            all_results[th].append(r)

        n_clips = sum(1 for th in thresholds if all_results[th])
        print(f"  hold={hold:30s} done ({n_clips} thresholds tested)")

    # Aggregate across folds
    print("\n[3/3] Results:\n")
    print(f"{'Thr':>5s}  {'Mean F1':>8s}  {'Mean P':>8s}  {'Mean R':>8s}  "
          f"{'TP':>5s}  {'FP':>5s}  {'FN':>5s}  {'Acc%':>6s}  {'F1 range':>20s}")
    print("-" * 85)

    best_thresh, best_f1 = 0.5, 0.0
    results_by_threshold = []

    for th in sorted(all_results.keys()):
        folds = all_results[th]
        if not folds:
            continue
        f1s = [f["cluster_F1"] for f in folds]
        ps = [f["cluster_P"] for f in folds]
        rs = [f["cluster_R"] for f in folds]
        tps = sum(f["cluster_TP"] for f in folds)
        fps = sum(f["cluster_FP"] for f in folds)
        fns = sum(f["cluster_FN"] for f in folds)
        acc_pct = sum(f["n_accepted"] for f in folds) / max(1, sum(f["n_total"] for f in folds))

        mean_f1 = np.mean(f1s)
        results_by_threshold.append({
            "threshold": th,
            "mean_f1": mean_f1,
            "mean_p": np.mean(ps),
            "mean_r": np.mean(rs),
            "tp": tps, "fp": fps, "fn": fns,
            "accept_pct": acc_pct,
            "f1_std": np.std(f1s),
            "f1_min": min(f1s),
            "f1_max": max(f1s),
        })

        marker = ""
        if mean_f1 > best_f1:
            best_f1 = mean_f1
            best_thresh = th
            marker = " <<< BEST"

        f1_range = f"{min(f1s):.3f}-{max(f1s):.3f}"
        print(f"  {th:.2f}  {np.mean(f1s):8.4f}  {np.mean(ps):8.4f}  {np.mean(rs):8.4f}  "
              f"{tps:5d}  {fps:5d}  {fns:5d}  {acc_pct*100:5.1f}%  {f1_range:>20s}{marker}")

    print(f"\n{'='*70}")
    print(f"  OPTIMAL THRESHOLD: {best_thresh:.2f} → mean LOCO F1 = {best_f1:.4f}")
    print(f"{'='*70}")

    # Project onto full-day counts
    print("\n" + "=" * 70)
    print("  FULL-DAY PROJECTION")
    print("=" * 70)
    try:
        fd = json.loads((HERE / "taping_fullday.json").read_text())
        fd_probs = [e["v4_prob"] for e in fd["events"]]
        print(f"  Current (th=0.50): {len(fd_probs)} events")
        for th in [0.55, 0.60, 0.65, 0.70, best_thresh, 0.78, 0.80, 0.85]:
            surviving = sum(1 for p in fd_probs if p > th)
            marker = " <<< OPTIMAL" if abs(th - best_thresh) < 0.01 else ""
            print(f"  th={th:.2f}: {surviving:5d} events ({surviving/max(1,len(fd_probs))*100:.0f}%){marker}")
    except Exception as e:
        print(f"  [warn] Could not project: {e}")

    # Save results
    out = {
        "scanned_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "optimal_threshold": float(best_thresh),
        "sweep": results_by_threshold,
    }
    out_path = HERE / "threshold_optimization.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\n[saved] {out_path}")

    return best_thresh


if __name__ == "__main__":
    main()
