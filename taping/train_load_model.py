#!/usr/bin/env python3
"""LOAD detector — standalone training (separate from toss model).

Trains XGBoost classifiers to detect blanket LOADING events
on each table. Uses same features and architecture as the toss model
but labels candidates against LOAD clusters instead of TOSS.

Output:
  taping_pulse_classifier_load_v4_left.pkl
  taping_pulse_classifier_load_v4_right.pkl
  classifier_metadata_load_left.json
  classifier_metadata_load_right.json
"""

import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold, cross_val_score

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from train_taping_classifier import (
    CLIPS_FOR_TRAINING, load_clip_gt, collect_candidates,
    extract_features, features_to_array, FEATURE_NAMES,
    MATCH_TOL_SEC, FEATURE_PRE, FEATURE_POST, AIR_THRESH_FOR_SHAPE,
)


def label_load_candidates(candidates, gt_clusters, tol_sec=MATCH_TOL_SEC, fps=25.0):
    """Label candidates for LOAD — positive if near any LOAD cluster frame."""
    load_frames_by_table = {"left": set(), "right": set()}
    for c in gt_clusters:
        if c["type"] == "load":
            for f in c.get("all_frames", [c["peak_frame"]]):
                load_frames_by_table[c["table"]].add(f)

    tol_frames = int(tol_sec * fps)
    out = []
    for cand in candidates:
        tbl = cand["table"]
        peak_f = int(cand["frame"])
        nearest = min((abs(peak_f - f) for f in load_frames_by_table[tbl]),
                       default=float("inf"))
        out.append((cand, 1 if nearest <= tol_frames else 0))
    return out


def build_dataset(clip_name):
    """Build (X, y, candidates) for one clip using LOAD labels."""
    labels, clusters, fps = load_clip_gt(clip_name)
    clip_path = HERE / "gt_clips" / f"{clip_name}.mp4"
    if not clip_path.exists():
        return None

    results = collect_candidates(clip_path, fps=fps)
    candidates = results.get("events", [])
    frame_data = results.get("frame_data", [])

    X_rows, y_rows, enriched = [], [], []
    labelled = label_load_candidates(candidates, clusters, fps=fps)
    for cand, y in labelled:
        feats = extract_features(cand, frame_data)
        cand["features"] = feats
        cand["label"] = y
        cand["clip"] = clip_name
        enriched.append(cand)
        X_rows.append(features_to_array(feats))
        y_rows.append(y)

    return np.array(X_rows, dtype=float), np.array(y_rows, dtype=int), enriched


def main():
    print("=" * 70)
    print("  CH27 v4 — LOAD DETECTOR training (standalone)")
    print("=" * 70)

    # Build per-clip datasets
    all_X_L, all_y_L = [], []
    all_X_R, all_y_R = [], []

    for clip in CLIPS_FOR_TRAINING:
        t0 = time.time()
        res = build_dataset(clip)
        if res is None:
            continue
        X, y, enriched = res
        for i, cand in enumerate(enriched):
            tbl = cand.get("table", "")
            if tbl == "left":
                all_X_L.append(X[i])
                all_y_L.append(y[i])
            elif tbl == "right":
                all_X_R.append(X[i])
                all_y_R.append(y[i])

        n_pos = int(y.sum())
        print(f"  {clip}: n={len(y)} pos={n_pos} ({time.time()-t0:.0f}s)")

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
        print(f"  LOAD {name} TABLE")
        print(f"  Dataset: {len(y_tbl)} candidates ({n_pos} pos, {n_neg} neg, "
              f"{100*n_pos/len(y_tbl):.1f}% pos)")

        cv = StratifiedKFold(n_splits=min(5, n_pos, n_neg), shuffle=True, random_state=42)
        clf = xgb.XGBClassifier(
            n_estimators=400, max_depth=8, learning_rate=0.03,
            colsample_bytree=0.6, subsample=0.8,
            min_child_weight=5, gamma=0.1,
            reg_lambda=1.0, reg_alpha=0.1,
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

        # Feature importance
        imps = sorted(zip(FEATURE_NAMES, clf.feature_importances_), key=lambda x: -x[1])
        print(f"\n  Top 5 features:")
        for feat_name, imp in imps[:5]:
            bar = "█" * int(imp * 40)
            print(f"    {feat_name:32s} {imp:.4f} {bar}")

        # Save
        import joblib
        tbl_key = "left" if name == "LEFT" else "right"
        pkl_path = HERE / f"taping_pulse_classifier_load_v4_{tbl_key}.pkl"
        joblib.dump(clf, pkl_path, compress=3)

        meta = {
            "table": tbl_key,
            "model_type": "load",
            "feature_names": FEATURE_NAMES,
            "n_features": len(FEATURE_NAMES),
            "trained_at": datetime.now().isoformat(),
            "training_clips": CLIPS_FOR_TRAINING,
            "cv_f1_mean": float(f1s.mean()),
            "cv_f1_std": float(f1s.std()),
            "cv_precision": float(ps.mean()),
            "cv_recall": float(rs.mean()),
            "n_samples": len(y_tbl),
            "n_positives": n_pos,
            "n_negatives": n_neg,
            "feature_importances": dict(zip(
                FEATURE_NAMES, [float(x) for x in clf.feature_importances_])),
            "model": "XGBClassifier(n_estimators=400, max_depth=8)",
        }
        meta_path = HERE / f"classifier_metadata_load_{tbl_key}.json"
        meta_path.write_text(json.dumps(meta, indent=2))
        print(f"\n  [saved] {pkl_path.name}")

    print(f"\nDone. Load models saved separately from toss models.")


if __name__ == "__main__":
    main()
