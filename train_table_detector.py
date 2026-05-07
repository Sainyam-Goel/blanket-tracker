#!/usr/bin/env python3
"""Train a binary 'blanket being processed' detector from table ROIs only.

Uses frames between Load and Toss clusters as POSITIVES (blanket on table,
being taped/folded) and frames from idle clips as NEGATIVES (empty table).

This is a non-adaptive, table-only classifier — immune to baseline drift.
Integrates into v4 pipeline as a soft feature or hard gate.
"""

import json
import sys
from collections import deque
from pathlib import Path

import cv2
import numpy as np
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold, cross_val_score

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "taping"))
sys.path.insert(0, str(HERE))

from train_taping_classifier import CLIPS_FOR_TRAINING, cluster_labels
from taping_counter import LEFT_TABLE_ROI_V2, RIGHT_TABLE_ROI_V2

CLIPS_DIR = HERE / "taping" / "gt_clips"

# Table ROI features (non-adaptive — immune to baseline drift)
FEATURE_NAMES = [
    "raw_mean",       # raw grayscale mean of table ROI
    "raw_std",        # raw grayscale std-dev (texture — the idle gold signal!)
    "B_mean",         # Blue channel mean
    "G_mean",         # Green channel mean
    "R_mean",         # Red channel mean
    "B_std",
    "G_std",
    "R_std",
    "quad_LR_mean",   # lower-right quadrant (loading side)
    "quad_UR_mean",   # upper-right quadrant
    "quad_LL_mean",   # lower-left quadrant
]


def build_dataset():
    """Extract table ROI features from every frame of every labeled clip.

    Returns X (features), y (1=blanket on table, 0=empty), per per-table.
    """
    X_left, y_left = [], []
    X_right, y_right = [], []

    for clip in CLIPS_FOR_TRAINING:
        labels_file = CLIPS_DIR / f"{clip}.labels.json"
        video_file = CLIPS_DIR / f"{clip}.mp4"

        if not labels_file.exists() or not video_file.exists():
            print(f"  [skip] {clip}")
            continue

        data = json.loads(labels_file.read_text())
        labels = data["labels"]
        clusters = cluster_labels(labels)

        # Find Load→Toss windows per table
        loads_by_table = {"left": [], "right": []}
        tosses_by_table = {"left": [], "right": []}
        for c in clusters:
            if c["type"] == "load":
                loads_by_table[c["table"]].append(c["peak_frame"])
            elif c["type"] == "toss":
                tosses_by_table[c["table"]].append(c["peak_frame"])

        # Build positive windows: from Load.frame to Toss.frame
        positive_frames = {"left": set(), "right": set()}
        for tbl in ["left", "right"]:
            loads = sorted(loads_by_table[tbl])
            tosses = sorted(tosses_by_table[tbl])
            # Match each toss to the nearest preceding load
            for t in tosses:
                # Find the most recent load before this toss
                prev_loads = [l for l in loads if l < t]
                if prev_loads:
                    l = prev_loads[-1]
                    for f in range(l, t + 1):
                        positive_frames[tbl].add(f)

        rois = {
            "left":  LEFT_TABLE_ROI_V2,
            "right": RIGHT_TABLE_ROI_V2,
        }

        cap = cv2.VideoCapture(str(video_file))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 25

        # Sample every 5th frame for speed (5Hz)
        for f_idx in range(0, total_frames, 5):
            cap.set(cv2.CAP_PROP_POS_FRAMES, f_idx)
            ret, frame = cap.read()
            if not ret:
                break

            for tbl in ["left", "right"]:
                x1, y1, x2, y2 = rois[tbl]
                roi = frame[y1:y2, x1:x2]
                if roi.size == 0:
                    continue

                gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                raw_mean = float(np.mean(gray))
                raw_std = float(np.std(gray))
                B_mean = float(np.mean(roi[:, :, 0]))
                G_mean = float(np.mean(roi[:, :, 1]))
                R_mean = float(np.mean(roi[:, :, 2]))
                B_std = float(np.std(roi[:, :, 0].astype(np.float64)))
                G_std = float(np.std(roi[:, :, 1].astype(np.float64)))
                R_std = float(np.std(roi[:, :, 2].astype(np.float64)))

                # Quadrants
                mx = (x1 + x2) // 2 - x1
                my = (y1 + y2) // 2 - y1
                quad_LR = gray[my:, mx:]
                quad_UR = gray[:my, mx:]
                quad_LL = gray[my:, :mx]

                feats = [
                    raw_mean, raw_std,
                    B_mean, G_mean, R_mean,
                    B_std, G_std, R_std,
                    float(np.mean(quad_LR)) if quad_LR.size > 0 else 0.0,
                    float(np.mean(quad_UR)) if quad_UR.size > 0 else 0.0,
                    float(np.mean(quad_LL)) if quad_LL.size > 0 else 0.0,
                ]

                # Label: 1 if between Load and Toss, 0 otherwise
                label = 1 if f_idx in positive_frames[tbl] else 0

                if tbl == "left":
                    X_left.append(feats)
                    y_left.append(label)
                else:
                    X_right.append(feats)
                    y_right.append(label)

        cap.release()

        n_pos_left = sum(y_left) - sum(1 for _ in X_left if False)  # will recalc below
        print(f"  {clip}: {len(positive_frames['left'])} L+frames, {len(positive_frames['right'])} R+frames")

    return (np.array(X_left, dtype=float), np.array(y_left, dtype=int),
            np.array(X_right, dtype=float), np.array(y_right, dtype=int))


def main():
    print("=" * 70)
    print("  TABLE PROCESSING DETECTOR — training")
    print("=" * 70)

    X_L, y_L, X_R, y_R = build_dataset()

    for tbl, X, y, name in [("left", X_L, y_L, "LEFT"),
                             ("right", X_R, y_R, "RIGHT")]:
        n_pos = int(y.sum())
        n_neg = len(y) - n_pos
        if n_pos < 5 or n_neg < 5:
            print(f"\n  {name}: skip — {n_pos} pos, {n_neg} neg")
            continue

        print(f"\n{'='*50}")
        print(f"  {name} TABLE")
        print(f"  Dataset: {len(y)} samples ({n_pos} pos, {n_neg} neg, "
              f"{100*n_pos/len(y):.1f}% pos)")
        print(f"{'='*50}")

        # Train XGBoost
        cv = StratifiedKFold(n_splits=min(5, n_pos, n_neg), shuffle=True, random_state=42)
        clf = xgb.XGBClassifier(
            n_estimators=200, max_depth=5, learning_rate=0.05,
            scale_pos_weight=n_neg / max(1, n_pos),
            objective="binary:logistic", eval_metric="logloss",
            random_state=42, n_jobs=-1, verbosity=0,
        )

        f1s = cross_val_score(clf, X, y, cv=cv, scoring="f1")
        ps = cross_val_score(clf, X, y, cv=cv, scoring="precision")
        rs = cross_val_score(clf, X, y, cv=cv, scoring="recall")

        print(f"  CV F1:       {f1s.mean():.3f} ± {f1s.std():.3f}")
        print(f"  CV Precision: {ps.mean():.3f}")
        print(f"  CV Recall:    {rs.mean():.3f}")

        # Final fit
        clf.fit(X, y)

        # Feature importances
        imps = sorted(zip(FEATURE_NAMES, clf.feature_importances_), key=lambda x: -x[1])
        print(f"\n  Top features:")
        for feat_name, imp in imps[:5]:
            bar = "█" * int(imp * 40)
            print(f"    {name:20s} {imp:.4f} {bar}")

        # Save
        pkl_path = HERE / f"table_processing_detector_{name.lower()}.pkl"
        import joblib
        joblib.dump(clf, pkl_path, compress=3)
        meta = {
            "table": name.lower(),
            "feature_names": FEATURE_NAMES,
            "n_features": len(FEATURE_NAMES),
            "cv_f1_mean": float(f1s.mean()),
            "cv_f1_std": float(f1s.std()),
            "cv_precision": float(ps.mean()),
            "cv_recall": float(rs.mean()),
            "n_samples": len(y),
            "n_positives": n_pos,
            "n_negatives": n_neg,
            "feature_importances": dict(zip(
                FEATURE_NAMES, [float(x) for x in clf.feature_importances_])),
        }
        meta_path = HERE / f"table_processing_detector_{name.lower()}.json"
        meta_path.write_text(json.dumps(meta, indent=2))
        print(f"\n  [saved] {pkl_path}")
        print(f"  [saved] {meta_path}")

    print(f"\nDone. Models saved as:")
    print(f"  table_processing_detector_left.pkl")
    print(f"  table_processing_detector_right.pkl")


if __name__ == "__main__":
    main()
