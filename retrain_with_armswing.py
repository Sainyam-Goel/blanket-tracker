#!/usr/bin/env python3
"""
Retrain TOSS model with arm-swing hard negative mining.

For clips 8 & 9, finds LEFT candidates in the 2.0-5.0s window
before each GT toss and adds them as Class=0 negatives (1x weight).
"""
import json, sys, os, joblib, time
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from train_taping_classifier import (
    build_dataset_for_clip, FEATURE_NAMES,
    CLIPS_FOR_TRAINING, _make_clf,
)
from eval_clips import load_windows

HERE = Path(__file__).parent


def main():
    print("Building training datasets for all clips...")
    per_clip = {}
    for clip in CLIPS_FOR_TRAINING:
        sys.stdout.write(f"  {clip}... "); sys.stdout.flush()
        t0 = time.time()
        res = build_dataset_for_clip(clip, verbose=False)
        if res is None:
            print("SKIP"); continue
        X, y, enriched = res
        per_clip[clip] = (X, y, enriched)
        nL = sum(1 for e in enriched if e.get("table") == "left")
        nR = sum(1 for e in enriched if e.get("table") == "right")
        n1 = int(np.sum(y == 1))
        n0 = int(np.sum(y == 0))
        print(f"{len(y)} rows (L={nL} R={nR} pos={n1} neg={n0}) {time.time()-t0:.0f}s")

    # Find arm-swing negatives for clips 8 & 9
    print("\nFinding arm-swing negatives (2-5s before GT LEFT toss)...")
    arm_swing_rows = {"left": [], "right": []}

    for clip in ["gt_clip8_afternoon", "gt_clip9_latemorning"]:
        if clip not in per_clip:
            continue
        X, y, enriched = per_clip[clip]

        labels_path = HERE / "gt_clips" / f"{clip}.labels.json"
        windows = load_windows(str(labels_path))
        gt_left_toss = [w for w in windows
                        if w["type"] == "toss" and w["table"] == "left"]

        count = 0
        for i, cand in enumerate(enriched):
            if cand.get("table") != "left":
                continue
            ct = cand.get("time_sec", 0)
            for gt in gt_left_toss:
                delta = gt["start_t"] - ct
                if 2.0 <= delta <= 5.0:
                    arm_swing_rows["left"].append(X[i])
                    count += 1
                    break
        print(f"  {clip}: {count} arm-swing negatives")

    # Assemble training data per table
    all_X = {"left": [], "right": []}
    all_y = {"left": [], "right": []}

    for clip in CLIPS_FOR_TRAINING:
        if clip not in per_clip:
            continue
        X, y, enriched = per_clip[clip]
        for i, (cand, label) in enumerate(zip(enriched, y)):
            tbl = cand.get("table", "")
            if tbl in all_X:
                all_X[tbl].append(X[i])
                all_y[tbl].append(label)

    # Add arm-swing negatives (1x weight)
    for tbl in ["left"]:
        for row in arm_swing_rows[tbl]:
            all_X[tbl].append(row)
            all_y[tbl].append(0)

    # Retrain per-table
    for tbl in ["left", "right"]:
        X_tbl = np.array(all_X[tbl], dtype=float)
        y_tbl = np.array(all_y[tbl], dtype=int)
        n_pos = int(np.sum(y_tbl == 1))
        n_neg = int(np.sum(y_tbl == 0))

        arm_count = len(arm_swing_rows[tbl])
        print(f"\n  {tbl.upper()}: X={X_tbl.shape}, pos={n_pos}, neg={n_neg} "
              f"(+{arm_count} arm-swing)")

        clf = _make_clf(n_pos, n_neg)
        clf.fit(X_tbl, y_tbl)

        pkl_path = HERE / f"taping_pulse_classifier_toss_v5_{tbl}.pkl"
        joblib.dump(clf, pkl_path)
        print(f"  Saved → {pkl_path}")

    # Quick CV check
    print("\n=== CV F1 (5-fold) ===")
    for tbl in ["left", "right"]:
        from sklearn.model_selection import cross_val_predict
        from sklearn.metrics import f1_score
        X_tbl = np.array(all_X[tbl], dtype=float)
        y_tbl = np.array(all_y[tbl], dtype=int)
        n_pos = int(np.sum(y_tbl == 1))
        n_neg = int(np.sum(y_tbl == 0))
        clf = _make_clf(n_pos, n_neg)
        y_pred = cross_val_predict(clf, X_tbl, y_tbl, cv=5, n_jobs=-1)
        f1 = f1_score(y_tbl, y_pred)
        print(f"  {tbl.upper()}: CV F1 = {f1:.3f}")

    print("\nDone. v5 models ready.")


if __name__ == "__main__":
    main()
