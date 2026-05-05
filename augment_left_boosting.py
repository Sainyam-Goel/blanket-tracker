#!/usr/bin/env python3
"""
Augment LEFT table training data for clips 8 & 9.

Task 1: Hard Positive Boosting — duplicate LEFT GT-matched candidates
        that the current model scores < 0.50 (13 misses).

Task 2: Arm-Swing Hard Negative Mining — for each GT LEFT toss,
        find candidates in the 2.0-5.0s pre-toss window and
        label them as Class=0 hard negatives (duplicate them).

Usage:
    python3 augment_left_boosting.py           # detection only
    python3 augment_left_boosting.py --apply    # retrain with augmentations
"""
import json, sys, os, joblib
import numpy as np
from pathlib import Path
from collections import deque

sys.path.insert(0, str(Path(__file__).parent))
from train_taping_classifier import (
    build_dataset_for_clip, FEATURE_NAMES,
    CLIPS_FOR_TRAINING, _make_clf,
)
from eval_clips import load_windows


def find_hard_positives(clip_name, model, per_clip_data):
    """Find LEFT candidates with label=1 but model prob < 0.50."""
    if clip_name not in per_clip_data:
        return []
    X, y, enriched = per_clip_data[clip_name]

    hard_positives = []
    for i, (cand_row, label) in enumerate(zip(enriched, y)):
        if cand_row.get("table") != "left":
            continue
        if label != 1:
            continue
        # Score with current model
        feat_vec = X[i].reshape(1, -1)
        prob = float(model.predict_proba(feat_vec)[0, 1])
        if prob < 0.50:
            hard_positives.append({
                "clip": clip_name,
                "idx": i,
                "frame": int(cand_row.get("frame", 0)),
                "time_sec": cand_row.get("time_sec", 0),
                "prob": round(prob, 3),
            })
    return hard_positives


def find_arm_swing_negatives(clip_name, per_clip_data):
    """For each GT LEFT toss, mark candidates in 2-5s pre-toss window as negatives."""
    if clip_name not in per_clip_data:
        return []
    X, y, enriched = per_clip_data[clip_name]

    # Load GT windows
    labels_path = f"gt_clips/{clip_name}.labels.json"
    windows = load_windows(labels_path)
    gt_left_toss = [w for w in windows
                    if w["type"] == "toss" and w["table"] == "left"]

    arm_swing_indices = set()
    for i, cand_row in enumerate(enriched):
        if cand_row.get("table") != "left":
            continue
        ct = cand_row.get("time_sec", 0)
        # Check if this candidate is in the 2-5s pre-toss window of any GT
        for gt in gt_left_toss:
            delta = gt["start_t"] - ct
            if 2.0 <= delta <= 5.0:
                arm_swing_indices.add(i)
                break

    result = []
    for idx in sorted(arm_swing_indices):
        result.append({
            "clip": clip_name,
            "idx": idx,
            "frame": int(enriched[idx].get("frame", 0)),
            "time_sec": enriched[idx].get("time_sec", 0),
        })
    return result


def main(dry_run=True):
    print("Loading current LEFT toss model...")
    base = Path(__file__).parent
    left_model_path = base / "taping_pulse_classifier_toss_v4_left.pkl"
    model = joblib.load(left_model_path)

    print("Building datasets for all training clips...")
    per_clip = {}
    for clip in CLIPS_FOR_TRAINING:
        sys.stdout.write(f"  {clip}... "); sys.stdout.flush()
        res = build_dataset_for_clip(clip, verbose=False)
        if res is None:
            print("SKIP")
            continue
        X, y, enriched = res
        per_clip[clip] = (X, y, enriched)
        n_left = sum(1 for e in enriched if e.get("table") == "left")
        n_right = sum(1 for e in enriched if e.get("table") == "right")
        n_pos = sum(1 for v in y if v == 1)
        print(f"{len(y)} rows (L={n_left}, R={n_right}, pos={n_pos})")

    # Task 1: Find hard positives
    print("\n=== TASK 1: Hard Positives (LEFT label=1 but prob<0.50) ===")
    all_hard_pos = []
    for clip in ["gt_clip8_afternoon", "gt_clip9_latemorning"]:
        hp = find_hard_positives(clip, model, per_clip)
        all_hard_pos.extend(hp)
        print(f"  {clip}: {len(hp)} hard positives")
        for h in hp:
            print(f"    frame={h['frame']} t={h['time_sec']:.1f}s prob={h['prob']:.3f}")

    # Task 2: Find arm-swing negatives
    print(f"\n=== TASK 2: Arm-Swing Negatives (LEFT, 2-5s before GT toss) ===")
    all_arm_swing = []
    for clip in ["gt_clip8_afternoon", "gt_clip9_latemorning"]:
        as_neg = find_arm_swing_negatives(clip, per_clip)
        all_arm_swing.extend(as_neg)
        print(f"  {clip}: {len(as_neg)} arm-swing candidates")
        for a in as_neg[:5]:
            print(f"    frame={a['frame']} t={a['time_sec']:.1f}s")

    print(f"\n=== SUMMARY ===")
    print(f"  Hard positives to duplicate: {len(all_hard_pos)}")
    print(f"  Arm-swing negatives to duplicate: {len(all_arm_swing)}")

    if not dry_run:
        print("\n=== RETRAINING WITH AUGMENTATIONS ===")
        retrain_with_augmentation(per_clip, all_hard_pos, all_arm_swing)


def retrain_with_augmentation(per_clip, hard_pos, arm_swing):
    """Retrain per-table classifiers with augmented data."""
    from train_taping_classifier import _make_clf

    # Build index lookup: (clip, idx_in_clip) -> row
    # For augmentation we need to duplicate specific rows per clip

    all_X = {"left": [], "right": []}
    all_y = {"left": [], "right": []}
    clip_row_map = {}  # clip -> list of (global_idx, tbl)

    for clip in CLIPS_FOR_TRAINING:
        if clip not in per_clip:
            continue
        X, y, enriched = per_clip[clip]
        clip_row_map[clip] = []
        for i, (cand, label) in enumerate(zip(enriched, y)):
            tbl = cand.get("table", "")
            if tbl in all_X:
                all_X[tbl].append(X[i])
                all_y[tbl].append(label)
                clip_row_map[clip].append((len(all_X[tbl]) - 1, tbl))

    # Apply Task 1: Duplicate hard positives (2x)
    hp_by_clip = {}
    for hp in hard_pos:
        hp_by_clip.setdefault(hp["clip"], []).append(hp["idx"])

    for clip, idxs in hp_by_clip.items():
        X, y, enriched = per_clip[clip]
        for idx in idxs:
            cand = enriched[idx]
            tbl = cand.get("table", "")
            if tbl in all_X:
                all_X[tbl].append(X[idx])
                all_y[tbl].append(int(y[idx]))  # duplicate as class=1
        print(f"  Duplicated {len(idxs)} hard positives from {clip}")

    # Apply Task 2: Duplicate arm-swing as hard negatives (3x)
    as_by_clip = {}
    for as_ in arm_swing:
        as_by_clip.setdefault(as_["clip"], []).append(as_["idx"])

    for clip, idxs in as_by_clip.items():
        X, y, enriched = per_clip[clip]
        for _ in range(3):  # 3x duplicate
            for idx in idxs:
                cand = enriched[idx]
                tbl = cand.get("table", "")
                if tbl in all_X:
                    all_X[tbl].append(X[idx])
                    all_y[tbl].append(0)  # force as negative
        print(f"  Duplicated {len(idxs)} arm-swing negatives from {clip} (3x)")

    # Retrain per-table
    base = Path(__file__).parent
    for tbl in ["left", "right"]:
        X_tbl = np.array(all_X[tbl], dtype=float)
        y_tbl = np.array(all_y[tbl], dtype=int)
        n_pos = int(np.sum(y_tbl == 1))
        n_neg = int(np.sum(y_tbl == 0))
        print(f"\n  {tbl.upper()}: X={X_tbl.shape}, y={y_tbl.shape}, "
              f"pos={n_pos}, neg={n_neg}")

        clf = _make_clf(X_tbl, y_tbl)
        clf.fit(X_tbl, y_tbl)

        pkl_path = base / f"taping_pulse_classifier_toss_v5_{tbl}.pkl"
        joblib.dump(clf, pkl_path)
        print(f"  Saved to {pkl_path}")

    print("\nDone. New models saved as v5.")


if __name__ == "__main__":
    dry_run = "--apply" not in sys.argv
    if not dry_run:
        print("WARNING: --apply mode will RETRAIN models. Ensure you want this.")
    main(dry_run=dry_run)
