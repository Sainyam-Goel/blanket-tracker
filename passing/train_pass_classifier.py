#!/usr/bin/env python3
"""CH21 passing classifier — train accept/reject model from labeled GT clips.

Phase 1: Cluster per-frame labels into event windows.
Phase 2: Group clusters into blanket cycles (load → scale → throw).
Phase 3: Extract features from each cycle.
Phase 4: Train XGBoost binary classifier (left_throw vs right_throw).

Usage:
    python3 train_pass_classifier.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from datetime import datetime
from collections import defaultdict

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

CLIPS_DIR = HERE / "clips"
CLUSTER_GAP_FRAMES = 25  # ≤1s gap → same action window
MATCH_TOL_SEC = 2.0       # throw event matched if within ±2s of GT throw cluster

FEATURE_NAMES = [
    "load_duration_sec",
    "load_span_frames",
    "scale_duration_sec",
    "scale_span_frames",
    "has_scale",
    "load_to_scale_gap_sec",
    "scale_to_throw_gap_sec",
    "load_to_throw_gap_sec",
    "throw_span_frames",
]


# ─────────────────────────────────────────────────────────────────
#  Phase 1: GT clustering
# ─────────────────────────────────────────────────────────────────

def cluster_for_clip(labels_json_path):
    """Load labels, cluster into event windows, return list of cluster dicts.
    
    Each cluster: {type, start_frame, end_frame, n_labels, span_sec}
    """
    data = json.loads(Path(labels_json_path).read_text())
    labels = data["labels"]
    fps = data.get("fps", 25.0)
    
    by_type = defaultdict(list)
    for l in labels:
        by_type[l["type"]].append(l)
    
    clusters = []
    for typ, arr in by_type.items():
        arr = sorted(arr, key=lambda l: l["frame"])
        cur = [arr[0]]
        for l in arr[1:]:
            if l["frame"] - cur[-1]["frame"] <= CLUSTER_GAP_FRAMES:
                cur.append(l)
            else:
                clusters.append(_finalize(cur, typ, fps))
                cur = [l]
        clusters.append(_finalize(cur, typ, fps))
    
    # Sort by start frame
    clusters.sort(key=lambda c: c["start_frame"])
    return clusters, fps


def _finalize(arr, typ, fps):
    frames = [l["frame"] for l in arr]
    return {
        "type": typ,
        "start_frame": int(min(frames)),
        "end_frame": int(max(frames)),
        "start_sec": min(frames) / fps,
        "end_sec": max(frames) / fps,
        "span_sec": (max(frames) - min(frames)) / fps,
        "n_labels": len(arr),
    }


# ─────────────────────────────────────────────────────────────────
#  Phase 2: Cycle assembly
# ─────────────────────────────────────────────────────────────────

def assemble_cycles(clusters, fps):
    """Group clusters into blanket cycles.
    
    Each throw (left_throw or right_throw) defines a cycle.
    Find the closest preceding load and scale clusters within a 
    reasonable lookback window.
    
    Returns list of cycle dicts.
    """
    throws = [c for c in clusters if c["type"] in ("left_throw", "right_throw")]
    loads = [c for c in clusters if c["type"] == "load"]
    scales = [c for c in clusters if c["type"] == "scale"]
    
    cycles = []
    used_loads = set()
    used_scales = set()
    
    for i, throw in enumerate(throws):
        cycle = {
            "throw": throw,
            "label": 1 if throw["type"] == "left_throw" else 0,  # 1=accept, 0=reject
        }
        
        # Find preceding load: closest load that ends before throw starts
        best_load = None
        best_load_dist = float("inf")
        for j, ld in enumerate(loads):
            if j in used_loads: continue
            if ld["end_sec"] > throw["start_sec"]: continue
            dist = throw["start_sec"] - ld["end_sec"]
            if dist < best_load_dist and dist <= 30.0:  # 30s max lookback
                best_load_dist = dist
                best_load = (j, ld)
        
        if best_load:
            used_loads.add(best_load[0])
            cycle["load"] = best_load[1]
            cycle["load_to_throw_gap_sec"] = round(best_load_dist, 2)
        
        # Find preceding scale: closest scale between load end and throw start
        best_scale = None
        best_scale_dist = float("inf")
        load_end = cycle["load"]["end_sec"] if "load" in cycle else 0
        for j, sc in enumerate(scales):
            if j in used_scales: continue
            if sc["start_sec"] < load_end: continue
            if sc["end_sec"] > throw["start_sec"]: continue
            dist = throw["start_sec"] - sc["end_sec"]
            if dist < best_scale_dist and dist <= 20.0:
                best_scale_dist = dist
                best_scale = (j, sc)
        
        if best_scale:
            used_scales.add(best_scale[0])
            cycle["scale"] = best_scale[1]
            cycle["scale_to_throw_gap_sec"] = round(best_scale_dist, 2)
            cycle["has_scale"] = 1
            if "load" in cycle:
                cycle["load_to_scale_gap_sec"] = round(
                    cycle["scale"]["start_sec"] - cycle["load"]["end_sec"], 2)
        else:
            cycle["has_scale"] = 0
        
        cycles.append(cycle)
    
    return cycles


# ─────────────────────────────────────────────────────────────────
#  Phase 3: Feature extraction
# ─────────────────────────────────────────────────────────────────

def extract_features(cycle, fps=25.0):
    """Extract feature vector from a single blanket cycle."""
    feats = {}
    
    # Load features
    if "load" in cycle:
        feats["load_duration_sec"] = round(cycle["load"]["span_sec"], 2)
        feats["load_span_frames"] = float(cycle["load"]["end_frame"] - cycle["load"]["start_frame"])
    else:
        feats["load_duration_sec"] = 0.0
        feats["load_span_frames"] = 0.0
    
    # Scale features
    if "scale" in cycle:
        feats["scale_duration_sec"] = round(cycle["scale"]["span_sec"], 2)
        feats["scale_span_frames"] = float(cycle["scale"]["end_frame"] - cycle["scale"]["start_frame"])
    else:
        feats["scale_duration_sec"] = 0.0
        feats["scale_span_frames"] = 0.0
    
    feats["has_scale"] = float(cycle.get("has_scale", 0))
    feats["load_to_scale_gap_sec"] = float(cycle.get("load_to_scale_gap_sec", 0))
    feats["scale_to_throw_gap_sec"] = float(cycle.get("scale_to_throw_gap_sec", 0))
    feats["load_to_throw_gap_sec"] = float(cycle.get("load_to_throw_gap_sec", 0))
    feats["throw_span_frames"] = float(cycle["throw"]["end_frame"] - cycle["throw"]["start_frame"])
    
    return feats


def features_to_array(feat_dict):
    return np.array([feat_dict[name] for name in FEATURE_NAMES], dtype=float)


# ─────────────────────────────────────────────────────────────────
#  Phase 4: Training
# ─────────────────────────────────────────────────────────────────

def train_classifier(classifier="xgb"):
    """Train accept/reject classifier across all labeled clips."""
    from sklearn.model_selection import StratifiedKFold, cross_val_score, cross_val_predict
    from sklearn.metrics import precision_score, recall_score, f1_score
    from sklearn.ensemble import RandomForestClassifier
    import xgboost as xgb
    import joblib
    
    # Collect all cycles from all labeled clips
    all_cycles = {}
    clip_files = sorted(CLIPS_DIR.glob("*.labels.json"))
    
    for lf in clip_files:
        clip_name = lf.stem  # e.g. "1130_1135"
        clusters, fps = cluster_for_clip(lf)
        cycles = assemble_cycles(clusters, fps)
        if cycles:
            all_cycles[clip_name] = (cycles, fps)
            print(f"  {clip_name}: {len(cycles)} cycles "
                  f"(accept={sum(1 for c in cycles if c['label']==1)}, "
                  f"reject={sum(1 for c in cycles if c['label']==0)})")
    
    if not all_cycles:
        print("No cycles found in any clip!")
        return
    
    # Build feature matrix
    X_all = []
    y_all = []
    clip_map = []
    for clip_name, (cycles, fps) in all_cycles.items():
        for cycle in cycles:
            feats = extract_features(cycle, fps)
            X_all.append(features_to_array(feats))
            y_all.append(cycle["label"])
            clip_map.append(clip_name)
    
    X = np.array(X_all, dtype=float)
    y = np.array(y_all, dtype=int)
    
    n_pos = int(y.sum())
    n_neg = len(y) - n_pos
    
    print(f"\n{'='*60}")
    print(f"  TOTAL: {len(y)} cycles ({n_pos} accept, {n_neg} reject, "
          f"{100*n_pos/len(y):.1f}% accept)")
    print(f"{'='*60}")
    
    if n_pos < 3 or n_neg < 3:
        print("Not enough samples for either class!")
        return
    
    # Create classifier
    if classifier == "xgb":
        clf = xgb.XGBClassifier(
            n_estimators=300, max_depth=6, learning_rate=0.05,
            colsample_bytree=0.7, subsample=0.8,
            min_child_weight=3, gamma=0.1,
            scale_pos_weight=n_neg / max(1, n_pos),
            objective="binary:logistic", eval_metric="logloss",
            random_state=42, n_jobs=-1, verbosity=0,
        )
    else:
        clf = RandomForestClassifier(
            n_estimators=200, max_depth=6,
            class_weight="balanced", random_state=42, n_jobs=-1,
        )
    
    # CV
    n_folds = min(5, n_pos, n_neg)
    cv = StratifiedKFold(n_splits=max(2, n_folds), shuffle=True, random_state=42)
    cv_f1 = cross_val_score(clf, X, y, cv=cv, scoring="f1")
    cv_p = cross_val_score(clf, X, y, cv=cv, scoring="precision")
    cv_r = cross_val_score(clf, X, y, cv=cv, scoring="recall")
    
    print(f"\n  CV F1:       {cv_f1.mean():.3f} ± {cv_f1.std():.3f}")
    print(f"  CV Precision: {cv_p.mean():.3f} ± {cv_p.std():.3f}")
    print(f"  CV Recall:    {cv_r.mean():.3f} ± {cv_r.std():.3f}")
    
    # OOF threshold tuning
    oof_prob = cross_val_predict(clf, X, y, cv=cv, method="predict_proba", n_jobs=-1)[:, 1]
    best_thresh = 0.5
    best_f1 = 0.0
    for th in np.arange(0.30, 0.81, 0.02):
        yp = (oof_prob > th).astype(int)
        if yp.sum() == 0: continue
        f = f1_score(y, yp, zero_division=0)
        if f > best_f1:
            best_thresh, best_f1 = th, f
    print(f"\n  Best threshold: {best_thresh:.2f} → OOF F1={best_f1:.3f}")
    
    # Per-clip breakdown
    print(f"\n  Per-clip:")
    for clip_name in all_cycles:
        mask = [c == clip_name for c in clip_map]
        if sum(mask) == 0: continue
        idx = [i for i, m in enumerate(mask) if m]
        Xc, yc = X[idx], y[idx]
        if len(yc) < 2: continue
        
        # LOCO-like: train on all except this clip
        train_idx = [i for i in range(len(y)) if clip_map[i] != clip_name]
        if not train_idx: continue
        clf_hold = xgb.XGBClassifier(
            n_estimators=300, max_depth=6, learning_rate=0.05,
            colsample_bytree=0.7, subsample=0.8,
            min_child_weight=3, gamma=0.1,
            scale_pos_weight=(len(train_idx)-y[train_idx].sum())/max(1, y[train_idx].sum()),
            objective="binary:logistic", eval_metric="logloss",
            random_state=42, n_jobs=-1, verbosity=0,
        )
        clf_hold.fit(X[train_idx], y[train_idx])
        yp = clf_hold.predict(Xc)
        p = precision_score(yc, yp, zero_division=0)
        r = recall_score(yc, yp, zero_division=0)
        f = f1_score(yc, yp, zero_division=0)
        print(f"    hold={clip_name:20s}  P={p:.3f} R={r:.3f} F1={f:.3f}  "
              f"(n={len(yc)}, acc={yc.sum()}/{len(yc)-yc.sum()})")
    
    # Final fit
    print(f"\n  [final fit] on all {len(y)} cycles")
    clf.fit(X, y)
    
    # Feature importances
    if hasattr(clf, "feature_importances_"):
        print(f"\n  Feature importances:")
        imps = sorted(zip(FEATURE_NAMES, clf.feature_importances_), key=lambda x: -x[1])
        for name, imp in imps:
            bar = "█" * int(imp * 50)
            print(f"    {name:32s}  {imp:.3f}  {bar}")
    
    # Save
    pkl_path = HERE / "pass_classifier_v1.pkl"
    joblib.dump(clf, pkl_path, compress=3)
    print(f"\n  [saved] {pkl_path}")
    
    # Save metadata
    meta = {
        "feature_names": FEATURE_NAMES,
        "n_features": len(FEATURE_NAMES),
        "trained_at": datetime.now().isoformat(),
        "training_clips": sorted(all_cycles.keys()),
        "n_samples": len(y),
        "n_accept": n_pos,
        "n_reject": n_neg,
        "cv_f1_mean": float(cv_f1.mean()),
        "cv_f1_std": float(cv_f1.std()),
        "cv_precision_mean": float(cv_p.mean()),
        "cv_recall_mean": float(cv_r.mean()),
        "feature_importances": (
            dict(zip(FEATURE_NAMES, [float(x) for x in clf.feature_importances_]))
            if hasattr(clf, "feature_importances_") else {}),
        "model": f"XGBClassifier(n_estimators=300, max_depth=6)",
        "decision_threshold": float(best_thresh),
    }
    meta_path = HERE / "pass_classifier_metadata.json"
    meta_path.write_text(json.dumps(meta, indent=2))
    print(f"  [saved] {meta_path}")
    
    return clf, meta


if __name__ == "__main__":
    train_classifier()
