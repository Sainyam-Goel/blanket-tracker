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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--coverage-only", action="store_true",
                        help="Run coverage diagnostic only (no training)")
    parser.add_argument("--clip", default=None,
                        help="Run on a single clip (default: all)")
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

    print("Phase 4-5 not yet implemented. Run with --coverage-only first.")


if __name__ == "__main__":
    main()
