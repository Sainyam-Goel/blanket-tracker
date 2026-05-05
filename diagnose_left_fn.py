#!/usr/bin/env python3
"""Diagnostic audit: trace missed LEFT tosses through the pipeline for clips 8 & 9."""
import json, sys, os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from taping_counter import TapingCounter
from eval_clips import load_windows


def diagnose_clip(clip_name):
    video = f"gt_clips/{clip_name}.mp4"
    labels = f"gt_clips/{clip_name}.labels.json"

    windows = load_windows(labels)
    gt_left_toss = [w for w in windows
                    if w["type"] == "toss" and w["table"] == "left"]

    c = TapingCounter(video, version="v4")
    result = c.run()
    events = result["events"]
    suppressed = result.get("suppressed_candidates", [])

    model_left_toss = [e for e in events if e.get("table") == "left"]

    # Match GT to model (one-to-one greedy, ±3s)
    matched_gt = set()
    remaining_gt = set(range(len(gt_left_toss)))
    remaining_md = set(range(len(model_left_toss)))
    while True:
        best = None; best_d = 999
        for i in list(remaining_gt):
            for j in list(remaining_md):
                d = abs(model_left_toss[j]["time_sec"] - gt_left_toss[i]["start_t"])
                if d < best_d:
                    best_d = d; best = (i, j)
        if best and best_d <= 3.0:
            i, j = best
            matched_gt.add(i)
            remaining_gt.discard(i)
            remaining_md.discard(j)
        else:
            break

    missed = [i for i in range(len(gt_left_toss)) if i not in matched_gt]

    print(f"\n{'='*70}")
    print(f"  DIAGNOSTIC: {clip_name}")
    print(f"{'='*70}")
    print(f"  GT LEFT toss windows: {len(gt_left_toss)}")
    print(f"  Model LEFT tosses:    {len(model_left_toss)}")
    print(f"  Matched: {len(matched_gt)}  Missed: {len(missed)}")

    if not missed:
        print("  No missed LEFT tosses — diagnostic not needed.")
        return

    # For each missed GT, find the closest suppressed candidate
    stage1 = 0  # no candidate at all
    stage2 = 0  # candidate fired but prob < 0.50
    stage3 = 0  # candidate fired with 0.50-0.85, cycle_confirm_fail
    stage_unknown = 0

    for idx in missed:
        gt = gt_left_toss[idx]
        gt_t = gt["start_t"]

        # Find all suppressed LEFT candidates within ±5s of this GT
        nearby = [s for s in suppressed
                  if s.get("table") == "left"
                  and abs(s["time_sec"] - gt_t) <= 5.0]
        nearby.sort(key=lambda s: abs(s["time_sec"] - gt_t))

        if not nearby:
            stage1 += 1
            continue

        closest = nearby[0]
        prob = closest.get("v4_prob", 0)
        reason = closest.get("reason", "?")

        if reason == "cycle_confirm_fail":
            stage3 += 1
        elif reason == "classifier_reject":
            # Check if it was borderline (0.50-0.85 would have been cycle_confirm_fail
            # if it passed classifier, but classifier_reject means prob < threshold)
            # Actually, we can't distinguish stage2 from classifier below threshold
            # from low prob. Let's use prob < 0.50 as stage2 marker.
            if prob < 0.50:
                stage2 += 1
            else:
                # prob >= 0.50 but suppressed by classifier_reject - this means
                # the prob was below the effective decision threshold
                stage2 += 1  # effectively same as low toss prob
        elif reason == "cooldown":
            stage_unknown += 1
        else:
            stage_unknown += 1

    # Also check: among events that DID match, what were their probs?
    matched_probs = []
    for i in matched_gt:
        gt = gt_left_toss[i]
        # Find matching model event
        for e in model_left_toss:
            if abs(e["time_sec"] - gt["start_t"]) <= 3.0:
                matched_probs.append(e.get("v4_prob", 0))
                break

    god = sum(1 for p in matched_probs if p >= 0.85)
    border = sum(1 for p in matched_probs if 0.50 <= p < 0.85)
    low = sum(1 for p in matched_probs if p < 0.50)

    print(f"\n  MATCHED LEFT TOSSES ({len(matched_gt)}):")
    print(f"    God-tier (prob >= 0.85): {god}")
    print(f"    Borderline (0.50-0.85): {border}")
    print(f"    Low prob (< 0.50):      {low}")
    if matched_probs:
        print(f"    Mean prob: {sum(matched_probs)/len(matched_probs):.3f}")

    print(f"\n  MISSED LEFT TOSSES ({len(missed)}):")
    print(f"    Stage 1 (No candidate):       {stage1}")
    print(f"    Stage 2 (Low toss prob):      {stage2}")
    print(f"    Stage 3 (Cycle confirm fail): {stage3}")
    print(f"    Other (cooldown/unknown):     {stage_unknown}")

    # Extra detail: show all suppressed LEFT candidates' prob distribution
    left_supp = [s for s in suppressed if s.get("table") == "left"]
    if left_supp:
        probs = [s.get("v4_prob", 0) for s in left_supp]
        print(f"\n  ALL suppressed LEFT candidates: {len(left_supp)}")
        print(f"    Mean prob: {sum(probs)/len(probs):.3f}")
        print(f"    God-tier (>=0.85): {sum(1 for p in probs if p >= 0.85)}")
        print(f"    Borderline (0.50-0.85): {sum(1 for p in probs if 0.50 <= p < 0.85)}")
        print(f"    Low (< 0.50): {sum(1 for p in probs if p < 0.50)}")
        reasons = {}
        for s in left_supp:
            r = s.get("reason", "?")
            reasons[r] = reasons.get(r, 0) + 1
        print(f"    Reasons: {dict(sorted(reasons.items()))}")


if __name__ == "__main__":
    diagnose_clip("gt_clip8_afternoon")
    diagnose_clip("gt_clip9_latemorning")
