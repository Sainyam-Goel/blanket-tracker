#!/usr/bin/env python3
"""Per-clip precision/recall evaluation against per-frame GT labels."""
import json, sys, os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from taping_counter import TapingCounter


def load_windows(labels_path):
    """Extract contiguous per-table event windows from per-frame labels.
    
    Groups consecutive frames (gap <= 2 frames) of the same (table, type)
    into a single event window.
    """
    j = json.load(open(labels_path))
    labels = j.get("labels", [])
    if not labels:
        return []
    labels.sort(key=lambda l: (l["frame"], l["table"], l["type"]))

    windows = []
    for tbl in ["left", "right"]:
        for typ in ["toss", "load"]:
            subset = [l for l in labels
                      if l["table"] == tbl and l["type"] == typ]
            if not subset:
                continue
            start = subset[0]
            for i in range(1, len(subset)):
                if subset[i]["frame"] > subset[i - 1]["frame"] + 2:
                    windows.append({
                        "table": tbl, "type": typ,
                        "start_t": start["time_sec"],
                        "end_t": subset[i - 1]["time_sec"],
                    })
                    start = subset[i]
            windows.append({
                "table": tbl, "type": typ,
                "start_t": start["time_sec"],
                "end_t": subset[-1]["time_sec"],
            })
    return windows


def evaluate_clip(video_path, labels_path, clip_name):
    if not os.path.exists(labels_path):
        return None

    windows = load_windows(labels_path)
    toss_windows = [w for w in windows if w["type"] == "toss"]
    load_win = [w for w in windows if w["type"] == "load"]

    c = TapingCounter(str(video_path), version="v4")
    result = c.run()
    events = result["events"]

    # Match per-table: one-to-one greedy by shortest distance
    per_tbl = {}
    for tbl in ["left", "right"]:
        gt_toss = [w for w in toss_windows if w["table"] == tbl]
        model_toss = [e for e in events if e.get("table") == tbl]
        # Build distance matrix and do greedy one-to-one matching
        remaining_gt = set(range(len(gt_toss)))
        remaining_model = set(range(len(model_toss)))
        matched_gt = set()   # indices of matched GT windows
        matched_model = set()  # indices of matched model events
        # Greedy: repeatedly pair closest (gt, model) within 3s
        while True:
            best_dist = 999
            best_pair = None
            for i in list(remaining_gt):
                gtt = gt_toss[i]["start_t"]
                for j in list(remaining_model):
                    d = abs(model_toss[j]["time_sec"] - gtt)
                    if d < best_dist:
                        best_dist = d
                        best_pair = (i, j)
            if best_pair is not None and best_dist <= 3.0:
                i, j = best_pair
                matched_gt.add(i)
                matched_model.add(j)
                remaining_gt.discard(i)
                remaining_model.discard(j)
            else:
                break
        tp = len(matched_gt)
        fn = len(gt_toss) - tp
        fp = len(model_toss) - len(matched_model)
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
        per_tbl[tbl] = {
            "gt": len(gt_toss), "model": len(model_toss),
            "tp": tp, "fp": fp, "fn": fn,
            "precision": prec, "recall": rec, "f1": f1,
        }

    # Combined (both tables)
    combined_tp = per_tbl["left"]["tp"] + per_tbl["right"]["tp"]
    combined_fp = per_tbl["left"]["fp"] + per_tbl["right"]["fp"]
    combined_fn = per_tbl["left"]["fn"] + per_tbl["right"]["fn"]
    combined_prec = combined_tp / (combined_tp + combined_fp) if (combined_tp + combined_fp) > 0 else 0
    combined_rec = combined_tp / (combined_tp + combined_fn) if (combined_tp + combined_fn) > 0 else 0
    combined_f1 = 2 * combined_prec * combined_rec / (combined_prec + combined_rec) if (combined_prec + combined_rec) > 0 else 0

    return {
        "clip": clip_name,
        "gt_toss": len(toss_windows), "gt_load": len(load_win),
        "model_toss": len(events),
        "left": per_tbl["left"], "right": per_tbl["right"],
        "combined": {"tp": combined_tp, "fp": combined_fp, "fn": combined_fn,
                     "precision": combined_prec, "recall": combined_rec, "f1": combined_f1},
        "suppressed": len(result.get("suppressed_candidates", [])),
    }


if __name__ == "__main__":
    import glob
    clips = sorted(glob.glob("gt_clips/gt_clip*.mp4"))
    for clip_path in clips:
        name = os.path.basename(clip_path).replace(".mp4", "")
        labels_path = f"gt_clips/{name}.labels.json"
        if not os.path.exists(labels_path):
            print(f"{name:<30s} SKIP (no labels)")
            continue
        print(f"{name:<30s} ", end="", flush=True)
        r = evaluate_clip(clip_path, labels_path, name)
        if r:
            L = r["left"]
            R = r["right"]
            C = r["combined"]
            print(f"GT(L={L['gt']:>3d} R={R['gt']:>3d})→{r['gt_toss']:>4d}  "
                  f"Model(L={L['model']:>3d} R={R['model']:>3d})→{r['model_toss']:>4d}  "
                  f"L:P={L['precision']:.2f}R={L['recall']:.2f}F1={L['f1']:.2f}  "
                  f"R:P={R['precision']:.2f}R={R['recall']:.2f}F1={R['f1']:.2f}  "
                  f"TOT:F1={C['f1']:.3f}")
