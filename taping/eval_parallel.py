#!/usr/bin/env python3
"""Parallel per-clip evaluation with fast_mode."""
import json, os, sys, time
from pathlib import Path
from multiprocessing import Pool, cpu_count

BASE = Path(__file__).parent
sys.path.insert(0, str(BASE))

def merge_windows(windows, min_gap=4.0):
    by_key = {}
    for w in windows:
        by_key.setdefault((w['type'], w['table']), []).append(w)
    result = []
    for k, ws in by_key.items():
        ws = sorted(ws, key=lambda w: w['start_t'])
        cur = None
        for w in ws:
            if cur is None or w['start_t'] - cur['end_t'] > min_gap:
                if cur is not None: result.append(cur)
                cur = dict(w)
            else:
                cur['end_t'] = max(cur['end_t'], w['end_t'])
        if cur is not None: result.append(cur)
    return result

def eval_one(args):
    clip_name, version = args
    from taping_counter import TapingCounter
    from eval_clips import load_windows

    video = str(BASE / "gt_clips" / f"{clip_name}.mp4")
    labels = str(BASE / "gt_clips" / f"{clip_name}.labels.json")
    if not os.path.exists(video) or not os.path.exists(labels):
        return None

    windows = load_windows(labels)
    merged = merge_windows(windows)

    t0 = time.time()
    c = TapingCounter(video, version=version, fast_mode=True)
    result = c.run()
    elapsed = time.time() - t0
    events = result["events"]

    per_tbl = {}
    for tbl in ['left', 'right']:
        gt = [w for w in merged if w['table'] == tbl and w['type'] == 'toss']
        model = [e for e in events if e.get('table') == tbl]
        if not gt and not model: continue
        rg = set(range(len(gt))); rm = set(range(len(model)))
        mg = set(); mm = set()
        while True:
            bi = 0; bj = 0; bd = 999
            for i in list(rg):
                for j in list(rm):
                    d = abs(model[j]['time_sec'] - gt[i]['start_t'])
                    if d < bd: bd = d; bi = i; bj = j
            if bd <= 3.0: mg.add(bi); mm.add(bj); rg.discard(bi); rm.discard(bj)
            else: break
        tp = len(mg); fp = len(model) - len(mm); fn = len(gt) - tp
        prec = tp/(tp+fp) if (tp+fp) else 0
        rec = tp/(tp+fn) if (tp+fn) else 0
        f1 = 2*prec*rec/(prec+rec) if (prec+rec) else 0
        per_tbl[tbl] = {
            "gt": len(gt), "model": len(model),
            "tp": tp, "fp": fp, "fn": fn, "f1": f1,
        }

    return {
        "clip": clip_name,
        "elapsed": elapsed,
        "left": per_tbl.get("left"),
        "right": per_tbl.get("right"),
    }

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", default="v11")
    args = parser.parse_args()
    version = args.version

    labeled = sorted([f.replace('.labels.json', '') for f in os.listdir(str(BASE / "gt_clips")) if f.endswith('.labels.json')])
    existing = [c for c in labeled if os.path.exists(str(BASE / "gt_clips" / f"{c}.mp4"))]

    print(f"Evaluating {len(existing)} clips with version={version} in parallel...")
    workers = min(cpu_count(), len(existing), 6)  # cap at 6 to avoid memory thrash

    with Pool(workers) as pool:
        results = pool.map(eval_one, [(c, version) for c in existing])

    results = [r for r in results if r is not None]
    results.sort(key=lambda r: r["clip"])

    all_tp = 0; all_fp = 0; all_fn = 0
    print(f"\n{'CLIP':<28s} {'TBL':>5s} {'GT':>4s} {'M':>4s} {'TP':>4s} {'FP':>4s} {'FN':>4s} {'F1':>6s}  {'Time':>5s}")
    print('-'*80)

    for r in results:
        for tbl in ['left', 'right']:
            d = r[tbl]
            if d is None: continue
            all_tp += d['tp']; all_fp += d['fp']; all_fn += d['fn']
            print(f"{r['clip']:<28s} {tbl:>5s} {d['gt']:>4d} {d['model']:>4d} {d['tp']:>4d} {d['fp']:>4d} {d['fn']:>4d} {d['f1']:>5.3f}  {r['elapsed']:>4.0f}s")

    P = all_tp/(all_tp+all_fp) if (all_tp+all_fp) else 0
    R = all_tp/(all_tp+all_fn) if (all_tp+all_fn) else 0
    F1 = 2*P*R/(P+R) if (P+R) else 0
    print(f"{'OVERALL':<28s} {'':>5s} {'':>4s} {'':>4s} {all_tp:>4d} {all_fp:>4d} {all_fn:>4d} {F1:>5.3f}")
    print(f"P={P:.3f} R={R:.3f}")
    print(f"Total time: {sum(r['elapsed'] for r in results):.0f}s wall-clock, ~{sum(r['elapsed'] for r in results)/workers:.0f}s parallel")

if __name__ == "__main__":
    main()
