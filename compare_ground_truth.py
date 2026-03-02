"""
Compare blanket counter detections against manually verified ground truth.

Ground truth covers first 12 min + last ~8 min only — middle not analyzed.
Usage: python compare_ground_truth.py blanket_count_1hr.json [blanket_count_v2.json ...]
"""
import json
import sys
import statistics

# ── Ground Truth from user analysis ──
# Analyzed: first 12 min + 52:00–59:21 — middle (12–52 min) not analyzed

ACCEPTED_TIMESTAMPS = [  # MM:SS → seconds
    # 0:00 – 12:00  (1:47 was rejected, not accepted)
    5, 12, 20, 27, 33, 41, 49, 55, 61, 68, 75, 83, 89, 98,
    115, 122, 130, 137, 145, 169, 179, 186, 194, 201, 231, 269,
    290, 321, 441, 484, 506, 515, 523, 529, 536, 541, 562,
    707, 718,
    # 52:00 – 55:00  (54:03 & 54:40 occluded by man but still accepted)
    3167, 3174, 3202, 3212, 3224, 3234, 3243, 3251, 3258, 3269, 3280, 3290,
    # 55:00 – 59:21
    3304, 3333, 3343, 3362, 3370, 3377, 3388, 3398, 3408,
    3421, 3429, 3436, 3451, 3460, 3468, 3476, 3483,
    3509, 3530, 3540, 3548,
]

REJECTED_TIMESTAMPS = [
    # 0:00 – 12:00
    107, 153, 281, 299, 312, 356, 376, 392, 416, 431, 467, 491,
    623, 633, 647, 662, 671, 699,
    # 52:00 – 59:21
    3193, 3355, 3494, 3522,
]

# Major events
SCALE_MOVED = [611, 1315]  # 10:11 and 21:55
BREAK_AT = 1636  # 27:16
MAN_BLOCKS = [(473, 490), (3105, 3285), (3561, 3600)]
ANALYZED_RANGES = [(0, 720), (3100, 3561)]  # first 12 min + 51:40–59:21

MATCH_WINDOW = 5  # seconds tolerance for matching


def load_detections(json_path):
    with open(json_path) as f:
        data = json.load(f)
    events = data["videos"][0]["events"]
    return [e for e in events if e["type"] == "scale_cycle_complete"]


def match_detections(detections, ground_truth, window=MATCH_WINDOW):
    """Match detected events to ground truth timestamps."""
    det_times = [d["time_sec"] for d in detections]
    matched = []
    unmatched_gt = []
    unmatched_det = list(det_times)

    for gt_t in ground_truth:
        best = None
        best_dist = float("inf")
        for dt in unmatched_det:
            dist = abs(dt - gt_t)
            if dist < best_dist:
                best_dist = dist
                best = dt
        if best is not None and best_dist <= window:
            matched.append((gt_t, best, best_dist))
            unmatched_det.remove(best)
        else:
            unmatched_gt.append(gt_t)

    return matched, unmatched_gt, unmatched_det


def fmt(secs):
    return f"{int(secs)//60}:{int(secs)%60:02d}"


def analyze(json_path):
    """Run full analysis on one JSON file."""
    detections = load_detections(json_path)
    det_times = [d["time_sec"] for d in detections]

    print(f"\n{'=' * 70}")
    print(f"  GROUND TRUTH COMPARISON: {json_path}")
    print(f"{'=' * 70}")
    print(f"  Detected events: {len(detections)}")
    print(f"  Ground truth: {len(ACCEPTED_TIMESTAMPS)} accepted, "
          f"{len(REJECTED_TIMESTAMPS)} rejected")

    # ── Match against ACCEPTED ──
    matched_acc, missed_acc, _ = match_detections(detections, ACCEPTED_TIMESTAMPS)
    recall = len(matched_acc) / len(ACCEPTED_TIMESTAMPS) * 100

    print(f"\n  --- ACCEPTED (should detect) ---")
    print(f"  Recall: {len(matched_acc)}/{len(ACCEPTED_TIMESTAMPS)} = {recall:.0f}%")
    if missed_acc:
        print(f"  Missed: {[fmt(t) for t in missed_acc]}")
    if matched_acc:
        offsets = [d for _, _, d in matched_acc]
        print(f"  Timing offset: mean={statistics.mean(offsets):.1f}s, "
              f"median={statistics.median(offsets):.1f}s, "
              f"max={max(offsets):.1f}s")

    # ── Match against REJECTED ──
    matched_rej, _, _ = match_detections(detections, REJECTED_TIMESTAMPS)
    fp_rate = len(matched_rej) / len(REJECTED_TIMESTAMPS) * 100

    print(f"\n  --- REJECTED (should NOT detect) ---")
    print(f"  Falsely detected: {len(matched_rej)}/{len(REJECTED_TIMESTAMPS)} = {fp_rate:.0f}%")
    if matched_rej:
        print(f"  False positive times: {[fmt(gt) for gt, _, _ in matched_rej]}")
    print(f"  Correctly filtered: {len(REJECTED_TIMESTAMPS) - len(matched_rej)}"
          f"/{len(REJECTED_TIMESTAMPS)}")

    # ── Analyze region breakdown ──
    in_analyzed = [t for t in det_times
                   if any(s <= t <= e for s, e in ANALYZED_RANGES)]
    in_unanalyzed = [t for t in det_times
                     if not any(s <= t <= e for s, e in ANALYZED_RANGES)]

    all_matched = (set(d for _, d, _ in matched_acc) |
                   set(d for _, d, _ in matched_rej))
    extras = [t for t in in_analyzed if t not in all_matched]

    print(f"\n  --- REGION BREAKDOWN ---")
    print(f"  In analyzed range (0-12 + 52-60 min): {len(in_analyzed)} detections")
    print(f"    Matched to accepted GT: {len(matched_acc)}")
    print(f"    Matched to rejected GT: {len(matched_rej)}")
    print(f"    Extra (unmatched):      {len(extras)}")
    if extras:
        print(f"    Extra timestamps: {[fmt(t) for t in extras]}")
    print(f"  In unanalyzed middle (12-52 min): {len(in_unanalyzed)} detections")

    if in_unanalyzed:
        print(f"\n  --- MIDDLE SECTION BY 5-MIN WINDOWS ---")
        for m in range(12, 55, 5):
            bucket = [t for t in in_unanalyzed if m * 60 <= t < (m + 5) * 60]
            if bucket:
                rate = len(bucket) / 5
                print(f"    {m:2d}-{m+5:2d}min: {len(bucket):3d} blankets "
                      f"({rate:.1f}/min)")

    # ── Duration analysis ──
    durations = [d.get("on_duration_sec", 0) for d in detections
                 if d.get("on_duration_sec")]
    if durations:
        print(f"\n  --- ON-SCALE DURATIONS ---")
        print(f"    Range: {min(durations):.2f}s - {max(durations):.2f}s")
        print(f"    Median: {statistics.median(durations):.2f}s, "
              f"Mean: {statistics.mean(durations):.2f}s")

    # ── Summary ──
    print(f"\n  {'─' * 50}")
    print(f"  SUMMARY")
    print(f"  {'─' * 50}")
    print(f"  Recall:           {len(matched_acc)}/{len(ACCEPTED_TIMESTAMPS)} = {recall:.0f}%")
    print(f"  False pos rate:   {len(matched_rej)}/{len(REJECTED_TIMESTAMPS)} = {fp_rate:.0f}%")
    print(f"  Extra detections: {len(extras)} (in analyzed range)")
    print(f"  Middle section:   {len(in_unanalyzed)} (unverified)")
    print(f"  Total:            {len(detections)}")
    print(f"  Projected rate:   ~{len(detections)}/hr")
    print(f"{'=' * 70}\n")

    return {
        "file": json_path,
        "total": len(detections),
        "recall": len(matched_acc),
        "recall_total": len(ACCEPTED_TIMESTAMPS),
        "false_pos": len(matched_rej),
        "false_pos_total": len(REJECTED_TIMESTAMPS),
        "extras": len(extras),
        "middle": len(in_unanalyzed),
    }


def main():
    if len(sys.argv) < 2:
        print("Usage: python compare_ground_truth.py result1.json [result2.json ...]")
        sys.exit(1)

    results = []
    for path in sys.argv[1:]:
        results.append(analyze(path))

    # Side-by-side comparison if multiple files
    if len(results) > 1:
        print(f"\n{'=' * 70}")
        print(f"  COMPARISON TABLE")
        print(f"{'=' * 70}")
        headers = ["Metric"] + [r["file"].split("/")[-1] for r in results]
        rows = [
            ["Total events"] + [str(r["total"]) for r in results],
            ["Recall"] + [f"{r['recall']}/{r['recall_total']} "
                          f"({r['recall']/r['recall_total']*100:.0f}%)"
                          for r in results],
            ["False positives"] + [f"{r['false_pos']}/{r['false_pos_total']} "
                                   f"({r['false_pos']/r['false_pos_total']*100:.0f}%)"
                                   for r in results],
            ["Extras in analyzed"] + [str(r["extras"]) for r in results],
            ["Middle (unverified)"] + [str(r["middle"]) for r in results],
        ]

        # Calculate column widths
        widths = [max(len(row[i]) for row in [headers] + rows)
                  for i in range(len(headers))]

        fmt_row = "  " + " | ".join(f"{{:<{w}}}" for w in widths)
        print(fmt_row.format(*headers))
        print("  " + "-+-".join("-" * w for w in widths))
        for row in rows:
            print(fmt_row.format(*row))
        print()


if __name__ == "__main__":
    main()
