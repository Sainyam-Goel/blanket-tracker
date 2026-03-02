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


def load_events(json_path):
    with open(json_path) as f:
        data = json.load(f)
    return data["videos"][0]["events"]


def match_times(det_times, ground_truth, window=MATCH_WINDOW):
    """Match detected event times to ground truth timestamps."""
    matched = []
    unmatched_gt = []
    remaining = list(det_times)

    for gt_t in ground_truth:
        best = None
        best_dist = float("inf")
        for dt in remaining:
            dist = abs(dt - gt_t)
            if dist < best_dist:
                best_dist = dist
                best = dt
        if best is not None and best_dist <= window:
            matched.append((gt_t, best, best_dist))
            remaining.remove(best)
        else:
            unmatched_gt.append(gt_t)

    return matched, unmatched_gt, remaining


def fmt(secs):
    return f"{int(secs)//60}:{int(secs)%60:02d}"


def in_analyzed(t):
    return any(s <= t <= e for s, e in ANALYZED_RANGES)


def analyze(json_path):
    """Run full analysis on one JSON file."""
    events = load_events(json_path)

    # Separate by type
    scale_complete = [e for e in events if e["type"] == "scale_cycle_complete"]
    blanket_accepted = [e for e in events if e["type"] == "blanket_accepted"]
    blanket_rejected = [e for e in events if e["type"] == "blanket_rejected"]
    has_classification = len(blanket_accepted) > 0 or len(blanket_rejected) > 0

    print(f"\n{'=' * 70}")
    print(f"  GROUND TRUTH COMPARISON: {json_path}")
    print(f"{'=' * 70}")
    print(f"  Scale cycle events: {len(scale_complete)}")
    if has_classification:
        print(f"  Classified: {len(blanket_accepted)} accepted, "
              f"{len(blanket_rejected)} rejected, "
              f"{len(blanket_accepted) + len(blanket_rejected)} total")
    print(f"  Ground truth: {len(ACCEPTED_TIMESTAMPS)} accepted, "
          f"{len(REJECTED_TIMESTAMPS)} rejected")

    # ── ACCEPTED: match scale detections to accepted GT ──
    scale_times = [e["time_sec"] for e in scale_complete]
    matched_acc, missed_acc, _ = match_times(scale_times, ACCEPTED_TIMESTAMPS)
    recall_acc = len(matched_acc) / len(ACCEPTED_TIMESTAMPS) * 100

    print(f"\n  --- ACCEPTED GT ({len(ACCEPTED_TIMESTAMPS)}) vs Scale Detections ---")
    print(f"  Recall: {len(matched_acc)}/{len(ACCEPTED_TIMESTAMPS)} = {recall_acc:.0f}%")
    if missed_acc:
        print(f"  Missed: {[fmt(t) for t in missed_acc]}")
    if matched_acc:
        offsets = [d for _, _, d in matched_acc]
        print(f"  Timing: mean={statistics.mean(offsets):.1f}s, "
              f"median={statistics.median(offsets):.1f}s, "
              f"max={max(offsets):.1f}s")

    # ── REJECTED GT: check if scale falsely detects them ──
    matched_rej_scale, _, _ = match_times(scale_times, REJECTED_TIMESTAMPS)
    fp_rate = len(matched_rej_scale) / len(REJECTED_TIMESTAMPS) * 100

    print(f"\n  --- REJECTED GT ({len(REJECTED_TIMESTAMPS)}) vs Scale (false positives) ---")
    print(f"  Falsely weighed: {len(matched_rej_scale)}/{len(REJECTED_TIMESTAMPS)} = {fp_rate:.0f}%")
    if matched_rej_scale:
        print(f"  Times: {[fmt(gt) for gt, _, _ in matched_rej_scale]}")

    # ── CLASSIFICATION ACCURACY (if blanket_accepted/rejected events exist) ──
    rej_recall = None
    if has_classification:
        rej_times = [e["time_sec"] for e in blanket_rejected]
        matched_rej, missed_rej, extra_rej = match_times(
            rej_times, REJECTED_TIMESTAMPS
        )
        rej_recall = len(matched_rej) / len(REJECTED_TIMESTAMPS) * 100

        print(f"\n  --- REJECTED GT ({len(REJECTED_TIMESTAMPS)}) vs Rejected Detections ---")
        print(f"  Recall: {len(matched_rej)}/{len(REJECTED_TIMESTAMPS)} = {rej_recall:.0f}%")
        if missed_rej:
            print(f"  Missed: {[fmt(t) for t in missed_rej]}")
        if matched_rej:
            print(f"  Detected: {[fmt(gt) for gt, _, _ in matched_rej]}")

        # Check for misclassifications
        # Accepted GT matched by rejected detector = misclassified
        misclass_acc, _, _ = match_times(rej_times, ACCEPTED_TIMESTAMPS)
        if misclass_acc:
            print(f"\n  --- MISCLASSIFICATIONS ---")
            print(f"  Accepted GT detected as rejected: {len(misclass_acc)}")
            print(f"  Times: {[fmt(gt) for gt, _, _ in misclass_acc]}")

        # Show weight-rejected events
        weight_rej = [e for e in blanket_rejected if e.get("reject_reason") == "weight_rejected"]
        no_scale_rej = [e for e in blanket_rejected if e.get("reject_reason") == "no_scale"]
        if weight_rej:
            print(f"\n  --- WEIGHT-REJECTED ({len(weight_rej)}) ---")
            for wr in weight_rej:
                print(f"    {fmt(wr['time_sec'])} "
                      f"(scale_diff={wr.get('scale_diff', '?')}, "
                      f"slope={wr.get('texture_slope', '?')})")

        # Extra rejected in analyzed range
        rej_in_analyzed = [t for t in rej_times if in_analyzed(t)]
        rej_in_middle = [t for t in rej_times if not in_analyzed(t)]

        # Breakdown
        print(f"\n  --- REJECTED DETECTION BREAKDOWN ---")
        print(f"  In analyzed range: {len(rej_in_analyzed)} detected "
              f"(GT: {len(REJECTED_TIMESTAMPS)})")
        print(f"  In middle (unverified): {len(rej_in_middle)}")
        if weight_rej:
            print(f"  Weight-rejected: {len(weight_rej)}")
        if no_scale_rej:
            print(f"  No-scale rejected: {len(no_scale_rej)}")

    # ── Region breakdown for scale events ──
    scale_in_analyzed = [t for t in scale_times if in_analyzed(t)]
    scale_in_middle = [t for t in scale_times if not in_analyzed(t)]

    all_matched_scale = (set(d for _, d, _ in matched_acc) |
                         set(d for _, d, _ in matched_rej_scale))
    extras = [t for t in scale_in_analyzed if t not in all_matched_scale]

    print(f"\n  --- REGION BREAKDOWN (scale events) ---")
    print(f"  Analyzed range: {len(scale_in_analyzed)} scale events")
    print(f"    Matched accepted GT: {len(matched_acc)}")
    print(f"    Matched rejected GT: {len(matched_rej_scale)}")
    print(f"    Extra: {len(extras)}")
    print(f"  Middle (12-52 min): {len(scale_in_middle)} (unverified)")

    if scale_in_middle:
        print(f"\n  --- MIDDLE BY 5-MIN WINDOWS (scale events) ---")
        for m in range(12, 55, 5):
            bucket = [t for t in scale_in_middle if m * 60 <= t < (m + 5) * 60]
            if bucket:
                rate = len(bucket) / 5
                print(f"    {m:2d}-{m+5:2d}min: {len(bucket):3d} blankets "
                      f"({rate:.1f}/min)")

    # ── Duration analysis ──
    durations = [e.get("on_duration_sec", 0) for e in scale_complete
                 if e.get("on_duration_sec")]
    if durations:
        print(f"\n  --- ON-SCALE DURATIONS ---")
        print(f"    Range: {min(durations):.2f}s - {max(durations):.2f}s")
        print(f"    Median: {statistics.median(durations):.2f}s, "
              f"Mean: {statistics.mean(durations):.2f}s")

    # ── Summary ──
    print(f"\n  {'─' * 50}")
    print(f"  SUMMARY")
    print(f"  {'─' * 50}")
    print(f"  Accepted recall:  {len(matched_acc)}/{len(ACCEPTED_TIMESTAMPS)} = {recall_acc:.0f}%")
    print(f"  Scale false pos:  {len(matched_rej_scale)}/{len(REJECTED_TIMESTAMPS)} = {fp_rate:.0f}%")
    if rej_recall is not None:
        print(f"  Rejected recall:  {len(matched_rej)}/{len(REJECTED_TIMESTAMPS)} = {rej_recall:.0f}%")
    print(f"  Extra (analyzed): {len(extras)} scale events")
    print(f"  Middle section:   {len(scale_in_middle)} (unverified)")
    print(f"  Scale total:      {len(scale_complete)}")
    if has_classification:
        print(f"  Total blankets:   {len(blanket_accepted) + len(blanket_rejected)} "
              f"({len(blanket_accepted)} accepted + {len(blanket_rejected)} rejected)")
    print(f"{'=' * 70}\n")

    return {
        "file": json_path,
        "scale_total": len(scale_complete),
        "accepted_recall": len(matched_acc),
        "accepted_total": len(ACCEPTED_TIMESTAMPS),
        "rejected_recall": len(matched_rej) if has_classification else None,
        "rejected_total": len(REJECTED_TIMESTAMPS),
        "scale_false_pos": len(matched_rej_scale),
        "extras": len(extras),
        "middle": len(scale_in_middle),
        "classified_accepted": len(blanket_accepted),
        "classified_rejected": len(blanket_rejected),
    }


def main():
    if len(sys.argv) < 2:
        print("Usage: python compare_ground_truth.py result1.json [result2.json ...]")
        sys.exit(1)

    results = []
    for path in sys.argv[1:]:
        results.append(analyze(path))

    if len(results) > 1:
        print(f"\n{'=' * 70}")
        print(f"  COMPARISON TABLE")
        print(f"{'=' * 70}")
        headers = ["Metric"] + [r["file"].split("/")[-1] for r in results]
        rows = [
            ["Scale events"] + [str(r["scale_total"]) for r in results],
            ["Accepted recall"] + [
                f"{r['accepted_recall']}/{r['accepted_total']} "
                f"({r['accepted_recall']/r['accepted_total']*100:.0f}%)"
                for r in results],
            ["Scale false pos"] + [
                f"{r['scale_false_pos']}/{r['rejected_total']} "
                f"({r['scale_false_pos']/r['rejected_total']*100:.0f}%)"
                for r in results],
            ["Rejected recall"] + [
                f"{r['rejected_recall']}/{r['rejected_total']} "
                f"({r['rejected_recall']/r['rejected_total']*100:.0f}%)"
                if r['rejected_recall'] is not None else "N/A"
                for r in results],
            ["Total blankets"] + [
                f"{r['classified_accepted']}+{r['classified_rejected']}"
                f"={r['classified_accepted']+r['classified_rejected']}"
                if r['classified_rejected'] else str(r["scale_total"])
                for r in results],
        ]

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
