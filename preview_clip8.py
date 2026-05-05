#!/usr/bin/env python3
"""Annotate clip8 with GT labels for visual inspection."""
import json, cv2, numpy as np, sys
from pathlib import Path

CLIP = "gt_clip8_afternoon"
video_path = f"gt_clips/{CLIP}.mp4"
labels_path = f"gt_clips/{CLIP}.labels.json"

labels = json.load(open(labels_path))["labels"]
labels.sort(key=lambda l: l["frame"])

# Build per-frame label lookup
frame_labels = {}
for l in labels:
    f = l["frame"]
    if f not in frame_labels:
        frame_labels[f] = []
    frame_labels[f].append(l)

# Get unique event windows
from eval_clips import load_windows
windows = load_windows(labels_path)
toss_windows = [w for w in windows if w["type"] == "toss"]

print(f"Clip: {CLIP}")
print(f"Total labels: {len(labels)} per-frame entries")
print(f"Toss windows: {len(toss_windows)}")
print(f"  LEFT:  {sum(1 for w in toss_windows if w['table']=='left')} ")
print(f"  RIGHT: {sum(1 for w in toss_windows if w['table']=='right')}")

# Save annotated frames at each toss peak
cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS) or 25

for i, tw in enumerate(toss_windows):
    mid_t = (tw["start_t"] + tw["end_t"]) / 2
    mid_f = int(mid_t * fps)
    cap.set(cv2.CAP_PROP_POS_FRAMES, mid_f)
    ret, frame = cap.read()
    if not ret:
        continue

    tbl = tw["table"]
    color = (0, 255, 0) if tbl == "left" else (0, 0, 255)
    label_text = f"{tbl.upper()} TOSS #{i+1}  t={tw['start_t']:.1f}s"

    # Draw table ROI
    if tbl == "left":
        x1, y1, x2, y2 = 188, 684, 687, 1068
    else:
        x1, y1, x2, y2 = 1243, 816, 1734, 1073
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    cv2.putText(frame, label_text, (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    out_path = f"/tmp/clip8_toss_{i:03d}_{tbl}.jpg"
    cv2.imwrite(out_path, frame)

cap.release()
print(f"\nSaved {len(toss_windows)} annotated frames to /tmp/clip8_toss_*.jpg")
print(f"Open with: open /tmp/clip8_toss_000_left.jpg")

# Also print time distribution
print(f"\nTime distribution:")
for tw in toss_windows:
    tbl = tw["table"]
    bar = "L" if tbl == "left" else "R"
    print(f"  {tw['start_t']:6.1f}s  {bar}  dur={tw['end_t']-tw['start_t']:.1f}s")
