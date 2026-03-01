# Blanket Tracker

A computer vision system for counting blankets on a factory production floor using OpenCV.

## Overview

Blanket Tracker uses OpenCV's MOG2 background subtraction algorithm to detect and count blanket processing events across defined zones in factory camera feeds. It supports both live RTSP streams from an NVR and pre-recorded video files, and outputs a JSON log of all detected events.

The companion dashboard (`blanket_tracker_dashboard.html`) visualizes the results with per-zone counts, motion timelines, and a full event log.

## How It Works

1. Each camera feed is divided into named zones (e.g. Cutting Table, Weighing Station)
2. Each frame is processed through MOG2 background subtraction + morphological filtering to isolate motion
3. When motion in a zone crosses a configurable activity threshold, a counting event is recorded
4. A cooldown period prevents duplicate counts from sustained motion
5. Results are saved to a JSON log file at the end of the session

## Files

| File | Description |
|------|-------------|
| `blanket_tracker.py` | Main tracking script — runs on video files or live RTSP streams |
| `blanket_tracker_dashboard.html` | Interactive dashboard to visualize analysis results |

## Requirements

```bash
pip install opencv-python numpy
```

Optional (for future YOLOv8 upgrade):
```bash
pip install ultralytics supervision
```

## Usage

**Analyze a video file:**
```bash
python blanket_tracker.py --source path/to/video.mp4 --camera ch21
```

**Connect to a live RTSP stream:**
```bash
python blanket_tracker.py --source rtsp://user:pass@192.168.1.100/ch21 --camera ch21 --live
```

**Run both cameras simultaneously:**
```bash
python blanket_tracker.py --source rtsp://user:pass@NVR_IP/ch19 --camera ch19 --live &
python blanket_tracker.py --source rtsp://user:pass@NVR_IP/ch21 --camera ch21 --live &
```

## Camera Configuration

Two cameras are pre-configured with zone layouts:

- **CH19** — Cutting Floor: Cutting Table, Left Sorting Worker, Right Pile Area
- **CH21** — Finishing Station: Weighing Station, Folding Table, Sewing Machines

Zone rectangles (pixel coordinates) can be tuned in `ZONE_CONFIG` inside `blanket_tracker.py` to match your specific camera angles.

## Output

Each session produces a `counts.json` file:

```json
{
  "camera": "ch21",
  "total": 7,
  "blankets_counted": {
    "Weighing Station": 3,
    "Folding Table": 1,
    "Sewing Machines": 3
  },
  "events": [ ... ]
}
```

Open `blanket_tracker_dashboard.html` in a browser to visualize the results.

## Accuracy Notes

- Background subtraction works best for indoor factory scenes with stable lighting
- The **Weighing Station** on CH21 is the recommended chokepoint — every finished blanket passes through it
- For higher accuracy, consider training a YOLOv8 model on labeled frames of folded blankets using [Roboflow](https://roboflow.com)
