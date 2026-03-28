# Blanket Production Tracker

Computer vision system for counting blankets on a factory production floor using OpenCV. Tracks both **cutting** (CH19) and **passing/weighing** (CH21) stations from NVR security camera feeds.

**Live dashboard**: [sainyam-goel.github.io/blanket-tracker](https://sainyam-goel.github.io/blanket-tracker/)

## Overview

Built for a blanket factory in Panipat, Haryana. The system processes 1920x1080 @ 25fps NVR recordings and produces a self-contained HTML dashboard with production analytics.

### Two-Camera Pipeline

| Camera | Station | What It Counts | Method |
|--------|---------|----------------|--------|
| **CH19** | Cutting Table | Blanket cuts (pieces sliding off table) | Multi-scale brightness derivative detection |
| **CH21** | Passing Station | Accepted + rejected blankets | Scale reference-frame comparison + table texture |

## Results

### Full Day (7.6 hrs, 27 Feb 2026, 11:00–18:35)

| Metric | Value |
|--------|-------|
| **CH19 cuts detected** | 1,726 |
| **CH21 accepted blankets** | 1,488 |
| **CH21 rejected blankets** | 431 |
| **CH21 reject rate** | 22.5% |
| **CH19 active cutting rate** | 4.6 cuts/min |
| **CH21 finish rate** | 253 blankets/hr |
| **Processing speed** | ~10x real-time per camera |

### Accuracy (validated on 1hr ground truth)

| Metric | Value |
|--------|-------|
| **CH19 4-worker recall** | 32/32 = 100% |
| **CH19 2-worker recall** | 14/14 = 100% |
| **CH19 false positives** | 0 |
| **CH21 accepted recall** | 66/72 = 92% |

## How It Works

### CH19 — Cutting Counter (v5)

Detects when cut blanket pieces slide off the white table, exposing the surface:

1. **Multi-scale derivative detection**: Two derivative windows (d35 for strong 4-worker cuts, d25 for rapid 2-worker scissor cuts) run in parallel
2. **Echo suppression**: Removes weak "bounce" detections following strong events
3. **Close-pair merge**: When two detections fire within 2.5s and at least one is strong (deriv >30), they're the same physical event — keeps only the stronger one. Consecutive 2-worker cuts (weak deriv <25) pass through unaffected
4. **Multi-ROI cross-validation**: Left-table ROI provides secondary signal for confidence scoring
5. **Break detection**: Suppresses counting when table is empty (brightness >235)

### CH21 — Passing Counter (v4)

Tracks blankets through the weighing scale chokepoint:

1. **Scale detection**: Reference-frame comparison with hysteresis (ON=25, OFF=15 dead zone)
2. **Table texture**: Grayscale std deviation tracks blanket presence on folding table
3. **Accept/reject classification**: Scale events = accepted. Table events without nearby scale event = rejected
4. **Drift recovery**: Auto-recalibrates when stuck "loaded" >10s (handles physical scale movement)
5. **Lighting change detection**: Pauses counting during sudden brightness jumps, recalibrates on recovery

## Files

| File | Description |
|------|-------------|
| `cutting_counter.py` | CH19 cutting counter v5 (~700 lines) |
| `blanket_counter.py` | CH21 passing counter v4 (~830 lines) |
| `generate_dashboard.py` | Reads both JSON outputs, generates dashboard HTML |
| `run_full_day.py` | Batch processor for multi-segment NVR recordings |
| `blanket_tracker_dashboard.html` | Self-contained dual-camera dashboard (~2.5MB) |
| `index.html` | GitHub Pages entry point (copy of dashboard) |
| `compare_ground_truth.py` | CH21 ground truth comparison tool |
| `PROJECT_NOTES.md` | Comprehensive technical notes, all timestamps, insights |

## Requirements

```bash
pip install opencv-python numpy
```

## Usage

**Count cuts from CH19 video:**
```bash
python3 cutting_counter.py /path/to/ch19_video.mp4 --output cutting_results.json
```

**Count blankets from CH21 video:**
```bash
python3 blanket_counter.py /path/to/ch21_video.mp4 --output blanket_results.json
```

**Process full day of NVR recordings (both cameras in parallel):**
```bash
python3 run_full_day.py                  # Both cameras
python3 run_full_day.py --ch19-only      # Cutting only
python3 run_full_day.py --ch21-only      # Passing only
```
Place NVR files in `frames/New Long video data/Cutting/` and `frames/New Long video data/Passing/`.

**Generate dashboard:**
```bash
python3 generate_dashboard.py
```
This reads `cutting_fullday.json` and `blanket_fullday.json`, then generates `blanket_tracker_dashboard.html`. Open in any browser.

**Compare against ground truth (CH21):**
```bash
python3 compare_ground_truth.py blanket_count_1hr_v4.json
```

## Dashboard

The dashboard is a single self-contained HTML file with embedded data — no server needed. Features:

- **KPI cards**: Cuts, accepted, rejected, rates for both cameras
- **Hourly breakdown table**: Per-hour stats with visual bars and confidence distribution
- **Combined production timeline**: Dual-axis step chart showing both cameras over time
- **Signal charts**: Raw brightness derivative (CH19) and scale/table signals (CH21)
- **Production breakdown**: Adaptive interval (30-min for full day) grouped bar chart
- **Session summary**: Key stats per camera

Dark theme, Canvas API rendering, retina-aware. No external dependencies. Hosted on GitHub Pages.

## Architecture

```
NVR Camera Feed (1920x1080 @ 25fps)
    │
    ├── CH19: cutting_counter.py
    │   ├── Table ROI (820,240,1020,360) → brightness derivative
    │   ├── Left-table ROI (600,240,820,360) → cross-validation
    │   ├── Slide ROI (720,370,960,520) → motion metadata
    │   └── Output: cutting_fullday.json (1,726 cuts)
    │
    ├── CH21: blanket_counter.py
    │   ├── Scale ROI (1440,440,1520,500) → reference-frame diff
    │   ├── Table ROI (980,340,1240,450) → texture std
    │   └── Output: blanket_fullday.json (1,919 blankets)
    │
    ├── run_full_day.py (batch multi-segment processor)
    │
    └── generate_dashboard.py
        └── blanket_tracker_dashboard.html → index.html (GitHub Pages)
```

## Key Learnings

- **Derivative detection beats absolute thresholds**: Blanket color varies wildly (baseline 77-155), but the brightness *change* when a piece slides off is consistent
- **Hysteresis prevents chattering**: Dead zone between ON/OFF thresholds eliminates state oscillation
- **Physics-based filters work**: Min duration and min gap derived from actual worker pace
- **Multi-scale catches different work phases**: 4-worker and 2-worker cutting produce fundamentally different signals
- **Close-pair merge > echo suppression**: Double-detections aren't always weaker — they can be equally strong or stronger than the initial detection
