# Blanket Tracker — Project Notes & Insights

## Client & Setup
- **Client**: Mr. Goyal, blanket factory in Panipat, Haryana
- **Goal**: Count blankets being processed via NVR security camera feeds
- **Cameras**: CH19 (cutting floor), CH21 (finishing station — weighing scale is the definitive counting chokepoint)
- **NVR format**: Files like `NVR_ch19_main_YYYYMMDDHHMMSS_*.mp4`, 1920x1080 @ 25fps
- **Test videos**:
  - Short clip: `NVR_ch21_main_20260227110000_20260227110009.mp4` (8.36s, 2 blankets)
  - 1-hour: `NVR_ch21_main_20260227110009_20260227120000 (1).mp4` (59.9 min, 897MB, 89817 frames)

## Worker Workflow (CH21)
Worker B throws blanket on table → A & B fold → A places on scale → checks weight:
- **Tosses LEFT** if accepted (weighed, good weight)
- **Tosses RIGHT** if rejected (not weighed properly, bad quality/weight)
- Worker B may overlap next blanket on table during weighing

## Detection ROIs (1920x1080 CH21)
- **Scale**: (1440, 440, 1520, 500) — 80x60 pixel crop of weighing platform (far right of frame)
- **Table**: (980, 340, 1240, 450) — folding table top surface (center-right of frame)

---

## Ground Truth Data (27 Feb 2026, 11:00-12:00)

### Analyzed Ranges
- First 12 minutes: 0:00 – 12:00
- Last ~8 minutes: 52:00 – 59:21
- Middle (12:00 – 52:00) NOT analyzed

### Major Events
| Time | Event |
|------|-------|
| 10:11 | Scale was physically moved/shifted |
| 21:55 | Scale was physically moved/shifted again |
| 27:16 | Worker break |
| 7:53 – 8:10 | Man standing in front of scale, blocking camera view |
| 51:45 – 54:45 | Man (black hair) standing directly in front of scale, very hard to see |
| 59:21 – 59:51 (end) | Man blocks scale's view until end of video |

### Accepted Blankets (Weighed, Tossed LEFT) — 72 total

**0:00 – 12:00 (39 blankets)**
```
0:05  0:12  0:20  0:27  0:33  0:41  0:49  0:55  1:01  1:08
1:15  1:23  1:29  1:38  1:55  2:02  2:10  2:17  2:25  2:49
2:59  3:06  3:14  3:21  3:51  4:29  4:50  5:21  7:21  8:04
8:26  8:35  8:43  8:49  8:56  9:01  9:22  11:47  11:58
```

**52:00 – 55:00 (12 blankets)**
*Note: 54:03 & 54:40 — man with black hair directly in front of scale, very hard to see but still accepted*
```
52:47  52:54  53:22  53:32  53:44  53:54  54:03  54:11  54:18  54:29
54:40  54:50
```

**55:00 – 59:21 (21 blankets)**
```
55:04  55:33  55:43  56:02  56:10  56:17  56:28  56:38  56:48  57:01
57:09  57:16  57:31  57:40  57:48  57:56  58:03  58:29  58:50  59:00
59:08
```

### Rejected Blankets (NOT Weighed, Tossed RIGHT) — 22 total

**0:00 – 12:00 (18 blankets)**
```
1:47  2:33  4:41  4:59  5:12  5:56  6:16  6:32  6:56  7:11
7:47  8:11  10:23  10:33  10:47  11:02  11:11  11:39
```

**52:00 – 59:21 (4 blankets)**
```
53:13  55:55  58:14  58:42
```

### Notes on specific timestamps
- 1:47 was REJECTED (tossed right, not weighed) — initially misclassified as accepted
- 54:03 & 54:40 are accepted but man blocking makes detection very hard
- 59:08 — last accepted blanket before man blocks at 59:21

---

## Scale Detection — How It Works

**Method**: Reference-frame comparison (color-agnostic)
- Learn an "empty scale" reference from calibration frames
- Each frame: compute mean absolute diff between scale ROI and reference
- Hysteresis state machine: empty ↔ loaded
- Count completed cycles (loaded → empty)

**Configuration (hardened from ground truth)**:
| Parameter | Value | Why |
|-----------|-------|-----|
| ON_THRESHOLD | 25 | Diff must exceed this to trigger "loaded" |
| OFF_THRESHOLD | 15 | Diff must fall below this for "empty" (dead zone 15-25 prevents chattering) |
| SMOOTH_WINDOW | 13 | ~0.52s at 25fps |
| DEBOUNCE_FRAMES | 5 | Rising edge must sustain 5 frames before state change |
| MIN_ON_FRAMES | 15 | 0.6s minimum — all real blankets are ≥0.68s |
| MIN_CYCLE_GAP | 100 | 4s at 25fps — real blankets always ≥5s apart |
| MAX_LOADED_FRAMES | 250 | 10s — drift detection: no blanket stays this long |
| DRIFT_MARGIN | 1.5 | If diff < ON_THRESHOLD × 1.5, it's baseline drift |
| REF_ADAPT_RATE | 0.005 | 0.5% blend when idle >3s |
| REF_IDLE_FRAMES | 75 | Must be empty 3s before adapting reference |

**Performance**: 92% accepted recall (66/72), 4/22 rejected falsely counted as accepted

### 6 Missed Accepted Blankets
| Time | Why missed |
|------|-----------|
| 54:03 | Peak diff 23.3, below 25 threshold — man blocking |
| 54:40 | Peak diff 18.0 — man blocking |
| 55:33 | Only 0.4s duration (10 frames), below 0.6s minimum |
| 56:10 | Peak diff 20.7, below threshold |
| 56:17 | Peak diff 26.2, borderline — debounce prevents detection |
| 59:08 | Peak diff 20.2, below threshold — near man blocking at 59:21 |

### 4 Rejected Blankets Falsely Counted as Accepted
| Time | Peak diff | Duration | Why |
|------|-----------|----------|-----|
| 2:33 | 31.5 | 1.52s | Blanket actually went on scale, worker rejected based on weight |
| 6:56 | 81.5 | 3.52s | Same — weighed then rejected |
| 10:33 | 58.4 | 2.24s | Same |
| 58:14 | 39.3 | 1.00s | Same |

**Key insight**: These 4 blankets DID physically go on the scale. The worker checked the weight and THEN rejected them. Scale detection cannot distinguish this — it just sees a normal weighing cycle.

---

## Table Detection — How It Works

**Method**: Grayscale texture standard deviation in table ROI
- High std (>75) = textured surface = blanket present
- Low std (<75) = bare table = empty
- State machine: empty ↔ covered

**Configuration (improved)**:
| Parameter | Value | Why |
|-----------|-------|-----|
| TEXTURE_THRESHOLD | 75 | Std deviation boundary |
| SMOOTH_WINDOW | 9 | Smoothing buffer |
| MIN_CYCLE_FRAMES | 50 | 2s minimum (was 10 = 0.4s) |
| DEBOUNCE_FRAMES | 5 | Rising edge debounce |
| MIN_CYCLE_GAP | 100 | 4s between events |

**Before improvements**: 951 table cycles (73% were noise <2s)
**After improvements**: 244 table cycles (much closer to reality)

---

## Accept/Reject Classification — How It Works

**Method**: Cross-correlation of scale + table events (post-processing pass)

1. Every `scale_cycle_complete` → `blanket_accepted` (always)
2. For each `table_blanket_off`, check if ANY scale event within [-2s, +10s]
   - If yes → table-side of an accepted blanket (skip, already counted)
   - If no → `blanket_rejected` (tossed without weighing)

**Results on 1hr ground truth**:
- Accepted: 66/72 = 92% recall
- Rejected: 11/22 = 50% recall
- Total detected: 223 accepted + 86 rejected = 309 blankets

### Why 50% rejected recall
| Category | Count | Explanation |
|----------|-------|-------------|
| Correctly detected as rejected | 11/22 | Table cycle fires, no scale event |
| Detected as accepted (false) | 4/22 | Blanket DID touch scale, then rejected by weight |
| Not detected at all | 7/22 | No table or scale signal — too brief or too noisy |

### Rejected Blanket Physics (why they're hard to detect)
- 15/22 (68%) have peak scale_diff < 15 → **completely invisible to scale** — blanket never touches it
- 3/22 have scale_diff 15-25 → "brief touch" below ON threshold
- 4/22 have scale_diff > 25 → actually trigger scale (worker weighed, then rejected on weight)
- Median peak scale_diff for rejected: 9.9 (vs accepted detection diff median: 13.8)

---

## Processing Performance
- **1-hour video (89,817 frames)**: ~7 min 44 sec processing time
- **Speed**: ~7.7x real-time (processes 1 hour in <8 min)
- **CPU**: ~214% (2 cores via OpenCV threading)
- **Headroom for live**: Only needs ~13% of real-time capacity per frame

---

## Architecture & File Map

| File | Purpose |
|------|---------|
| `blanket_counter.py` | Main counter — scale + table detection + classification (~700 lines) |
| `blanket_tracker.py` | Legacy MOG2 motion tracker (deprecated) |
| `blanket_tracker_dashboard.html` | Dashboard with embedded 1hr data (~860 lines, ~400KB) |
| `compare_ground_truth.py` | Comparison tool with all ground truth timestamps |
| `blanket_count_1hr_v2.json` | 1hr results before accept/reject (223 blankets) |
| `blanket_count_1hr_v3.json` | 1hr results with accept/reject (223+86=309 blankets) |

---

## Roadmap / Next Steps

1. **Improve rejected detection**:
   - Add "reject pile" ROI (right side of table area) to detect directional toss motion
   - This would catch rejected blankets that skip the scale entirely
   - Alternatively: track motion vectors after table_off to detect LEFT vs RIGHT toss

2. **Live counting from RTSP feeds** (primary goal):
   - `--live` flag already exists with reconnection logic
   - Need: RTSP URL for CH21, machine on same network
   - Dashboard would need periodic JSON refresh or WebSocket

3. **WhatsApp alerts**: Significant events/changes notification

4. **Improve accepted recall** (92% → 95%+):
   - Adaptive thresholds during man-blocking periods
   - Lower ON_THRESHOLD when man detected (accept lower diffs)
   - Train simple classifier on accepted vs noise patterns

---

## Key Learnings

1. **Reference-frame comparison beats motion detection**: MOG2/optical flow is too noisy for production counting. Simple mean-absolute-diff against a learned reference is far more reliable.

2. **Hysteresis is essential**: Single threshold causes chattering. Dead zone between ON (25) and OFF (15) eliminates state oscillation.

3. **Physics-based filters work**: Min duration (0.6s) and min gap (4s) are derived from the actual physical process — no blanket can be weighed in <0.6s, and workers can't process faster than 1 every 5s.

4. **Drift detection saves accuracy**: Scale reference slowly changes due to lighting. Detecting "stuck loaded >10s" and auto-recalibrating prevents accumulating errors.

5. **Table signal is inherently noisier than scale**: Texture std fluctuates during folding (hands, shadows, wrinkles). Scale reference-frame diff is binary — either something is there or not.

6. **Rejected blankets skip the scale**: 68% of rejected blankets have zero scale signal. The only way to detect them is through the table or motion in the reject direction.

7. **Some rejected blankets ARE weighed**: 4/22 rejected blankets went on the scale (worker checked weight, then rejected). These are indistinguishable from accepted blankets using only scale data.

8. **Ground truth corrections matter hugely**: Initial "43 extra detections" turned out to be mostly real blankets once the ground truth was expanded. Always verify GT thoroughly.
