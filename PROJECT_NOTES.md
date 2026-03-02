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

### Visual Patterns for Accepted vs Rejected
- **Accepted blankets are ALWAYS fully folded** before being placed on the scale. The folded blanket sits as a neat rectangular bundle on the platform (pics 1 & 2 from user).
- **Rejected blankets are NEVER folded**. The worker tears the tag off the unfolded blanket and throws it to the right (pic 3 from user). The blanket is still spread/bunched, not a neat rectangle.
- This means the folding table activity pattern should differ: accepted blankets have a complete fold cycle (longer table coverage, neater texture), rejected blankets may have shorter/messier table signals since folding is aborted partway.
- **THE KEY INDICATOR: The final fold.** Accepted blankets ALWAYS get a complete final fold (compact layered rectangle). Rejected blankets NEVER get the final fold — the worker aborts folding, tears the tag, and throws it away. If we can detect whether the final fold happened on the table, that's the definitive accepted/rejected signal.
- **Reject pile is NOT reliable** — its location varies and blankets get regularly taken away. Do NOT use a reject pile ROI.
- **Better approach: Track the acceptance pile** (where folded blankets land after weighing, tossed left by worker A). Check if a folded blanket was thrown there after a table cycle. The acceptance pile is more stable/predictable than the reject pile.
- **Detection ideas**:
  1. Table texture pattern change: final fold creates a compact, high-texture rectangle vs spread-out blanket. The texture std signature should be different (sharp rise then stable high plateau for folded vs gradual messy signal for unfolded).
  2. Acceptance pile ROI: monitor the area where accepted blankets land. A new item appearing = confirmed acceptance.
  3. Shape analysis: folded blanket = compact rectangular blob in table ROI. Unfolded = spread across most of the ROI.

### Rejection Sequence (from reject.mp4, 10.5s clip)
Observed frame-by-frame from `/Users/sai/Downloads/reject.mp4`:
1. **t=0s**: Blanket spread flat on table, both workers visible. Scale empty (diff=0.3).
2. **t=2-3s**: Worker A inspects blanket, decides to reject. Blanket still unfolded.
3. **t=4-5s**: Worker A pulls blanket off table toward himself — NOT folding, dragging it off.
4. **t=6-7s**: Worker A throws blanket DOWN/LEFT onto reject pile on floor. Blanket flies through air, unfolded.
5. **t=7-8s**: Blanket lands on floor pile. Scale never touched (peak diff only 9.2).
6. **t=9-10s**: Workers reset, next blanket arriving on table.

**Critical observation**: The reject pile location varies and blankets get taken away regularly — NOT a reliable detection target. The acceptance pile (where folded blankets land after weighing) is more stable and predictable.

**Signal during rejection**: Scale diff peaked at only 9.2 (well below 25 ON threshold). Table texture barely fluctuated (72-75 range, hovering near threshold). The rejection happens quickly (~3-4s from decision to throw) and leaves almost no signal on either detector — this explains why 7/22 rejected blankets were completely undetected.

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

## ROI & Feature Diagnostic (Session 3)

### Motion Heatmap Analysis
Computed average motion heatmaps (|frame_after - frame_before|) for accepted vs rejected events:
- **Accepted motion**: Concentrated in bottom-center of frame (y:700+) = acceptance pile area
- **Rejected motion**: Concentrated around table area (worker throwing action)
- **Worker body motion dominates ALL candidate ROIs** near the table/scale → ROI-based landing zone detection not viable

### Candidate ROIs Tested (9 regions)
| ROI | Accepted | Rejected | Separation |
|-----|----------|----------|------------|
| far_right (1550,380,1750,560) | 12.6 | 10.8 | 1.09 |
| floor_center (1100,550,1350,720) | 39.7 | 50.3 | 0.87 |
| alz_tight (1280,520,1440,660) | 54.8 | 72.4 | 0.84 |
| alz_wide (1200,500,1460,700) | 55.0 | 65.9 | 0.76 |
| scale (1440,440,1520,500) | 16.1 | 10.6 | 0.42 |
| table (980,340,1240,450) | 43.7 | 44.7 | 0.05 |

**Conclusion**: No single ROI provides clean discrimination. Worker motion noise is the main obstacle.

### Feature Analysis (Diagnostic at GT Timestamps)
| Feature | Accepted (mean±std) | Rejected (mean±std) | Separation |
|---------|---------------------|---------------------|------------|
| peak_scale_diff | 87.3±11.1 | 21.9±20.2 | **4.18** |
| texture_slope_2s | -12.5±9.6 | -0.2±10.3 | **1.24** |
| peak_texture | 97.5±6.2 | 88.8±9.4 | **1.12** |
| final_texture | 67.8±6.4 | 75.3±8.1 | **1.03** |
| above_duration | 5.2±1.3 | 5.0±2.2 | 0.15 |

### Texture Slope: Why It Didn't Work in Production
- At GT timestamps (frame-seeking): accepted slope mean = -12.5, clear separation
- At detected table_blanket_off events (live tracking): slopes much weaker (-1.2 to -6.8)
- Reason: smoothing buffer dampens rapid changes; cycle dynamics differ from point-in-time measurement
- A threshold of -5.0 on table cycle slope misclassified 170/223 scale events as "weight-rejected"
- **Fix needed**: Compute slope only over LAST 0.5s of cycle, not last 2s. The lift signal is brief.

### Weight-Rejected Blankets (4 events: 2:33, 6:56, 10:33, 58:14)
These go on scale (peak_diff: 39.7, 83.6, 60.4, 56.9) then get rejected by weight.
- Scale duration: 1.4s, 3.2s, 1.6s, 9.0s
- Texture slope at GT: +0.5, -0.1, +0.6, +1.3 (all near zero vs accepted -12.5)
- Currently indistinguishable from accepted using scale data alone
- Post-scale directional motion analysis is impractical (worker motion noise)

---

## Version History

| Version | Changes | Results |
|---------|---------|---------|
| v1 | Scale-only detection | 223 accepted, no reject detection |
| v2 | Scale + table detection | 223 scale + 951 table cycles (73% noise) |
| v3 | Improved table (debounce, min duration 2s, gap 4s) + accept/reject classification | 223 acc + 86 rej = 309 total. 92% accepted, 50% rejected recall. |
| v4 | Table min duration 1.2s, texture profiling (peak, slope) as metadata | 223 acc + 103 rej = 326 total. 92% accepted, 50% rejected recall. More table events, same GT match. |

---

## Roadmap / Next Steps

1. **Better texture slope computation**:
   - Current: slope over last 2s of table cycle → too noisy
   - Needed: slope over last 0.5s only, or rate-of-change at the exact moment of lift
   - Could significantly improve weight-rejection detection (catches 4 currently missed events)

2. **ML classifier approach**:
   - With labeled data (peak_texture, texture_slope, scale_diff, duration, above_segments)
   - Even a simple logistic regression or decision tree could improve on threshold-based rules
   - Need more GT data for training/validation

3. **Live counting from RTSP feeds** (primary goal):
   - `--live` flag already exists with reconnection logic
   - Need: RTSP URL for CH21, machine on same network
   - Dashboard would need periodic JSON refresh or WebSocket

4. **WhatsApp alerts**: Significant events/changes notification

5. **Improve accepted recall** (92% → 95%+):
   - Adaptive thresholds during man-blocking periods
   - Lower ON_THRESHOLD when man detected (accept lower diffs)

---

## CH19 Cutting Counter

### Process
4 workers at white cutting table (2 in back cut, 2 in front slide pieces off). Blanket spread across table → cut → piece slides down front → repeat. After 29:37 mark, only 2 workers.

### Video
- Full: `/Users/sai/Downloads/Full cut vido.mp4` (59.9 min, 25fps, 1920x1080, 898MB)
- Clip: `/Users/sai/Downloads/Cutting clip.mp4` (12.1s, 2 cuts at ~2s and ~8s)

### Key Timestamps (user-provided ground truth)
| Time | Event |
|------|-------|
| 0:10 | Process starts, blanket on table |
| 0:14 | First physical cut |
| 0:20 | First slide (piece off table) |
| 3:07 | Last piece of first blanket set |
| 3:09 | Break |
| 4:01 | New blanket set, new color |
| 14:34 | Sliding pile starts getting big |
| 29:37 | Break, then only 2 workers |
| 30:35 | Discard cut + first main cut (2 workers) |
| 30:41 | Discard cut |
| 30:45 | Main cut |
| 33:41 | Another cut |
| 33:47 | Last discard cut |
| 39:22-42:48 | Noisy small pieces period |
| 42:40 | Last cut (cloth doesn't fall completely) |
| 42:40+ | No cutting |

### Detection: v1 (absolute brightness thresholds) — FAILED
- TABLE_ROI (820,240,1020,360) mean brightness: covered=80-86, exposed=150-182 in short clip
- ON=120, OFF=100 with hysteresis state machine
- **Problem**: "Covered" baseline varies 77-155 depending on blanket color. ON=120 triggered 76.5% of full video. Completely unusable.

### Detection: v2 (brightness derivative) — CURRENT
**Method**: Detect positive derivative spikes in smoothed table brightness.
- When a piece slides off → white table exposed → brightness INCREASES rapidly
- Compute derivative: current smoothed brightness − smoothed brightness 2s ago
- Spike above threshold 20 = cut detected
- Color-agnostic: detects CHANGE, not absolute level

**Signal analysis across full video:**
| Phase | Brightness baseline | Typical derivative | Detection quality |
|-------|--------------------|--------------------|-------------------|
| 4 workers, dark fabric (0:19-3:07) | ~77 | +35 to +120 | Excellent |
| 4 workers, lighter fabrics (4:01-29:37) | 90-155 (varies) | +20 to +100 | Good |
| 2 workers (29:37-33:49) | 125-157 | +15 to +28 | Moderate (some misses) |
| Small pieces (39:22-42:48) | variable | wild oscillations | Noisy |
| Empty table / breaks | ~248 | ~0 | Suppressed correctly |

**Break periods detected (brightness > 235 for > 3s):**
- 3:08-3:45, 10:47-11:29, 20:34-20:49, 25:52-26:33, 29:36-29:58, 33:49-34:25, 39:21-39:40, 42:49-end

**Results (v2, full video):**
- 333 cuts detected
- Active time: ~56 min, Break time: ~3.8 min
- Rate: 5.9 cuts/min (averaged over all phases)
- Processing: 497 fps (19.9x realtime)

### ROI Analysis (from frame extraction)
- 13 frames extracted from cutting clip to `frames/ch19/`
- Table surface clearly white; right side (820-1020, 240-360) shows best signal
- Slide zone (720-960, 370-520) shows motion spikes during cuts but unreliable in 2-worker phase
- Frame-diff grid analysis confirmed motion hotspot at (720,180)-(960,360) during all cut events

### CH19 Files
| File | Purpose |
|------|---------|
| `cutting_counter.py` | CH19 counter v2 (derivative-based, ~300 lines) |
| `cutting_full_v2.json` | Full video results (333 cuts, events + frame_data) |
| `cutting_test.json` | Short clip results (2 cuts) |
| `frames/ch19/` | 13 extracted frames for ROI analysis |

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

9. **Diagnostic at GT timestamps ≠ actual detection signals**: Texture slope measured at GT timestamps (frame-seeking) showed clear separation (1.24). But the same feature computed during live table cycle tracking was much weaker due to smoothing, cycle dynamics, and overlap. Always validate features in the actual processing pipeline.

10. **Worker body motion is the #1 noise source**: All ROIs near the workspace are dominated by worker movement. Any motion-based feature must account for this.
