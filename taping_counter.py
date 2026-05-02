"""CH27 Taping Counter — counts taping cycles on two parallel tables.

Workers in the Panipat factory tape the two short ends of a folded blanket on
one of two independent taping tables (LEFT and RIGHT). After taping, the
blanket is tossed onto a central heap. Each "cycle" = blanket arrives →
taping happens → blanket leaves.

v1 algorithm (per table, fully independent):

  1. Per-frame texture std on the table ROI (np.std of grayscale region).
     Smoothed with a 7-frame deque (CH21 _smooth() pattern). Patterned
     blankets dominate the variance → std jumps cleanly when fabric covers
     the ROI even with hands moving inside it.

  2. Adaptive baseline: rolling median of last 30 s of std values, computed
     ONLY when both tables are empty AND no lighting change is in progress.
     Frozen while either table is "loaded" so active cycles can't bias it.
     (cutting_counter v6 pattern, generalized.)

  3. Hysteresis state machine with dead-zone:
        empty → loaded   when smoothed > baseline+ON_DELTA for ≥ MIN_ON_FRAMES
                         AND peak − baseline ≥ STRONG_DELTA at some point
                         (skin/hands have lower variance than patterned fabric)
        loaded → empty   when smoothed < baseline+OFF_DELTA for ≥ MIN_OFF_FRAMES
                         → emit taping_cycle_complete

  4. Cycle-duration gates at falling edge:
        < MIN_CYCLE_SEC (6s)  → drop, log to suppressed_candidates(too_short)
        > MAX_CYCLE_SEC (30s) → emit + flag long_cycle=True (still likely real)
        > STUCK_SEC (90s)     → drop + force per-table baseline recalibration

  5. Overlap detector — runs in parallel with the state machine. While loaded,
     watches the smoothed-std derivative for a downward spike (one blanket
     leaving) followed within 1.5 s by an upward spike (next blanket arriving)
     WITHOUT std crossing back below baseline+OFF_DELTA. When that pattern
     fires AFTER cycle has been loaded ≥ MIN_CYCLE_SEC, emits a cycle complete
     at the trough and resets cycle-start to the recovery point — without
     leaving the loaded state. This catches back-to-back cycles where the
     state machine alone would fuse them.

  6. Lighting-change detection (port from blanket_counter._check_lighting):
     whole-frame mean luma delta > 25 → emit lighting_change, pause both state
     machines, force recalibrate when stabilized.

  7. Heap ROI tracked at 0.4 Hz (validation only — never gates events).

  8. Suppressed-candidates audit log (cutting_counter v6 pattern).

Detection is color-agnostic: it tracks CHANGE in texture variance, not
absolute brightness, so it works regardless of blanket color.

Usage:
  python3 taping_counter.py /path/to/video.mp4 [--output results.json] [--debug]
"""

import argparse
import json
import os
import sys
import time
from collections import deque
from statistics import median

import cv2
import numpy as np


# ═══════════════════════════════════════════════════════════════
#  ROI CONFIG (1920×1080) — calibrate via taping_roi_calibrator.py
# ═══════════════════════════════════════════════════════════════

# Initial ROI estimates (refine on real footage with the calibrator script).
# Rule: place each table ROI on the half farther from the worker's standing
# position so worker hands/arms don't dominate the std signal.
LEFT_TABLE_ROI  = (60, 700, 580, 1000)    # x1, y1, x2, y2
RIGHT_TABLE_ROI = (1280, 700, 1860, 1000)
HEAP_ROI        = (700, 350, 1220, 750)   # validation-only
TAPE_DISPENSER_LEFT_ROI  = (0, 750, 120, 950)     # v2 cross-validator
TAPE_DISPENSER_RIGHT_ROI = (1800, 750, 1920, 950) # v2 cross-validator


# ═══════════════════════════════════════════════════════════════
#  ALGORITHM CONFIG — v1
# ═══════════════════════════════════════════════════════════════

# Smoothing
SMOOTH_WINDOW = 3        # frames (~0.12 s at 25 fps) — keep responsive

# Baseline (adaptive) — applied to BOTH mean and std signals
BASELINE_WINDOW_SEC = 30        # rolling buffer of readings
MEAN_BASELINE_PCTL = 80         # empty table is BRIGHT → 80th percentile of recent means
STD_BASELINE_PCTL  = 20         # empty/uniform texture LOW → 20th percentile of recent stds

# Combined activity score per frame:
#   activity = max(0, baseline_mean - smoothed_mean) + max(0, smoothed_std - baseline_std)
# - Captures dark blankets (mean drops) AND patterned blankets (std rises)
# - Both contributions are clipped at 0 so empty frames score ~0
# - Symmetric across blanket type — single threshold works for both tables

ACTIVITY_ON = 4.0        # rising edge when sustained activity > this
ACTIVITY_OFF = 1.5       # falling edge when sustained activity < this (dead zone)
ACTIVITY_STRONG = 6.0    # peak activity must exceed this once (rejects worker-only blips)

MIN_ON_FRAMES = 8        # 0.32 s sustained above ON to confirm rising edge
MIN_OFF_FRAMES = 30      # 1.2 s sustained below OFF to confirm falling edge
                         # (long enough to ignore mid-cycle folding dips ~1.6s recovery)

# Cycle duration gates
# Per GT review of /Users/sai/Downloads/sample 2.mp4 (90s, 12 tosses):
#   - RIGHT cycles ~8-11s typically
#   - LEFT cycles can be 6s up to 35s+ (the striped blanket case)
MIN_CYCLE_SEC = 4.0      # < this → drop as too_short (hand blip)
MAX_CYCLE_SEC = 60.0     # > this → emit but flag as long_cycle
STUCK_SEC = 180.0        # > this → drop + force baseline recalibration

# Overlap detector (back-to-back cycles)
DERIV_WINDOW_FRAMES = 12         # ~0.5 s derivative window
DROP_DELTA = 2.5                 # downward spike threshold (matches narrower dynamic range)
RISE_DELTA = 2.5                 # upward spike threshold
OVERLAP_RECOVERY_SEC = 1.5       # recovery must happen within this window

# Lighting-change detection (whole frame)
LIGHTING_DELTA = 25              # mean-luma jump that triggers a pause
LIGHTING_RESTORE_DELTA = 5       # delta below this for N frames → restored
LIGHTING_RESTORE_FRAMES = 25     # 1 s

# Heap sampling
HEAP_SAMPLE_EVERY_FRAMES = 60    # ~2.4 s at 25 fps (every 60 frames)

# Frame data sampling for dashboard
FRAME_DATA_EVERY_FRAMES = 5      # log every 5th frame (~5 Hz)

# Warm-up — frames before any rising edge can fire
# (First state is undefined; we don't want to count residuals from pre-recording.)
WARMUP_FRAMES = 25               # ~1 s settle


# ═══════════════════════════════════════════════════════════════
#  PER-TABLE TRACKER
# ═══════════════════════════════════════════════════════════════

class _TableTracker:
    """Independent state machine + signal tracker for one taping table.

    Computes per-frame:
      - mean_smoothed: smoothed mean intensity of ROI
      - std_smoothed:  smoothed texture std of ROI
      - activity_score = max(0, mean_baseline - mean_smoothed)
                       + max(0, std_smoothed - std_baseline)
        Captures both dark blankets (mean drops vs bright empty table) and
        patterned blankets (std rises vs uniform empty table). Both terms
        clipped at 0 so the empty-state activity is ~0.

    Baselines are running percentiles of recent values:
      - mean_baseline = 80th percentile (empty table is bright)
      - std_baseline  = 20th percentile (empty / uniform region has low std)
    """

    def __init__(self, name, roi, fps, config):
        self.name = name             # "left" | "right"
        self.roi = roi
        self.fps = fps
        self.cfg = config

        # Smoothing — separate deques for mean and std
        self.mean_smooth_buf = deque(maxlen=config["smooth_window"])
        self.std_smooth_buf  = deque(maxlen=config["smooth_window"])

        # Smoothed activity history (for derivative + overlap detector)
        max_window = max(config["deriv_window_frames"], 60)
        self.activity_history = deque(maxlen=max_window + 5)
        # Convenience aliases — kept so frame_data sampling stays clean
        self.smoothed_history = self.activity_history  # legacy alias

        # Adaptive baseline buffers (rolling raw mean / std readings)
        self.mean_buffer = deque(maxlen=int(config["baseline_window_sec"] * fps))
        self.std_buffer  = deque(maxlen=int(config["baseline_window_sec"] * fps))
        self.mean_baseline = 0.0
        self.std_baseline = 0.0
        # Last computed values (exposed for frame_data)
        self.last_mean = 0.0
        self.last_std = 0.0
        self.last_activity = 0.0
        self.current_baseline = 0.0  # legacy field for frame_data dump (= activity baseline = 0)

        # State
        self.state = "empty"
        self.frame_idx = 0           # local; outer loop sets this
        self.warmup_done = False

        # Cycle tracking
        self.cycle_start_frame = None
        self.cycle_start_t = None
        self.peak_std = 0.0
        self.peak_t = None
        self.std_samples = []        # mean computed on cycle close
        self.strong_seen = False     # has peak−baseline ≥ STRONG_DELTA happened yet?

        # Edge debounce counters
        self.frames_above_on = 0
        self.frames_below_off = 0

        # Overlap detector state
        # After a downward spike during loaded state, mark trough and start
        # waiting for an upward spike within OVERLAP_RECOVERY_SEC.
        self.overlap_trough_frame = None
        self.overlap_trough_t = None
        self.overlap_trough_value = None

    # ── Signal helpers ───────────────────────────────────────────

    def _smooth_mean(self, raw):
        self.mean_smooth_buf.append(raw)
        return float(np.mean(self.mean_smooth_buf))

    def _smooth_std(self, raw):
        self.std_smooth_buf.append(raw)
        return float(np.mean(self.std_smooth_buf))

    def _derivative(self):
        """Activity change over DERIV_WINDOW_FRAMES."""
        n = self.cfg["deriv_window_frames"]
        if len(self.activity_history) < n + 1:
            return 0.0
        return float(self.activity_history[-1] - self.activity_history[-1 - n])

    def update_baseline(self, gather_ok, raw_mean, raw_std):
        """Append latest raw mean/std to rolling buffers and recompute baselines.
        Percentile recomputed every BASELINE_RECOMPUTE_EVERY frames (the buffer
        changes slowly and the percentile call is the per-frame hot spot).
        """
        if not gather_ok:
            return
        self.mean_buffer.append(raw_mean)
        self.std_buffer.append(raw_std)
        self._baseline_tick = getattr(self, "_baseline_tick", 0) + 1
        if self._baseline_tick >= self.cfg.get("baseline_recompute_every", 10):
            self._baseline_tick = 0
            if self.mean_buffer:
                mean_arr = np.fromiter(self.mean_buffer, dtype=float)
                self.mean_baseline = float(np.percentile(mean_arr, self.cfg["mean_baseline_pctl"]))
            if self.std_buffer:
                std_arr = np.fromiter(self.std_buffer, dtype=float)
                self.std_baseline = float(np.percentile(std_arr, self.cfg["std_baseline_pctl"]))

    def force_recalibrate(self):
        """Wipe baseline buffers; baselines hold last value until refilled."""
        self.mean_buffer.clear()
        self.std_buffer.clear()

    # ── Per-frame update ─────────────────────────────────────────

    def update(self, gray_frame, frame_idx, t_sec, paused):
        """Process one frame; return (event_or_None, raw_mean, raw_std, activity)."""
        self.frame_idx = frame_idx
        x1, y1, x2, y2 = self.roi
        region = gray_frame[y1:y2, x1:x2]
        raw_mean = float(np.mean(region))
        raw_std  = float(np.std(region))
        self.last_mean = raw_mean
        self.last_std = raw_std

        mean_smoothed = self._smooth_mean(raw_mean)
        std_smoothed  = self._smooth_std(raw_std)

        # Seed baselines from first frame to avoid divide-by-zero / empty checks
        if not self.mean_buffer:
            self.mean_buffer.append(raw_mean)
            self.mean_baseline = raw_mean
        if not self.std_buffer:
            self.std_buffer.append(raw_std)
            self.std_baseline = raw_std

        # Combined activity score — both terms clipped at 0 so empty ≈ 0
        activity = (max(0.0, self.mean_baseline - mean_smoothed)
                    + max(0.0, std_smoothed - self.std_baseline))
        self.last_activity = activity
        self.activity_history.append(activity)

        # Warm-up: don't allow rising edge before WARMUP_FRAMES.
        # If activity is already elevated at end of warmup, snap directly to
        # loaded state — handles the "blanket already on table when recording
        # starts" case so the first toss event is still detectable.
        if not self.warmup_done and frame_idx >= self.cfg["warmup_frames"]:
            self.warmup_done = True
            if activity > self.cfg["activity_on"]:
                self._open_cycle(frame_idx, t_sec, activity)

        if paused:
            self.frames_above_on = 0
            self.frames_below_off = 0
            self.overlap_trough_frame = None
            return None, raw_mean, raw_std, activity

        on_thresh = self.cfg["activity_on"]
        off_thresh = self.cfg["activity_off"]
        strong_thresh = self.cfg["activity_strong"]

        event = None

        if self.state == "empty":
            if not self.warmup_done:
                self.frames_above_on = 0
            elif activity > on_thresh:
                self.frames_above_on += 1
                if self.frames_above_on >= self.cfg["min_on_frames"]:
                    self._open_cycle(frame_idx, t_sec, activity)
            else:
                self.frames_above_on = 0

        else:  # loaded
            if activity > self.peak_std:        # peak_std now stores peak ACTIVITY
                self.peak_std = activity
                self.peak_t = t_sec
            self.std_samples.append(activity)
            if activity >= strong_thresh:
                self.strong_seen = True

            if activity < off_thresh:
                self.frames_below_off += 1
                if self.frames_below_off >= self.cfg["min_off_frames"]:
                    event = self._close_cycle(frame_idx, t_sec, via_overlap=False)
            else:
                self.frames_below_off = 0

            # Overlap detector — only after min cycle duration met
            if event is None and self.cycle_start_t is not None:
                age = t_sec - self.cycle_start_t
                if age >= self.cfg["min_cycle_sec"]:
                    event = self._check_overlap(frame_idx, t_sec, activity)

        return event, raw_mean, raw_std, activity

    def _open_cycle(self, frame_idx, t_sec, activity):
        self.state = "loaded"
        self.cycle_start_frame = frame_idx
        self.cycle_start_t = t_sec
        self.peak_std = activity              # now stores peak ACTIVITY
        self.peak_t = t_sec
        self.std_samples = [activity]
        self.strong_seen = activity >= self.cfg["activity_strong"]
        self.frames_below_off = 0
        self.frames_above_on = 0
        self.overlap_trough_frame = None

    def _close_cycle(self, frame_idx, t_sec, via_overlap):
        """Common close logic — returns ('event'|'suppressed', dict) or None."""
        if self.cycle_start_t is None:
            self.state = "empty"
            return None

        duration = t_sec - self.cycle_start_t
        peak_activity = self.peak_std            # field re-purposed for activity
        mean_activity = float(np.mean(self.std_samples)) if self.std_samples else peak_activity
        baseline_mean_at_close = self.mean_baseline
        baseline_std_at_close = self.std_baseline

        # Reset edge counters
        self.frames_below_off = 0
        self.frames_above_on = 0

        result = None
        suppressed = None

        if duration > self.cfg["stuck_sec"]:
            self.force_recalibrate()
            self.state = "empty"
            self.cycle_start_t = None
            return ("suppressed", {
                "table": self.name,
                "time_sec": round(t_sec, 2),
                "frame": frame_idx,
                "reason": "stuck_recalibrate",
                "duration_sec": round(duration, 2),
                "peak_activity": round(peak_activity, 2),
            })

        if duration < self.cfg["min_cycle_sec"]:
            self.state = "empty"
            self.cycle_start_t = None
            return ("suppressed", {
                "table": self.name,
                "time_sec": round(t_sec, 2),
                "frame": frame_idx,
                "reason": "too_short",
                "duration_sec": round(duration, 2),
                "peak_activity": round(peak_activity, 2),
            })

        if not self.strong_seen:
            self.state = "empty"
            self.cycle_start_t = None
            return ("suppressed", {
                "table": self.name,
                "time_sec": round(t_sec, 2),
                "frame": frame_idx,
                "reason": "weak_peak",
                "duration_sec": round(duration, 2),
                "peak_activity": round(peak_activity, 2),
            })

        # Real cycle — emit
        result = {
            "type": "taping_cycle_complete",
            "table": self.name,
            "time_sec": round(t_sec, 2),
            "frame": frame_idx,
            "cycle_start_sec": round(self.cycle_start_t, 2),
            "cycle_duration_sec": round(duration, 2),
            "peak_activity": round(peak_activity, 2),
            "mean_activity": round(mean_activity, 2),
            "mean_baseline_at_close": round(baseline_mean_at_close, 2),
            "std_baseline_at_close": round(baseline_std_at_close, 2),
            "long_cycle": duration > self.cfg["max_cycle_sec"],
            "via_overlap_detector": via_overlap,
        }

        if via_overlap:
            # Reset to a new in-flight cycle starting at current point
            self.state = "loaded"
            self.cycle_start_frame = frame_idx
            self.cycle_start_t = t_sec
            cur_act = self.activity_history[-1]
            self.peak_std = cur_act
            self.peak_t = t_sec
            self.std_samples = [cur_act]
            self.strong_seen = cur_act >= self.cfg["activity_strong"]
        else:
            self.state = "empty"
            self.cycle_start_t = None

        return ("event", result)

    def _check_overlap(self, frame_idx, t_sec, smoothed):
        """Look for downward spike + upward spike pattern (back-to-back blankets)."""
        d = self._derivative()

        # Phase 1 — downward spike: mark trough
        if self.overlap_trough_frame is None and d <= -self.cfg["drop_delta"]:
            self.overlap_trough_frame = frame_idx
            self.overlap_trough_t = t_sec
            self.overlap_trough_value = smoothed
            return None

        if self.overlap_trough_frame is None:
            return None

        # Have a trough; phase 2 — wait for upward spike within window
        elapsed = t_sec - self.overlap_trough_t
        if elapsed > self.cfg["overlap_recovery_sec"]:
            # Window expired without recovery — clear trough
            self.overlap_trough_frame = None
            return None

        if d >= self.cfg["rise_delta"]:
            # Recovery — emit cycle at trough timestamp
            close_t = self.overlap_trough_t
            close_f = self.overlap_trough_frame
            self.overlap_trough_frame = None
            return self._close_cycle(close_f, close_t, via_overlap=True)

        return None


# ═══════════════════════════════════════════════════════════════
#  TAPING COUNTER
# ═══════════════════════════════════════════════════════════════

class TapingCounter:
    """Process a CH27 taping video, return per-table cycle counts."""

    def __init__(self, source, **config):
        self.source = source
        self.version = config.get("version", "v1")
        self.debug = config.get("debug", False)

        # Roi & params
        self.left_roi  = config.get("left_table_roi",  LEFT_TABLE_ROI)
        self.right_roi = config.get("right_table_roi", RIGHT_TABLE_ROI)
        self.heap_roi  = config.get("heap_roi",        HEAP_ROI)

        # Tracker config dict (passed by reference)
        self.tracker_cfg = {
            "smooth_window":          config.get("smooth_window", SMOOTH_WINDOW),
            "baseline_window_sec":    config.get("baseline_window_sec", BASELINE_WINDOW_SEC),
            "mean_baseline_pctl":     config.get("mean_baseline_pctl", MEAN_BASELINE_PCTL),
            "std_baseline_pctl":      config.get("std_baseline_pctl",  STD_BASELINE_PCTL),
            "activity_on":            config.get("activity_on",     ACTIVITY_ON),
            "activity_off":           config.get("activity_off",    ACTIVITY_OFF),
            "activity_strong":        config.get("activity_strong", ACTIVITY_STRONG),
            "min_on_frames":          config.get("min_on_frames",   MIN_ON_FRAMES),
            "min_off_frames":         config.get("min_off_frames",  MIN_OFF_FRAMES),
            "min_cycle_sec":          config.get("min_cycle_sec",   MIN_CYCLE_SEC),
            "max_cycle_sec":          config.get("max_cycle_sec",   MAX_CYCLE_SEC),
            "stuck_sec":              config.get("stuck_sec",       STUCK_SEC),
            "baseline_recompute_every": config.get("baseline_recompute_every", 10),
            "deriv_window_frames":    config.get("deriv_window_frames", DERIV_WINDOW_FRAMES),
            "drop_delta":             config.get("drop_delta", DROP_DELTA),
            "rise_delta":             config.get("rise_delta", RISE_DELTA),
            "overlap_recovery_sec":   config.get("overlap_recovery_sec", OVERLAP_RECOVERY_SEC),
            "warmup_frames":          config.get("warmup_frames", WARMUP_FRAMES),
        }

        # Lighting + sampling
        self.lighting_delta          = config.get("lighting_delta", LIGHTING_DELTA)
        self.lighting_restore_delta  = config.get("lighting_restore_delta", LIGHTING_RESTORE_DELTA)
        self.lighting_restore_frames = config.get("lighting_restore_frames", LIGHTING_RESTORE_FRAMES)
        self.heap_sample_every       = config.get("heap_sample_every", HEAP_SAMPLE_EVERY_FRAMES)
        self.frame_data_every        = config.get("frame_data_every", FRAME_DATA_EVERY_FRAMES)
        # Frame skipping — decode every Nth frame only (HEVC decode is the bottleneck on long files).
        # 1 = no skip (default), 2 = process every 2nd frame (~2x speedup on slow decode).
        # All time-based thresholds remain in seconds, so skipping doesn't change behavior.
        self.frame_step              = max(1, int(config.get("frame_step", 1)))

        # Output
        self.events = []
        self.breaks = []
        self.suppressed = []
        self.frame_data = []
        self.heap_trace = []

        # Will be set in run()
        self.fps = 25.0
        self.left = None
        self.right = None

        # Lighting state
        self.frame_luma_history = deque(maxlen=10)
        self.in_lighting_change = False
        self.lighting_pause_start_t = None
        self.lighting_restore_counter = 0

    # ── Lighting helpers ────────────────────────────────────────

    def _check_lighting(self, frame_idx, t_sec, frame_luma):
        """Returns True if a lighting transition is currently active (paused)."""
        prev = self.frame_luma_history[-1] if self.frame_luma_history else frame_luma
        self.frame_luma_history.append(frame_luma)
        delta = abs(frame_luma - prev)

        if not self.in_lighting_change:
            if delta > self.lighting_delta:
                # Enter pause
                self.in_lighting_change = True
                self.lighting_pause_start_t = t_sec
                self.lighting_restore_counter = 0
                self.breaks.append({
                    "type": "lighting_change",
                    "time_sec": round(t_sec, 2),
                    "frame": frame_idx,
                    "delta": round(delta, 1),
                })
                if self.debug:
                    print(f"  [{t_sec:6.1f}s] lighting_change Δ={delta:.1f}")
                return True
            return False

        # Currently paused — wait for stable signal
        if delta < self.lighting_restore_delta:
            self.lighting_restore_counter += 1
        else:
            self.lighting_restore_counter = 0

        if self.lighting_restore_counter >= self.lighting_restore_frames:
            self.in_lighting_change = False
            self.breaks.append({
                "type": "lighting_restored",
                "time_sec": round(t_sec, 2),
                "frame": frame_idx,
            })
            # Force per-table baseline recalibration
            if self.left:
                self.left.force_recalibrate()
            if self.right:
                self.right.force_recalibrate()
            if self.debug:
                print(f"  [{t_sec:6.1f}s] lighting_restored")
            return False

        return True

    # ── Main loop ───────────────────────────────────────────────

    def run(self):
        cap = cv2.VideoCapture(self.source)
        if not cap.isOpened():
            raise RuntimeError(f"Could not open {self.source}")

        self.fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration_sec = total_frames / self.fps if self.fps > 0 else 0

        self.left = _TableTracker("left", self.left_roi, self.fps, self.tracker_cfg)
        self.right = _TableTracker("right", self.right_roi, self.fps, self.tracker_cfg)

        print(f"CH27 Taping Counter {self.version}")
        print(f"  Source: {self.source}")
        print(f"  FPS: {self.fps}, Frames: {total_frames}, Duration: {duration_sec:.1f}s ({duration_sec/60:.1f} min)")
        print(f"  LEFT  ROI: {self.left_roi}")
        print(f"  RIGHT ROI: {self.right_roi}")
        print(f"  HEAP  ROI: {self.heap_roi}  (validation only)")
        print(f"  Activity hysteresis: ON>{self.tracker_cfg['activity_on']:.1f} / OFF<{self.tracker_cfg['activity_off']:.1f}, strong>{self.tracker_cfg['activity_strong']:.1f}")
        print(f"  Cycle gates: min={self.tracker_cfg['min_cycle_sec']:.0f}s, max={self.tracker_cfg['max_cycle_sec']:.0f}s, stuck={self.tracker_cfg['stuck_sec']:.0f}s")
        print(f"  Baselines: mean={self.tracker_cfg['mean_baseline_pctl']}th pct (empty bright), std={self.tracker_cfg['std_baseline_pctl']}th pct (empty low std)")
        print(f"  Overlap detector: drop≤-{self.tracker_cfg['drop_delta']:.0f} → rise≥+{self.tracker_cfg['rise_delta']:.0f} within {self.tracker_cfg['overlap_recovery_sec']:.1f}s")
        print(f"  Warm-up: {self.tracker_cfg['warmup_frames']} frames")

        start = time.time()
        frame_idx = 0
        progress_step = max(1, total_frames // 20) if total_frames > 0 else 1000

        while True:
            # Frame skipping: grab() is faster than read() because it skips decode.
            # We grab+retrieve only every frame_step-th frame; intermediate frames
            # are advanced via grab() alone (still pays demuxer cost but skips
            # YUV→BGR conversion and most of the HEVC inter-frame work).
            if self.frame_step > 1:
                for _ in range(self.frame_step - 1):
                    if not cap.grab():
                        break
                    frame_idx += 1
                ret, frame = cap.read()
            else:
                ret, frame = cap.read()
            if not ret:
                break
            t_sec = frame_idx / self.fps
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frame_luma = float(np.mean(gray))

            paused = self._check_lighting(frame_idx, t_sec, frame_luma)

            l_ret = self.left.update(gray, frame_idx, t_sec, paused)
            r_ret = self.right.update(gray, frame_idx, t_sec, paused)

            # update() returns (event_or_None, raw_mean, raw_std, activity)
            for ev, _rm, _rs, _act in (l_ret, r_ret):
                if ev is None:
                    continue
                kind, payload = ev
                if kind == "event":
                    self.events.append(payload)
                    if self.debug:
                        print(f"  [{payload['time_sec']:6.1f}s] cycle {payload['table']:5s} "
                              f"dur={payload['cycle_duration_sec']:5.1f}s "
                              f"peak_act={payload['peak_activity']:5.1f}"
                              f"{' (overlap)' if payload['via_overlap_detector'] else ''}")
                else:
                    self.suppressed.append(payload)
                    if self.debug:
                        print(f"  [{payload['time_sec']:6.1f}s] dropped {payload['table']:5s} "
                              f"reason={payload['reason']} dur={payload['duration_sec']:5.1f}s "
                              f"peak_act={payload['peak_activity']:5.1f}")

            # Update adaptive baselines (always except during lighting pause)
            gather_ok = not paused
            self.left.update_baseline(gather_ok, l_ret[1], l_ret[2])
            self.right.update_baseline(gather_ok, r_ret[1], r_ret[2])

            # Heap sampling
            if self.heap_roi and frame_idx % self.heap_sample_every == 0:
                hx1, hy1, hx2, hy2 = self.heap_roi
                hp = gray[hy1:hy2, hx1:hx2]
                self.heap_trace.append({
                    "time_sec": round(t_sec, 2),
                    "frame": frame_idx,
                    "mean": round(float(np.mean(hp)), 2),
                    "std": round(float(np.std(hp)), 2),
                })

            # Frame data sampling (for dashboard signal chart)
            if frame_idx % self.frame_data_every == 0:
                self.frame_data.append({
                    "frame": frame_idx,
                    "time_sec": round(t_sec, 2),
                    "left_mean":     round(self.left.last_mean, 2),
                    "left_std":      round(self.left.last_std, 2),
                    "left_activity": round(self.left.last_activity, 2),
                    "left_mean_base": round(self.left.mean_baseline, 2),
                    "left_std_base":  round(self.left.std_baseline, 2),
                    "right_mean":     round(self.right.last_mean, 2),
                    "right_std":      round(self.right.last_std, 2),
                    "right_activity": round(self.right.last_activity, 2),
                    "right_mean_base": round(self.right.mean_baseline, 2),
                    "right_std_base":  round(self.right.std_baseline, 2),
                    "frame_luma": round(frame_luma, 2),
                    "left_state": self.left.state,
                    "right_state": self.right.state,
                    "paused": paused,
                })

            frame_idx += 1
            if total_frames > 0 and frame_idx % progress_step == 0:
                pct = frame_idx / total_frames * 100
                elapsed = time.time() - start
                fps_proc = frame_idx / elapsed if elapsed > 0 else 0
                eta = (total_frames - frame_idx) / fps_proc if fps_proc > 0 else 0
                print(f"  {pct:5.1f}% | L={self.left_count():3d} R={self.right_count():3d} "
                      f"| {fps_proc:.0f} fps | ETA {eta:.0f}s")

        cap.release()
        elapsed = time.time() - start
        return self._build_results(frame_idx, duration_sec, elapsed)

    def left_count(self):
        return sum(1 for e in self.events if e.get("table") == "left")

    def right_count(self):
        return sum(1 for e in self.events if e.get("table") == "right")

    def _build_results(self, total_frames, duration_sec, elapsed):
        L = [e for e in self.events if e.get("table") == "left"]
        R = [e for e in self.events if e.get("table") == "right"]
        all_durs = [e["cycle_duration_sec"] for e in self.events]
        mean_dur = float(np.mean(all_durs)) if all_durs else 0.0
        med_dur = float(np.median(all_durs)) if all_durs else 0.0
        balance = (min(len(L), len(R)) / max(len(L), len(R))) if max(len(L), len(R)) > 0 else 1.0
        overlap_count = sum(1 for e in self.events if e.get("via_overlap_detector"))
        long_count = sum(1 for e in self.events if e.get("long_cycle"))
        lighting_pauses = sum(1 for b in self.breaks if b["type"] == "lighting_change")

        print()
        print("=" * 60)
        print(f"  CH27 RESULTS")
        print("=" * 60)
        print(f"  Total cycles:      {len(self.events)}")
        print(f"  LEFT  cycles:      {len(L)}")
        print(f"  RIGHT cycles:      {len(R)}")
        print(f"  Mean duration:     {mean_dur:.1f}s   Median: {med_dur:.1f}s")
        print(f"  Balance ratio:     {balance:.2f}  (min/max)")
        print(f"  Via overlap det:   {overlap_count}")
        print(f"  Long cycles:       {long_count}")
        print(f"  Lighting pauses:   {lighting_pauses}")
        print(f"  Suppressed:        {len(self.suppressed)}")
        if self.suppressed:
            from collections import Counter
            reasons = Counter(s["reason"] for s in self.suppressed)
            for r, n in reasons.most_common():
                print(f"    {r}: {n}")
        print(f"  Processing:        {total_frames} frames in {elapsed:.1f}s "
              f"({total_frames/elapsed:.0f} fps, {total_frames/self.fps/elapsed:.1f}x realtime)")

        return {
            "metadata": {
                "camera": "CH27",
                "source": self.source,
                "fps": self.fps,
                "total_frames": total_frames,
                "duration_sec": round(duration_sec, 2),
                "processing_time_sec": round(elapsed, 2),
                "version": self.version,
            },
            "config": {
                "left_table_roi":  list(self.left_roi),
                "right_table_roi": list(self.right_roi),
                "heap_roi":        list(self.heap_roi) if self.heap_roi else None,
                **self.tracker_cfg,
                "lighting_delta": self.lighting_delta,
                "lighting_restore_delta": self.lighting_restore_delta,
                "lighting_restore_frames": self.lighting_restore_frames,
            },
            "summary": {
                "total_cycles": len(self.events),
                "left_cycles":  len(L),
                "right_cycles": len(R),
                "mean_cycle_sec":   round(mean_dur, 2),
                "median_cycle_sec": round(med_dur, 2),
                "table_balance_ratio": round(balance, 3),
                "overlap_cycles": overlap_count,
                "long_cycles": long_count,
                "lighting_pauses": lighting_pauses,
                "suppressed_count": len(self.suppressed),
            },
            "events": self.events,
            "breaks": self.breaks,
            "suppressed_candidates": self.suppressed,
            "frame_data": self.frame_data,
            "heap_trace": self.heap_trace,
        }


# ═══════════════════════════════════════════════════════════════
#  CLI
# ═══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="CH27 Taping Counter (v1)")
    parser.add_argument("video", help="Path to video file")
    parser.add_argument("--output", "-o", help="Output JSON path")
    parser.add_argument("--debug", "-d", action="store_true",
                        help="Verbose per-event log")
    parser.add_argument("--version", default="v1", choices=["v1"],
                        help="Algorithm variant (currently only v1)")
    parser.add_argument("--frame-step", type=int, default=1,
                        help="Process every Nth frame (default 1). Use 2 to halve "
                             "HEVC decode time on long files; cycle detection is "
                             "still time-based so behavior is preserved.")
    args = parser.parse_args()

    counter = TapingCounter(args.video, debug=args.debug, version=args.version,
                            frame_step=args.frame_step)
    results = counter.run()

    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\n  Results saved to {args.output}")


if __name__ == "__main__":
    main()
