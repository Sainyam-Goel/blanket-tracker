"""
CH19 Cutting Counter — Counts blanket cuts from NVR camera feed.

Detection approach (v5 / v6): Multi-scale derivative + multi-ROI + close-pair merge.

  v6-permissive (pass --version v6): aggressive-recall variant that addresses
  suspected undercounting in 2-worker scissor phase and bright afternoon lighting.
  Changes from v5:
    - Dual-ROI detection: cut fires if EITHER table_roi OR left_roi crosses
      threshold (not just table_roi). Each event tagged with roi_source.
    - Close-pair merge REMOVED: close pairs are flagged (close_pair_suspect=True)
      but both kept. User can filter downstream if desired.
    - Echo suppression relaxed: ratio 0.6→0.4, window 3.0s→1.2s.
    - Adaptive break threshold: rolling-baseline + 50 (not fixed 235), hold 4s.
    - DERIV_THRESHOLD_SHORT: 10.0 → 8.0 (more margin for weak 2-worker signals).
    - Suppression audit log: every candidate dropped by any filter is logged
      to `suppressed_candidates` with reason, for downstream review.

  When a cut piece slides off the table, the white surface is exposed,
  causing a rapid brightness INCREASE in the table ROI. We detect these
  positive derivative spikes as cut events.

  Multi-scale detection:
  - TWO derivative windows run in parallel:
    * d_long (35 frames / 1.4s) — catches strong 4-worker cuts
    * d_short (25 frames / 1.0s) — catches rapid 2-worker scissor cuts
  - EITHER window crossing its threshold triggers a cut detection

  Robustness guardrails (v5):
  - Echo suppression: removes weak detections following a stronger event
  - Close-pair merge: when two events fire within 2.5s and at least one
    has strong deriv (>30), they're the same physical event — keep the
    stronger detection. 2-worker consecutive cuts (both weak, deriv <25)
    pass through unaffected.
  - Multi-ROI tracking: primary (right table), secondary (left table),
    slide (below table edge). Left-ROI derivative and spatial uniformity
    (brightness std) are stored as metadata for confidence scoring.
  - Brightness ceiling: detections with very high brightness (>230) during
    non-break periods are flagged (likely break transitions, not real cuts).

  Confidence scoring uses: peak_deriv, spike_duration, slide_motion,
  spatial_std, and left-ROI agreement to produce high/medium/low rating.

This approach is color-agnostic: it detects CHANGE rather than absolute
brightness, so it works regardless of blanket color.

Break detection: when brightness > 235 for extended periods, the table
is empty (idle). Cut detection is suppressed during breaks.

Usage:
  python3 cutting_counter.py /path/to/video.mp4 [--output results.json] [--debug]
"""

import cv2
import numpy as np
import json
import sys
import time
from collections import deque

# ═══════════════════════════════════════════════════════════════
#  CONFIG
# ═══════════════════════════════════════════════════════════════

# ── ROIs ──
# Primary table ROI — right portion of white table surface
TABLE_ROI = (820, 240, 1020, 360)
# Secondary table ROI — left portion (cross-validation)
LEFT_TABLE_ROI = (600, 240, 820, 360)
# Slide ROI — below table front edge (motion confirmation metadata)
SLIDE_ROI = (720, 370, 960, 520)

# ── Smoothing ──
SMOOTH_WINDOW = 7           # frames for brightness running average (~0.28s at 25fps)

# ── Multi-scale derivative detection ──
DERIV_WINDOW_LONG = 35      # long window (~1.4s) — strong 4-worker cuts
DERIV_WINDOW_SHORT = 25     # short window (~1.0s) — fast 2-worker scissor cuts
DERIV_THRESHOLD_LONG = 18.0   # threshold for long window
DERIV_THRESHOLD_SHORT = 10.0  # threshold for short window (catches weak 2-worker signals)
DERIV_STRONG_THRESHOLD = 30.0  # strong spike — needs fewer confirm frames
DERIV_CONFIRM_FRAMES = 2    # derivative must stay above threshold for N frames
DERIV_CONFIRM_STRONG = 1    # confirm frames needed for strong spikes (>30)

# ── Gap filter ──
MIN_CYCLE_FRAMES = 50       # min 2.0s gap between cuts (2-worker cuts can be 2s apart)

# ── Echo suppression — removes "bounce" detections after strong events ──
ECHO_WINDOW_SEC = 3.0       # look back this far for stronger preceding events
ECHO_RATIO = 0.6            # suppress if deriv < 60% of preceding event's deriv

# ── Close-pair merge — removes double-detections of same physical event ──
CLOSE_PAIR_GAP = 2.5        # max gap (seconds) to consider events as same physical event
CLOSE_PAIR_DERIV = 30.0     # merge only if at least one event has deriv above this
                             # (2-worker consecutive cuts have deriv <25, so they pass through)

# ── Brightness ceiling — flag high-brightness detections ──
BRIGHTNESS_CEILING = 230    # above this, flag as possible break transition

# ── Break detection ──
BREAK_BRIGHTNESS = 235      # brightness above this = empty table
BREAK_HOLD_FRAMES = 75      # must sustain for 3s to confirm break
BREAK_EXIT_FRAMES = 25      # must drop below for 1s to exit break

# ── Output ──
FRAME_SAMPLE_RATE = 25      # store frame_data every N frames (1 per second)

# ═══════════════════════════════════════════════════════════════
#  V6-PERMISSIVE CONFIG OVERRIDES (aggressive recall)
# ═══════════════════════════════════════════════════════════════
V6_CONFIG = {
    "deriv_threshold_short": 8.0,       # was 10.0 — more margin for 2-worker
    "echo_window_sec": 1.2,             # was 3.0 — only catch true bounces
    "echo_ratio": 0.4,                  # was 0.6 — only suppress much weaker echoes
    "close_pair_gap": 2.5,              # same — used ONLY for flagging now
    "close_pair_deriv": 30.0,           # same — flag threshold
    "break_hold_frames": 100,           # was 75 — need 4s sustained (not 3s)
    "break_exit_frames": 38,            # was 25 — need 1.5s below (not 1s)
    "break_offset_above_baseline": 50,  # new: adaptive threshold = baseline + 50
    "baseline_window_frames": 1500,     # new: 60s rolling baseline (at 25fps)
    "dual_roi": True,                   # new: cut fires if EITHER ROI triggers
    "merge_close_pairs": False,         # new: v6 does NOT merge, only flags
    "audit_suppressions": True,         # new: log every dropped candidate
}


# ═══════════════════════════════════════════════════════════════
#  CUTTING COUNTER
# ═══════════════════════════════════════════════════════════════

class CuttingCounter:
    def __init__(self, video_path, **config):
        self.video_path = video_path

        # Version selection (v5 = default/legacy, v6 = permissive/aggressive-recall)
        self.version = config.get("version", "v5")
        if self.version == "v6":
            # v6 overrides only apply if caller didn't explicitly pass them
            for k, v in V6_CONFIG.items():
                config.setdefault(k, v)

        # Config overrides
        self.table_roi = config.get("table_roi", TABLE_ROI)
        self.left_table_roi = config.get("left_table_roi", LEFT_TABLE_ROI)
        self.slide_roi = config.get("slide_roi", SLIDE_ROI)
        self.smooth_window = config.get("smooth_window", SMOOTH_WINDOW)
        self.deriv_window_long = config.get("deriv_window_long", DERIV_WINDOW_LONG)
        self.deriv_window_short = config.get("deriv_window_short", DERIV_WINDOW_SHORT)
        self.deriv_threshold_long = config.get("deriv_threshold_long", DERIV_THRESHOLD_LONG)
        self.deriv_threshold_short = config.get("deriv_threshold_short", DERIV_THRESHOLD_SHORT)
        self.deriv_strong_threshold = config.get("deriv_strong_threshold", DERIV_STRONG_THRESHOLD)
        self.deriv_confirm = config.get("deriv_confirm", DERIV_CONFIRM_FRAMES)
        self.deriv_confirm_strong = config.get("deriv_confirm_strong", DERIV_CONFIRM_STRONG)
        self.min_cycle = config.get("min_cycle", MIN_CYCLE_FRAMES)
        self.break_brightness = config.get("break_brightness", BREAK_BRIGHTNESS)
        self.break_hold = config.get("break_hold_frames", config.get("break_hold", BREAK_HOLD_FRAMES))
        self.break_exit = config.get("break_exit_frames", config.get("break_exit", BREAK_EXIT_FRAMES))
        # v6 adaptive break threshold
        self.break_offset = config.get("break_offset_above_baseline", 0)  # 0 = disabled (v5)
        self.baseline_window = config.get("baseline_window_frames", 1500)
        # v6 dual-ROI + merge + audit toggles
        self.dual_roi = config.get("dual_roi", False)
        self.merge_close_pairs_enabled = config.get("merge_close_pairs", True)
        self.audit_suppressions = config.get("audit_suppressions", False)
        self.echo_window_sec = config.get("echo_window_sec", ECHO_WINDOW_SEC)
        self.echo_ratio = config.get("echo_ratio", ECHO_RATIO)
        self.close_pair_gap = config.get("close_pair_gap", CLOSE_PAIR_GAP)
        self.close_pair_deriv = config.get("close_pair_deriv", CLOSE_PAIR_DERIV)
        self.brightness_ceiling = config.get("brightness_ceiling", BRIGHTNESS_CEILING)
        self.sample_rate = config.get("sample_rate", FRAME_SAMPLE_RATE)
        self.debug = config.get("debug", False)

        # Brightness tracking — buffer must hold longest derivative window
        self.brightness_buffer = deque(maxlen=self.smooth_window)
        max_window = max(self.deriv_window_long, self.deriv_window_short)
        self.smoothed_history = deque(maxlen=max_window + 10)

        # Left-table ROI tracking (secondary signal for cross-validation)
        self.left_brightness_buffer = deque(maxlen=self.smooth_window)
        self.left_smoothed_history = deque(maxlen=max_window + 10)

        # Derivative spike detection state
        self.last_cut_frame = -self.min_cycle
        self.deriv_confirm_counter = 0
        self.spike_detected = False
        self.spike_start_frame = None
        self.spike_peak_deriv = 0
        self.spike_peak_brightness = 0

        # Break detection state
        self.in_break = False
        self.break_counter = 0      # frames above break brightness
        self.break_exit_counter = 0  # frames below break brightness (for exit)

        # Slide motion
        self.prev_slide_region = None
        self.slide_motion_peak = 0

        # Multi-ROI tracking for current spike
        self.spike_left_deriv_peak = 0
        self.spike_spatial_std_at_peak = 0
        self.spike_roi_source = None  # "table" | "left" | "both" (v6)

        # v6 rolling baseline for adaptive break threshold
        self.baseline_buffer = deque(maxlen=self.baseline_window)

        # Results
        self.cuts = []
        self.breaks = []
        self.frame_data = []
        self.suppressed_candidates = []  # v6 audit log
        self.fps = 25.0

    def _get_smoothed(self, raw):
        """Add raw to buffer, return running average."""
        self.brightness_buffer.append(raw)
        smoothed = float(np.mean(self.brightness_buffer))
        self.smoothed_history.append(smoothed)
        return smoothed

    def _get_left_smoothed(self, raw):
        """Smoothed brightness for left-table ROI."""
        self.left_brightness_buffer.append(raw)
        smoothed = float(np.mean(self.left_brightness_buffer))
        self.left_smoothed_history.append(smoothed)
        return smoothed

    def _get_left_derivative(self):
        """Derivative of left-table ROI brightness (d_short window)."""
        if len(self.left_smoothed_history) >= self.deriv_window_short:
            return self.left_smoothed_history[-1] - self.left_smoothed_history[-self.deriv_window_short]
        return 0.0

    def _get_derivatives(self):
        """Compute brightness change at both derivative windows.

        Returns (d_long, d_short) — the change over each window.
        Either can trigger a cut detection if it exceeds its threshold.
        """
        current = self.smoothed_history[-1] if self.smoothed_history else 0
        d_long = 0.0
        d_short = 0.0
        if len(self.smoothed_history) >= self.deriv_window_long:
            d_long = current - self.smoothed_history[-self.deriv_window_long]
        if len(self.smoothed_history) >= self.deriv_window_short:
            d_short = current - self.smoothed_history[-self.deriv_window_short]
        return d_long, d_short

    def _check_threshold(self, d_long, d_short, left_deriv=0.0):
        """Check if EITHER derivative window crosses its threshold.

        Returns (passes, effective_deriv, roi_source) where:
          - effective_deriv is the best derivative value for peak tracking
          - roi_source is "table" | "left" | "both" (v6 dual-ROI) or None (v5)
        """
        long_pass = d_long >= self.deriv_threshold_long
        short_pass = d_short >= self.deriv_threshold_short
        table_pass = long_pass or short_pass

        effective = max(d_long, d_short)

        if not self.dual_roi:
            return table_pass, effective, ("table" if table_pass else None)

        # v6 dual-ROI: left ROI uses the SHORT threshold (same semantics)
        left_pass = left_deriv >= self.deriv_threshold_short

        if table_pass and left_pass:
            source = "both"
        elif table_pass:
            source = "table"
        elif left_pass:
            source = "left"
        else:
            source = None

        if left_pass and left_deriv > effective:
            effective = left_deriv

        return (table_pass or left_pass), effective, source

    def _compute_slide_motion(self, gray):
        """Frame-to-frame diff in slide ROI."""
        sx1, sy1, sx2, sy2 = self.slide_roi
        slide_region = gray[sy1:sy2, sx1:sx2].astype(np.float32)
        motion = 0.0
        if self.prev_slide_region is not None:
            motion = float(np.mean(np.abs(slide_region - self.prev_slide_region)))
        self.prev_slide_region = slide_region
        return motion

    def _current_break_threshold(self):
        """v5: fixed self.break_brightness. v6: rolling-baseline + offset (robust)."""
        if self.break_offset <= 0 or len(self.baseline_buffer) < 100:
            return self.break_brightness
        # Use median for robustness against spikes
        baseline = float(np.median(self.baseline_buffer))
        adaptive = baseline + self.break_offset
        # Never go below the v5 safety floor (avoids ridiculously low thresholds early)
        return max(adaptive, self.break_brightness - 20)

    def _update_break_state(self, smoothed, frame_idx):
        """Detect break periods (empty table). v6 uses adaptive threshold."""
        # v6: maintain rolling baseline (only when not in a spike and not in break)
        if self.break_offset > 0 and not self.in_break and not self.spike_detected:
            self.baseline_buffer.append(smoothed)

        threshold = self._current_break_threshold()

        if not self.in_break:
            if smoothed > threshold:
                self.break_counter += 1
                if self.break_counter >= self.break_hold:
                    self.in_break = True
                    self.break_exit_counter = 0
                    break_start = frame_idx - self.break_hold
                    if self.debug:
                        t = frame_idx / self.fps
                        print(f"  *** BREAK START at {t:.1f}s (brightness={smoothed:.1f})")
                    self.breaks.append({
                        "type": "break_start",
                        "frame": break_start,
                        "time_sec": round(break_start / self.fps, 2),
                    })
            else:
                self.break_counter = 0
        else:
            if smoothed < threshold:
                self.break_exit_counter += 1
                if self.break_exit_counter >= self.break_exit:
                    self.in_break = False
                    self.break_counter = 0
                    if self.debug:
                        t = frame_idx / self.fps
                        print(f"  *** BREAK END at {t:.1f}s (brightness={smoothed:.1f})")
                    self.breaks.append({
                        "type": "break_end",
                        "frame": frame_idx,
                        "time_sec": round(frame_idx / self.fps, 2),
                    })
            else:
                self.break_exit_counter = 0

    def process_frame(self, frame_idx, gray):
        """Process one grayscale frame. Returns event dict or None."""
        tx1, ty1, tx2, ty2 = self.table_roi

        # 1. Table brightness (primary ROI)
        table_region = gray[ty1:ty2, tx1:tx2]
        raw_brightness = float(np.mean(table_region))
        spatial_std = float(np.std(table_region))
        smoothed = self._get_smoothed(raw_brightness)

        # 1b. Left-table brightness (secondary ROI for cross-validation)
        lx1, ly1, lx2, ly2 = self.left_table_roi
        left_brightness = float(np.mean(gray[ly1:ly2, lx1:lx2]))
        left_smoothed = self._get_left_smoothed(left_brightness)
        left_deriv = self._get_left_derivative()

        # 2. Multi-scale derivatives
        d_long, d_short = self._get_derivatives()
        passes_threshold, effective_deriv, roi_source = self._check_threshold(
            d_long, d_short, left_deriv)

        # 3. Slide motion
        slide_motion = self._compute_slide_motion(gray)

        # 4. Break detection
        self._update_break_state(smoothed, frame_idx)

        # 5. Multi-scale derivative spike detection (suppressed during breaks)
        #    A cut is detected when EITHER derivative window crosses its threshold.
        #    Adaptive confirmation: strong spikes need fewer frames to confirm.
        event = None
        if not self.in_break:
            if passes_threshold:
                if not self.spike_detected:
                    # New spike starting — track max deriv during confirmation
                    self.deriv_confirm_counter += 1
                    self.spike_peak_deriv_candidate = max(
                        getattr(self, 'spike_peak_deriv_candidate', 0),
                        effective_deriv)

                    # Adaptive confirmation: strong spikes confirm faster
                    is_strong = self.spike_peak_deriv_candidate >= self.deriv_strong_threshold
                    needed = self.deriv_confirm_strong if is_strong else self.deriv_confirm

                    if self.deriv_confirm_counter >= needed:
                        # Check min gap
                        if (frame_idx - self.last_cut_frame) >= self.min_cycle:
                            self.spike_detected = True
                            self.spike_start_frame = frame_idx
                            self.spike_peak_deriv = self.spike_peak_deriv_candidate
                            self.spike_peak_brightness = smoothed
                            self.slide_motion_peak = slide_motion
                            self.spike_left_deriv_peak = left_deriv
                            self.spike_spatial_std_at_peak = spatial_std
                            self.spike_roi_source = roi_source
                            self.spike_duration = 0
                        else:
                            # Too close to last cut — reset + audit log
                            if self.audit_suppressions:
                                self.suppressed_candidates.append({
                                    "time_sec": round(frame_idx / self.fps, 2),
                                    "peak_deriv": round(self.spike_peak_deriv_candidate, 1),
                                    "peak_brightness": round(smoothed, 1),
                                    "roi_source": roi_source,
                                    "dropped_by": "min_gap",
                                    "preceding_cut_time": round(self.last_cut_frame / self.fps, 2),
                                })
                            self.deriv_confirm_counter = 0
                            self.spike_peak_deriv_candidate = 0
                else:
                    # Spike ongoing — track peak values
                    self.spike_duration += 1
                    if effective_deriv > self.spike_peak_deriv:
                        self.spike_peak_deriv = effective_deriv
                        self.spike_spatial_std_at_peak = spatial_std
                        # Upgrade roi_source if stronger signal comes from different ROI
                        if roi_source == "both":
                            self.spike_roi_source = "both"
                        elif self.spike_roi_source and roi_source and roi_source != self.spike_roi_source:
                            self.spike_roi_source = "both"
                    if smoothed > self.spike_peak_brightness:
                        self.spike_peak_brightness = smoothed
                    self.slide_motion_peak = max(self.slide_motion_peak, slide_motion)
                    self.spike_left_deriv_peak = max(self.spike_left_deriv_peak, left_deriv)
            else:
                if self.spike_detected:
                    # Spike ended — emit cut event with confidence score
                    self.spike_detected = False
                    self.last_cut_frame = self.spike_start_frame
                    t = self.spike_start_frame / self.fps

                    # Brightness ceiling flag
                    ceiling_flag = self.spike_peak_brightness > self.brightness_ceiling

                    # Confidence scoring based on multi-signal strength
                    confidence = self._compute_confidence(
                        self.spike_peak_deriv, self.spike_duration,
                        self.slide_motion_peak, self.spike_left_deriv_peak,
                        self.spike_spatial_std_at_peak, ceiling_flag)

                    cut_event = {
                        "type": "cut",
                        "frame": self.spike_start_frame,
                        "time_sec": round(t, 2),
                        "peak_deriv": round(self.spike_peak_deriv, 1),
                        "peak_brightness": round(self.spike_peak_brightness, 1),
                        "slide_motion_peak": round(self.slide_motion_peak, 1),
                        "spike_duration": self.spike_duration,
                        "left_deriv": round(self.spike_left_deriv_peak, 1),
                        "spatial_std": round(self.spike_spatial_std_at_peak, 1),
                        "ceiling_flag": ceiling_flag,
                        "roi_source": self.spike_roi_source or "table",
                        "confidence": confidence,
                    }
                    self.cuts.append(cut_event)
                    event = cut_event

                    if self.debug:
                        print(f"  >>> CUT #{len(self.cuts)} at {t:.1f}s "
                              f"(deriv={self.spike_peak_deriv:.1f}, "
                              f"bright={self.spike_peak_brightness:.1f}, "
                              f"slide={self.slide_motion_peak:.1f}, "
                              f"left_d={self.spike_left_deriv_peak:.1f}, "
                              f"std={self.spike_spatial_std_at_peak:.1f}, "
                              f"dur={self.spike_duration}, conf={confidence})")

                    self.spike_start_frame = None
                    self.spike_peak_deriv = 0
                    self.spike_peak_brightness = 0
                    self.slide_motion_peak = 0
                    self.spike_left_deriv_peak = 0
                    self.spike_spatial_std_at_peak = 0
                    self.spike_roi_source = None
                    self.spike_duration = 0

                self.deriv_confirm_counter = 0
                self.spike_peak_deriv_candidate = 0
        else:
            # Reset spike state during breaks — audit log any in-flight candidate
            if self.audit_suppressions and self.spike_detected and self.spike_start_frame is not None:
                self.suppressed_candidates.append({
                    "time_sec": round(self.spike_start_frame / self.fps, 2),
                    "peak_deriv": round(self.spike_peak_deriv, 1),
                    "peak_brightness": round(self.spike_peak_brightness, 1),
                    "roi_source": self.spike_roi_source,
                    "dropped_by": "break",
                    "preceding_cut_time": round(self.last_cut_frame / self.fps, 2),
                })
            self.spike_detected = False
            self.deriv_confirm_counter = 0
            self.spike_peak_deriv_candidate = 0
            self.spike_roi_source = None

        # 6. Store frame data (sampled)
        if frame_idx % self.sample_rate == 0:
            self.frame_data.append({
                "frame": frame_idx,
                "time_sec": round(frame_idx / self.fps, 2),
                "brightness": round(raw_brightness, 1),
                "smoothed": round(smoothed, 1),
                "derivative": round(d_long, 1),
                "deriv_short": round(d_short, 1),
                "left_brightness": round(left_brightness, 1),
                "left_deriv": round(left_deriv, 1),
                "spatial_std": round(spatial_std, 1),
                "slide_motion": round(slide_motion, 1),
                "in_break": self.in_break,
            })

        # Debug output (every 1s)
        if self.debug and frame_idx % 25 == 0:
            t = frame_idx / self.fps
            state = "BREAK" if self.in_break else ("SPIKE" if self.spike_detected else "active")
            print(f"  {t:7.1f}s | bright={smoothed:6.1f} | dL={d_long:+6.1f} dS={d_short:+6.1f} "
                  f"| slide={slide_motion:5.1f} | {state}")

        return event

    def run(self):
        """Process entire video and return results."""
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            print(f"ERROR: Cannot open {self.video_path}")
            sys.exit(1)

        self.fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        variant = "v6-permissive (dual-ROI, adaptive break, flag-only close-pairs)" \
            if self.version == "v6" else "v5-robust (multi-scale + multi-ROI + close-pair merge)"
        print(f"CH19 Cutting Counter {variant}")
        print(f"  Video: {self.video_path}")
        print(f"  FPS: {self.fps:.1f}, Frames: {total_frames}, "
              f"Duration: {total_frames/self.fps:.1f}s ({total_frames/self.fps/60:.1f} min)")
        print(f"  Table ROI: {self.table_roi}")
        print(f"  Deriv LONG:  {self.deriv_window_long} frames ({self.deriv_window_long/self.fps:.2f}s), "
              f"threshold={self.deriv_threshold_long}")
        print(f"  Deriv SHORT: {self.deriv_window_short} frames ({self.deriv_window_short/self.fps:.2f}s), "
              f"threshold={self.deriv_threshold_short}")
        print(f"  Strong spike: >{self.deriv_strong_threshold} (confirm {self.deriv_confirm_strong}f), "
              f"normal confirm: {self.deriv_confirm}f")
        print(f"  Min gap: {self.min_cycle} frames ({self.min_cycle/self.fps:.1f}s)")
        print(f"  Smoothing: {self.smooth_window} frames ({self.smooth_window/self.fps:.2f}s)")
        print(f"  Break: brightness>{self.break_brightness} for {self.break_hold} frames")
        print(f"  Echo suppression: window={self.echo_window_sec}s, ratio={self.echo_ratio}")
        print(f"  Close-pair merge: gap<={self.close_pair_gap}s, deriv>{self.close_pair_deriv}")
        print(f"  Left-table ROI: {self.left_table_roi}")
        print(f"  Brightness ceiling: {self.brightness_ceiling}")
        if self.version == "v6":
            print(f"  [v6] Dual-ROI detection: ENABLED (OR-gated)")
            print(f"  [v6] Close-pair merge: DISABLED (flag-only)")
            print(f"  [v6] Adaptive break threshold: baseline + {self.break_offset} "
                  f"(min {self.break_brightness - 20})")
            print(f"  [v6] Audit suppression log: ENABLED")
        print()

        start_time = time.time()
        frame_idx = 0
        progress_interval = max(1, total_frames // 20)  # 5% progress updates

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            self.process_frame(frame_idx, gray)
            frame_idx += 1

            # Progress
            if not self.debug and frame_idx % progress_interval == 0:
                pct = frame_idx / total_frames * 100
                elapsed = time.time() - start_time
                rate = frame_idx / elapsed
                eta = (total_frames - frame_idx) / rate
                print(f"  {pct:5.1f}% | {len(self.cuts)} cuts | "
                      f"{rate:.0f} fps | ETA {eta:.0f}s", end="\r")

        cap.release()
        elapsed = time.time() - start_time
        duration_sec = frame_idx / self.fps

        # Close any pending spike
        if self.spike_detected and self.spike_start_frame is not None:
            t = self.spike_start_frame / self.fps
            self.cuts.append({
                "type": "cut",
                "frame": self.spike_start_frame,
                "time_sec": round(t, 2),
                "peak_deriv": round(self.spike_peak_deriv, 1),
                "peak_brightness": round(self.spike_peak_brightness, 1),
                "slide_motion_peak": round(self.slide_motion_peak, 1),
                "spike_duration": 0,
                "left_deriv": round(self.spike_left_deriv_peak, 1),
                "spatial_std": round(self.spike_spatial_std_at_peak, 1),
                "ceiling_flag": self.spike_peak_brightness > self.brightness_ceiling,
                "confidence": "low",
            })

        # Post-processing filters
        self._suppress_echoes()
        self._merge_close_pairs()

        # Summary stats
        total_cuts = len(self.cuts)
        # Only count active time (exclude breaks) for rate calculation
        break_time = self._compute_break_time()
        active_time = duration_sec - break_time
        cuts_per_min = (total_cuts / active_time * 60) if active_time > 0 else 0

        cycle_times = []
        for i in range(1, len(self.cuts)):
            cycle_times.append(self.cuts[i]["time_sec"] - self.cuts[i - 1]["time_sec"])
        avg_cycle = float(np.mean(cycle_times)) if cycle_times else 0

        print(f"\n{'='*60}")
        print(f"  RESULTS")
        print(f"{'='*60}")
        print(f"  Total cuts detected:  {total_cuts}")
        print(f"  Active cutting time:  {active_time:.0f}s ({active_time/60:.1f} min)")
        print(f"  Break time:           {break_time:.0f}s ({break_time/60:.1f} min)")
        print(f"  Rate (active time):   {cuts_per_min:.1f} cuts/min")
        if avg_cycle > 0:
            print(f"  Avg cycle time:       {avg_cycle:.1f}s")
        print(f"  Processing:           {frame_idx} frames in {elapsed:.1f}s "
              f"({frame_idx/elapsed:.0f} fps, {frame_idx/elapsed/self.fps:.1f}x realtime)")

        # Print first/last 10 cuts
        # Confidence distribution
        if total_cuts > 0:
            high_conf = sum(1 for c in self.cuts if c.get("confidence") == "high")
            med_conf = sum(1 for c in self.cuts if c.get("confidence") == "medium")
            low_conf = sum(1 for c in self.cuts if c.get("confidence") == "low")
            print(f"  Confidence:           {high_conf} high, {med_conf} medium, {low_conf} low")

            if self.version == "v6":
                roi_table = sum(1 for c in self.cuts if c.get("roi_source") == "table")
                roi_left = sum(1 for c in self.cuts if c.get("roi_source") == "left")
                roi_both = sum(1 for c in self.cuts if c.get("roi_source") == "both")
                flagged = sum(1 for c in self.cuts if c.get("close_pair_suspect"))
                print(f"  ROI source:           {roi_table} table, {roi_left} left, {roi_both} both")
                print(f"  Close-pair suspects:  {flagged} events (not dropped)")
                print(f"  Suppressed candidates: {len(self.suppressed_candidates)} "
                      f"(audit log)")

            print(f"\n  First 10 cuts:")
            for i, cut in enumerate(self.cuts[:10]):
                mins = int(cut['time_sec'] // 60)
                secs = cut['time_sec'] % 60
                conf = cut.get('confidence', '?')
                print(f"    #{i+1:3d}: {mins:2d}:{secs:05.2f} "
                      f"(deriv={cut['peak_deriv']:+.1f}, bright={cut['peak_brightness']:.0f}, "
                      f"conf={conf})")
            if total_cuts > 20:
                print(f"  ... ({total_cuts - 20} more) ...")
                print(f"  Last 10 cuts:")
                for i, cut in enumerate(self.cuts[-10:]):
                    idx = total_cuts - 10 + i
                    mins = int(cut['time_sec'] // 60)
                    secs = cut['time_sec'] % 60
                    conf = cut.get('confidence', '?')
                    print(f"    #{idx+1:3d}: {mins:2d}:{secs:05.2f} "
                          f"(deriv={cut['peak_deriv']:+.1f}, bright={cut['peak_brightness']:.0f}, "
                          f"conf={conf})")

        # Break periods
        if self.breaks:
            print(f"\n  Break periods detected:")
            i = 0
            while i < len(self.breaks):
                if self.breaks[i]["type"] == "break_start":
                    start_t = self.breaks[i]["time_sec"]
                    end_t = duration_sec
                    if i + 1 < len(self.breaks) and self.breaks[i+1]["type"] == "break_end":
                        end_t = self.breaks[i+1]["time_sec"]
                        i += 1
                    dur = end_t - start_t
                    s_min = int(start_t // 60)
                    s_sec = start_t % 60
                    e_min = int(end_t // 60)
                    e_sec = end_t % 60
                    print(f"    {s_min:2d}:{s_sec:04.1f} - {e_min:2d}:{e_sec:04.1f} ({dur:.0f}s)")
                i += 1

        results = {
            "metadata": {
                "video": self.video_path,
                "fps": self.fps,
                "duration_sec": round(duration_sec, 2),
                "total_frames": frame_idx,
                "processing_time_sec": round(elapsed, 2),
                "version": "v6-permissive" if self.version == "v6" else "v5-robust",
            },
            "config": {
                "table_roi": list(self.table_roi),
                "left_table_roi": list(self.left_table_roi),
                "slide_roi": list(self.slide_roi),
                "smooth_window": self.smooth_window,
                "deriv_window_long": self.deriv_window_long,
                "deriv_window_short": self.deriv_window_short,
                "deriv_threshold_long": self.deriv_threshold_long,
                "deriv_threshold_short": self.deriv_threshold_short,
                "deriv_strong_threshold": self.deriv_strong_threshold,
                "deriv_confirm": self.deriv_confirm,
                "deriv_confirm_strong": self.deriv_confirm_strong,
                "min_cycle_frames": self.min_cycle,
                "echo_window_sec": self.echo_window_sec,
                "echo_ratio": self.echo_ratio,
                "close_pair_gap": self.close_pair_gap,
                "close_pair_deriv": self.close_pair_deriv,
                "brightness_ceiling": self.brightness_ceiling,
                "break_brightness": self.break_brightness,
            },
            "summary": {
                "total_cuts": total_cuts,
                "active_time_sec": round(active_time, 1),
                "break_time_sec": round(break_time, 1),
                "cuts_per_minute": round(cuts_per_min, 1),
                "avg_cycle_sec": round(avg_cycle, 1),
            },
            "events": self.cuts,
            "breaks": self.breaks,
            "frame_data": self.frame_data,
            "suppressed_candidates": self.suppressed_candidates,
        }

        return results

    def _suppress_echoes(self):
        """Post-processing: remove echo detections that follow a stronger event.

        An "echo" is a weaker detection within ECHO_WINDOW_SEC of a stronger
        preceding detection. In the 4-worker phase, strong cuts create
        derivative bounces that the sensitive short-window detector picks up.
        Real consecutive cuts (2-worker phase) have similar strengths, so
        they aren't suppressed.

        Rule: Remove event[i] if there exists event[j] (j < i) where:
          - time gap < echo_window_sec
          - event[j].peak_deriv > event[i].peak_deriv
          - event[i].peak_deriv < echo_ratio * event[j].peak_deriv
        """
        if not self.cuts or self.echo_ratio >= 1.0:
            return

        original_count = len(self.cuts)
        filtered = []

        for i, cut in enumerate(self.cuts):
            is_echo = False
            for j in range(i - 1, -1, -1):
                prev = self.cuts[j]
                gap = cut["time_sec"] - prev["time_sec"]
                if gap > self.echo_window_sec:
                    break
                if (prev["peak_deriv"] > cut["peak_deriv"] and
                        cut["peak_deriv"] < self.echo_ratio * prev["peak_deriv"]):
                    is_echo = True
                    break

            if not is_echo:
                filtered.append(cut)
            else:
                if self.audit_suppressions:
                    self.suppressed_candidates.append({
                        "time_sec": cut["time_sec"],
                        "peak_deriv": cut["peak_deriv"],
                        "peak_brightness": cut.get("peak_brightness", 0),
                        "roi_source": cut.get("roi_source"),
                        "dropped_by": "echo",
                        "preceding_cut_time": self.cuts[j]["time_sec"],
                        "preceding_cut_deriv": self.cuts[j]["peak_deriv"],
                    })
                if self.debug:
                    t = cut["time_sec"]
                    print(f"  ~~~ ECHO suppressed at {int(t)//60}:{t%60:05.2f} "
                          f"(deriv={cut['peak_deriv']:.1f}, prev={self.cuts[j]['peak_deriv']:.1f})")

        suppressed = original_count - len(filtered)
        if suppressed > 0:
            print(f"  Echo suppression: removed {suppressed} echoes "
                  f"(window={self.echo_window_sec}s, ratio={self.echo_ratio})")
        self.cuts = filtered

    def _compute_confidence(self, peak_deriv, spike_duration, slide_motion,
                            left_deriv, spatial_std, ceiling_flag):
        """Score cut confidence as high/medium/low based on multi-signal strength.

        Criteria:
          - peak_deriv >= 30 → strong signal indicator
          - spike_duration >= 3 → sustained spike (not noise)
          - slide_motion >= 5 → visible piece movement below table
          - left_deriv > 0 → secondary ROI confirms brightness increase
          - spatial_std < 100 → uniform brightness change (real table exposure)
          - ceiling_flag → brightness too high, possible break transition
        """
        score = 0

        # Primary signal strength
        if peak_deriv >= 40:
            score += 3
        elif peak_deriv >= 30:
            score += 2
        elif peak_deriv >= 22:
            score += 1

        # Spike duration
        if spike_duration >= 5:
            score += 2
        elif spike_duration >= 3:
            score += 1

        # Slide motion (piece falling off table)
        if slide_motion >= 8:
            score += 1
        elif slide_motion >= 5:
            score += 0.5

        # Left-ROI cross-validation (real cuts affect whole table)
        if left_deriv > 5:
            score += 1
        elif left_deriv > 0:
            score += 0.5

        # Spatial uniformity (real table exposure is uniform)
        if spatial_std < 90:
            score += 0.5

        # Brightness ceiling penalty
        if ceiling_flag:
            score -= 1

        if score >= 4:
            return "high"
        elif score >= 2:
            return "medium"
        else:
            return "low"

    def _merge_close_pairs(self):
        """Post-processing: merge double-detections of the same physical event.

        When two consecutive detections fire within CLOSE_PAIR_GAP seconds and
        at least one has peak_deriv above CLOSE_PAIR_DERIV, they are from the
        same physical cut event — keep only the stronger detection.

        2-worker consecutive cuts pass through unaffected because BOTH events
        have weak deriv (<25), well below CLOSE_PAIR_DERIV (30). This is the
        key distinguishing feature: 4-worker double-detections always involve
        at least one strong spike, while 2-worker consecutive cuts are uniformly
        weak but genuine.
        """
        if not self.cuts or self.close_pair_gap <= 0:
            return

        # v6: flag-only mode — keep all events, add close_pair_suspect metadata
        if not self.merge_close_pairs_enabled:
            flagged = 0
            for i in range(1, len(self.cuts)):
                gap = self.cuts[i]["time_sec"] - self.cuts[i-1]["time_sec"]
                if gap <= self.close_pair_gap:
                    max_d = max(self.cuts[i-1]["peak_deriv"], self.cuts[i]["peak_deriv"])
                    if max_d > self.close_pair_deriv:
                        self.cuts[i]["close_pair_suspect"] = True
                        self.cuts[i-1]["close_pair_suspect"] = True
                        flagged += 1
            if flagged > 0:
                print(f"  Close-pair flag: marked {flagged} pairs as suspect "
                      f"(gap<={self.close_pair_gap}s, deriv>{self.close_pair_deriv}) — NOT merged")
            return

        # v5: original merge behavior
        original_count = len(self.cuts)
        result = [self.cuts[0]]

        for i in range(1, len(self.cuts)):
            gap = self.cuts[i]["time_sec"] - result[-1]["time_sec"]
            if gap <= self.close_pair_gap:
                max_d = max(result[-1]["peak_deriv"], self.cuts[i]["peak_deriv"])
                if max_d > self.close_pair_deriv:
                    # Same physical event — keep the stronger detection
                    dropped = self.cuts[i] if result[-1]["peak_deriv"] >= self.cuts[i]["peak_deriv"] else result[-1]
                    if self.audit_suppressions:
                        self.suppressed_candidates.append({
                            "time_sec": dropped["time_sec"],
                            "peak_deriv": dropped["peak_deriv"],
                            "peak_brightness": dropped.get("peak_brightness", 0),
                            "roi_source": dropped.get("roi_source"),
                            "dropped_by": "close_pair_merge",
                        })
                    if self.cuts[i]["peak_deriv"] > result[-1]["peak_deriv"]:
                        if self.debug:
                            t1 = result[-1]["time_sec"]
                            t2 = self.cuts[i]["time_sec"]
                            print(f"  ~~~ MERGE: replacing {t1:.1f}s (d={result[-1]['peak_deriv']:.1f}) "
                                  f"with {t2:.1f}s (d={self.cuts[i]['peak_deriv']:.1f})")
                        result[-1] = self.cuts[i]
                    elif self.debug:
                        t2 = self.cuts[i]["time_sec"]
                        print(f"  ~~~ MERGE: dropping {t2:.1f}s (d={self.cuts[i]['peak_deriv']:.1f}), "
                              f"keeping {result[-1]['time_sec']:.1f}s (d={result[-1]['peak_deriv']:.1f})")
                    continue  # skip adding this event
            result.append(self.cuts[i])

        merged = original_count - len(result)
        if merged > 0:
            print(f"  Close-pair merge: removed {merged} double-detections "
                  f"(gap<={self.close_pair_gap}s, deriv>{self.close_pair_deriv})")
        self.cuts = result

    def _compute_break_time(self):
        """Sum up break durations from break events."""
        total = 0
        i = 0
        while i < len(self.breaks):
            if self.breaks[i]["type"] == "break_start":
                start = self.breaks[i]["time_sec"]
                end = self.cuts[-1]["time_sec"] if self.cuts else start  # fallback
                if i + 1 < len(self.breaks) and self.breaks[i+1]["type"] == "break_end":
                    end = self.breaks[i+1]["time_sec"]
                    i += 1
                total += end - start
            i += 1
        return total


# ═══════════════════════════════════════════════════════════════
#  CLI
# ═══════════════════════════════════════════════════════════════

def main():
    import argparse
    parser = argparse.ArgumentParser(description="CH19 Cutting Counter (v5-robust / v6-permissive)")
    parser.add_argument("video", help="Path to video file or RTSP URL")
    parser.add_argument("--output", "-o", help="Output JSON path")
    parser.add_argument("--debug", "-d", action="store_true", help="Verbose debug output")
    parser.add_argument("--version", choices=["v5", "v6"], default="v5",
                        help="Algorithm variant: v5-robust (default, precision-biased) or v6-permissive (recall-biased)")

    # Config overrides
    parser.add_argument("--threshold-long", type=float, default=DERIV_THRESHOLD_LONG,
                        help=f"Long-window derivative threshold (default: {DERIV_THRESHOLD_LONG})")
    parser.add_argument("--threshold-short", type=float, default=DERIV_THRESHOLD_SHORT,
                        help=f"Short-window derivative threshold (default: {DERIV_THRESHOLD_SHORT})")
    parser.add_argument("--smooth", type=int, default=SMOOTH_WINDOW,
                        help=f"Smoothing window frames (default: {SMOOTH_WINDOW})")
    parser.add_argument("--min-gap", type=int, default=MIN_CYCLE_FRAMES,
                        help=f"Minimum frames between cuts (default: {MIN_CYCLE_FRAMES})")

    args = parser.parse_args()

    counter = CuttingCounter(
        args.video,
        debug=args.debug,
        version=args.version,
        deriv_threshold_long=args.threshold_long,
        deriv_threshold_short=args.threshold_short,
        smooth_window=args.smooth,
        min_cycle=args.min_gap,
    )

    results = counter.run()

    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\n  Results saved to {args.output}")


if __name__ == "__main__":
    main()
