"""
blanket_counter.py — Accurate Blanket Counter for CH21
=======================================================
Counts blankets via two detection stations:

1. WEIGHING SCALE (primary) — reference-frame comparison, color-agnostic.
   Detects folded blankets placed on the platform scale by comparing each
   frame to a learned "empty scale" reference. Counts only completed cycles
   (loaded → cleared) to avoid false triggers.

2. FOLDING TABLE (secondary) — grayscale texture changes on the table
   surface. Cross-validates with scale events.

Workflow: Worker B throws blanket on table → A & B fold → A places on
scale → checks weight → A tosses to finished pile. B may throw next
blanket on table during weighing (overlap).

USAGE:
  python blanket_counter.py /path/to/NVR_ch21_*.mp4
  python blanket_counter.py video.mp4 --output results.json
  python blanket_counter.py rtsp://user:pass@NVR_IP/ch21 --live

INSTALL:
  pip install opencv-python numpy
"""

import cv2
import numpy as np
import argparse
import json
import sys
import time
from collections import deque
from datetime import datetime
from pathlib import Path


# ─────────────────────────────────────────────────────────────────
# DETECTION ZONES (for 1920x1080 CH21 camera)
# ─────────────────────────────────────────────────────────────────

SCALE_ROI = (1440, 440, 1520, 500)   # weighing scale platform (80x60)
TABLE_ROI = (980, 340, 1240, 450)    # folding table top surface

# ── Scale detection params ──
SCALE_ON_THRESHOLD = 25        # diff must exceed this to trigger "loaded"
SCALE_OFF_THRESHOLD = 15       # diff must fall below this to trigger "empty"
                                # hysteresis dead zone [15-25] prevents chattering
SCALE_SMOOTH_WINDOW = 13       # ~0.52s at 25fps (reduced from 21 to catch fast cycles)
SCALE_MIN_ON_FRAMES = 15       # blanket must be on scale for ≥0.6s to count
                                # ground truth shows all real blankets are ≥0.68s
SCALE_DEBOUNCE_FRAMES = 5      # rising edge must be sustained for 5 frames before
                                # transitioning to "loaded" (filters worker hand passes)
SCALE_MIN_CYCLE_GAP = 100      # 4s at 25fps — minimum gap between consecutive counts
                                # ground truth shows real blankets are always ≥5s apart;
                                # prevents rapid-fire false counting from noise oscillation

# ── Adaptive reference params ──
SCALE_REF_ADAPT_RATE = 0.005   # blend rate when scale is idle (0.5% per frame)
SCALE_REF_IDLE_FRAMES = 75     # must be "empty" for 3s before adapting reference
LIGHTING_JUMP_THRESHOLD = 30   # mean brightness jump > this = lighting change event
SCALE_MAX_LOADED_FRAMES = 250  # 10s at 25fps — no blanket stays on scale this long;
                                # if exceeded, assume reference drift and force recalibrate
SCALE_DRIFT_MARGIN = 1.5       # if loaded diff is within this factor of OFF_THRESHOLD,
                                # it's likely baseline drift not a real blanket

# ── Table detection params ──
TABLE_TEXTURE_THRESHOLD = 75
TABLE_SMOOTH_WINDOW = 9
TABLE_MIN_CYCLE_FRAMES = 30     # blanket must be on table ≥1.2s to count as a fold cycle
TABLE_DEBOUNCE_FRAMES = 5       # rising edge debounce (same logic as scale)
TABLE_MIN_CYCLE_GAP = 100       # 4s at 25fps — minimum gap between table events

# ── Accept/reject classification params ──
CLASSIFY_WINDOW_SEC = 10.0      # after table_off, look for scale event within this window
CLASSIFY_LOOKBACK_SEC = 2.0     # scale event can precede table_off by up to this

# ── Table texture profile params (for fold quality analysis) ──
TABLE_TEXTURE_HISTORY = 50      # store last 50 texture values (2s at 25fps) during table cycle
TABLE_SLOPE_WINDOW = 50         # frames for texture slope computation (2s)
# Texture slope thresholds for classification:
# Accepted blankets show steep negative slope (mean -12.5) as blanket is lifted to scale.
# Rejected blankets show flat slope (mean -0.2) as blanket is pulled/thrown off.
TEXTURE_SLOPE_LIFT_THRESHOLD = -5.0   # slope below this = blanket lifted (going to scale)

# ── Live mode params ──
LIVE_FRAME_DATA_LIMIT = 5000   # keep last N frame records in memory (~200s at 25fps)
CALIBRATION_SAMPLE_FRAMES = 30 # read this many frames to find best "empty" reference


class BlanketCounter:
    def __init__(self, source, live=False):
        self.source = source
        self.live = live

        # Scale state machine
        self.scale_state = 'warmup'    # warmup → empty / loaded
        self.scale_count = 0
        self.scale_on_frame = 0
        self.scale_consecutive_on = 0   # debounce: consecutive frames above ON threshold
        self.scale_idle_frames = 0      # frames continuously in "empty" state
        self.scale_ref = None           # reference "empty" grayscale patch
        self.warmup_above_count = 0    # frames above threshold during warmup
        self.scale_ref_brightness = 0   # mean brightness of reference
        self.scale_buffer = deque(maxlen=SCALE_SMOOTH_WINDOW)
        self.warmup_remaining = SCALE_SMOOTH_WINDOW  # frames before state resolution
        self.last_count_frame = -SCALE_MIN_CYCLE_GAP  # frame of last counted blanket

        # Table state machine
        self.table_state = 'empty'
        self.table_count = 0
        self.table_covered_frames = 0
        self.table_buffer = deque(maxlen=TABLE_SMOOTH_WINDOW)
        self.table_consecutive_on = 0       # debounce counter
        self.last_table_count_frame = -TABLE_MIN_CYCLE_GAP
        self.table_texture_history = deque(maxlen=TABLE_TEXTURE_HISTORY)
        self.table_peak_texture = 0.0       # max texture during current cycle

        # Accept/reject classification
        self.accepted_count = 0
        self.rejected_count = 0

        # Lighting monitoring
        self.prev_frame_brightness = None
        self.lighting_ok = True

        # Data
        self.frame_data = deque(maxlen=LIVE_FRAME_DATA_LIMIT) if live else []
        self.events = []
        self.frame_idx = 0
        self.start_wall_time = None
        self.fps = 25.0

    def _smooth(self, buffer, new_val):
        """Rolling mean using deque (O(1) append, bounded size)."""
        buffer.append(new_val)
        return float(np.mean(buffer))

    def _get_timestamp(self):
        """Wall-clock timestamp in live mode, frame-based in file mode."""
        if self.live and self.start_wall_time is not None:
            return time.time() - self.start_wall_time
        return self.frame_idx / self.fps

    def _calibrate_scale(self, cap):
        """Find the best 'empty scale' reference automatically.

        Reads CALIBRATION_SAMPLE_FRAMES frames, computes the scale ROI
        texture (std dev) for each, and picks the frame with the LOWEST
        texture as the most likely 'empty' state. This avoids the failure
        mode of blindly picking frame 50 which might have a blanket.

        For live streams, reads frames sequentially (no seek).
        For files, samples spread across the first few seconds.
        """
        x1, y1, x2, y2 = SCALE_ROI
        candidates = []
        is_live = self.live
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if is_live or total_frames <= 0:
            # Live: read frames sequentially, skip first 25 (auto-exposure settling)
            print("  Calibrating scale (reading frames sequentially)...")
            for i in range(25 + CALIBRATION_SAMPLE_FRAMES):
                ret, frame = cap.read()
                if not ret:
                    break
                if i < 25:
                    continue  # skip early frames
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                patch = gray[y1:y2, x1:x2]
                texture = float(np.std(patch))
                candidates.append((texture, patch.astype(float), i))
        else:
            # File: sample frames spread across first ~4 seconds
            sample_range = min(total_frames, int(self.fps * 4))
            step = max(1, sample_range // CALIBRATION_SAMPLE_FRAMES)
            for idx in range(0, sample_range, step):
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ret, frame = cap.read()
                if not ret:
                    continue
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                patch = gray[y1:y2, x1:x2]
                texture = float(np.std(patch))
                candidates.append((texture, patch.astype(float), idx))

        if not candidates:
            print("  WARNING: Could not read any frames for calibration!")
            # Fallback: create a neutral gray reference
            self.scale_ref = np.full((y2 - y1, x2 - x1), 128.0)
            self.scale_ref_brightness = 128.0
            return

        # Pick the lowest-texture frame (most likely to be empty metal surface)
        candidates.sort(key=lambda c: c[0])
        best_texture, best_patch, best_idx = candidates[0]

        # Average the top 3 lowest-texture frames for stability
        avg_patches = [c[1] for c in candidates[:min(3, len(candidates))]]
        self.scale_ref = np.mean(avg_patches, axis=0)
        self.scale_ref_brightness = float(np.mean(self.scale_ref))

        print(f"  Scale calibrated: best frame={best_idx} "
              f"texture={best_texture:.1f} "
              f"brightness={self.scale_ref_brightness:.1f}")

        # Save reference snapshot for debugging
        ref_path = Path(self.source).parent / "scale_reference.npy"
        try:
            np.save(str(ref_path), self.scale_ref)
        except Exception:
            pass  # non-critical

        # Rewind for file mode
        if not is_live:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    def _check_lighting(self, gray):
        """Detect sudden lighting changes (lights on/off, IR switch)."""
        mean_brightness = float(np.mean(gray))

        if self.prev_frame_brightness is not None:
            jump = abs(mean_brightness - self.prev_frame_brightness)
            if jump > LIGHTING_JUMP_THRESHOLD:
                if self.lighting_ok:
                    self.lighting_ok = False
                    self.events.append({
                        'type': 'lighting_change',
                        'time_sec': round(self._get_timestamp(), 3),
                        'frame': self.frame_idx,
                        'brightness_jump': round(jump, 1),
                        'note': 'counting paused until stable',
                    })
                    print(f"  [t={self._get_timestamp():6.2f}s] "
                          f"WARNING: Lighting change detected (jump={jump:.0f}). "
                          f"Pausing count.")
                return False
            elif not self.lighting_ok and jump < 5:
                # Lighting has stabilized — recalibrate and resume
                self.lighting_ok = True
                # Force recalibration by reading current frame as new reference
                x1, y1, x2, y2 = SCALE_ROI
                self.scale_ref = gray[y1:y2, x1:x2].astype(float)
                self.scale_ref_brightness = float(np.mean(self.scale_ref))
                self.scale_buffer.clear()
                self.scale_state = 'warmup'
                self.warmup_remaining = SCALE_SMOOTH_WINDOW
                self.events.append({
                    'type': 'lighting_restored',
                    'time_sec': round(self._get_timestamp(), 3),
                    'frame': self.frame_idx,
                    'note': 'recalibrated and resumed counting',
                })
                print(f"  [t={self._get_timestamp():6.2f}s] "
                      f"Lighting stabilized. Recalibrated, resuming count.")

        self.prev_frame_brightness = mean_brightness
        return self.lighting_ok

    def _validate_frame(self, frame):
        """Basic sanity check on frame integrity."""
        if frame is None:
            return False
        h, w = frame.shape[:2]
        if h < 720 or w < 1280:  # minimum expected resolution
            return False
        mean_val = float(np.mean(frame))
        if mean_val < 10 or mean_val > 250:  # all-black or all-white
            return False
        return True

    def process_frame(self, frame, timestamp):
        """Analyze one frame for scale and table detection."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # ── Lighting check ──
        lighting_ok = self._check_lighting(gray)

        # ── SCALE: reference-frame diff with hysteresis ──
        x1, y1, x2, y2 = SCALE_ROI
        scale_patch = gray[y1:y2, x1:x2].astype(float)
        scale_diff_raw = float(np.mean(np.abs(scale_patch - self.scale_ref)))
        scale_diff = self._smooth(self.scale_buffer, scale_diff_raw)

        scale_event = None

        if not lighting_ok:
            # Don't update state during lighting changes
            pass

        elif self.scale_state == 'warmup':
            # Track frames above threshold during warmup for accurate duration
            if scale_diff > SCALE_ON_THRESHOLD:
                self.warmup_above_count += 1

            # Wait for smoothing buffer to fill before resolving state
            self.warmup_remaining -= 1
            if self.warmup_remaining <= 0:
                if scale_diff > SCALE_ON_THRESHOLD:
                    self.scale_state = 'loaded'
                    # Use the earliest frame that was above threshold
                    # so loaded duration includes the warmup period
                    self.scale_on_frame = max(0, self.frame_idx - self.warmup_above_count)
                    scale_event = {
                        'type': 'scale_initial_loaded',
                        'time_sec': round(timestamp, 3),
                        'frame': self.frame_idx,
                        'diff': round(scale_diff, 1),
                        'warmup_above_frames': self.warmup_above_count,
                        'note': 'initial loaded state (counts when cycle completes)',
                    }
                    print(f"  [t={timestamp:6.2f}s f={self.frame_idx:4d}] "
                          f"SCALE: Initial state = loaded (diff={scale_diff:.1f}, "
                          f"above threshold for ~{self.warmup_above_count} warmup frames)")
                else:
                    self.scale_state = 'empty'
                    print(f"  [t={timestamp:6.2f}s f={self.frame_idx:4d}] "
                          f"SCALE: Initial state = empty (diff={scale_diff:.1f})")

        elif self.scale_state == 'empty':
            # Adaptive reference: slowly blend current frame when idle
            self.scale_idle_frames += 1
            if self.scale_idle_frames > SCALE_REF_IDLE_FRAMES:
                self.scale_ref = (
                    (1 - SCALE_REF_ADAPT_RATE) * self.scale_ref
                    + SCALE_REF_ADAPT_RATE * scale_patch
                )
                self.scale_ref_brightness = float(np.mean(self.scale_ref))

            # Rising edge with debounce
            if scale_diff > SCALE_ON_THRESHOLD:
                self.scale_consecutive_on += 1
                if self.scale_consecutive_on >= SCALE_DEBOUNCE_FRAMES:
                    self.scale_state = 'loaded'
                    self.scale_on_frame = self.frame_idx
                    self.scale_idle_frames = 0
                    scale_event = {
                        'type': 'scale_loaded',
                        'time_sec': round(timestamp, 3),
                        'frame': self.frame_idx,
                        'diff': round(scale_diff, 1),
                    }
            else:
                self.scale_consecutive_on = 0

        elif self.scale_state == 'loaded':
            self.scale_idle_frames = 0
            dur_frames = self.frame_idx - self.scale_on_frame

            # ── Drift detection: stuck in "loaded" too long ──
            # No blanket stays on the scale for >10s in normal workflow.
            # If we're stuck, the reference has drifted.
            if dur_frames > SCALE_MAX_LOADED_FRAMES:
                # Check if diff is hovering near the threshold (drift)
                # vs genuinely high (something really sitting there)
                is_drift = scale_diff < SCALE_ON_THRESHOLD * SCALE_DRIFT_MARGIN
                if is_drift:
                    # Reference has drifted — recalibrate from current frame
                    self.scale_ref = scale_patch.copy()
                    self.scale_ref_brightness = float(np.mean(self.scale_ref))
                    self.scale_buffer.clear()
                    self.scale_state = 'warmup'
                    self.warmup_remaining = SCALE_SMOOTH_WINDOW
                    self.warmup_above_count = 0
                    self.scale_consecutive_on = 0
                    # Count the blanket that was legitimately on the scale
                    # before drift took over (first ~1-2s were real)
                    gap_ok = (self.frame_idx - self.last_count_frame) >= SCALE_MIN_CYCLE_GAP
                    if dur_frames > SCALE_MIN_ON_FRAMES and gap_ok:
                        self.scale_count += 1
                        self.last_count_frame = self.frame_idx
                        scale_event = {
                            'type': 'scale_cycle_complete',
                            'time_sec': round(timestamp, 3),
                            'frame': self.frame_idx,
                            'diff': round(scale_diff, 1),
                            'blanket_number': self.scale_count,
                            'on_duration_frames': SCALE_MAX_LOADED_FRAMES,
                            'on_duration_sec': round(SCALE_MAX_LOADED_FRAMES / self.fps, 2),
                            'note': 'drift detected, recalibrated',
                        }
                    print(
                        f"  [t={timestamp:6.2f}s f={self.frame_idx:4d}] "
                        f"DRIFT: Stuck loaded for {dur_frames}f "
                        f"(diff={scale_diff:.1f}). Recalibrating reference."
                    )
                # else: diff is genuinely very high — something is really
                # sitting there (e.g. pile of blankets placed on scale).
                # Keep waiting, but warn periodically.
                elif dur_frames % (SCALE_MAX_LOADED_FRAMES * 2) == 0:
                    print(
                        f"  [t={timestamp:6.2f}s f={self.frame_idx:4d}] "
                        f"WARNING: Scale loaded for {dur_frames}f "
                        f"(diff={scale_diff:.1f}) — object may be sitting on scale"
                    )

            # ── Falling edge: use lower threshold (hysteresis) ──
            elif scale_diff < SCALE_OFF_THRESHOLD:
                gap_frames = self.frame_idx - self.last_count_frame
                gap_ok = gap_frames >= SCALE_MIN_CYCLE_GAP
                if dur_frames >= SCALE_MIN_ON_FRAMES and gap_ok:
                    # Count HERE — on completed cycle, not on loaded transition
                    self.scale_count += 1
                    self.last_count_frame = self.frame_idx
                    scale_event = {
                        'type': 'scale_cycle_complete',
                        'time_sec': round(timestamp, 3),
                        'frame': self.frame_idx,
                        'diff': round(scale_diff, 1),
                        'blanket_number': self.scale_count,
                        'on_duration_frames': dur_frames,
                        'on_duration_sec': round(dur_frames / self.fps, 2),
                    }
                    print(
                        f"  [t={timestamp:6.2f}s f={self.frame_idx:4d}] "
                        f"BLANKET #{self.scale_count:3d} weighed "
                        f"(on scale for {dur_frames} frames / "
                        f"{dur_frames / self.fps:.1f}s)"
                    )
                elif dur_frames < SCALE_MIN_ON_FRAMES:
                    # Too brief — likely a hand pass, not a real blanket
                    scale_event = {
                        'type': 'scale_false_trigger',
                        'time_sec': round(timestamp, 3),
                        'frame': self.frame_idx,
                        'diff': round(scale_diff, 1),
                        'rejected_duration_frames': dur_frames,
                        'reject_reason': 'too_brief',
                    }
                    print(
                        f"  [t={timestamp:6.2f}s f={self.frame_idx:4d}] "
                        f"SCALE: Rejected brief trigger "
                        f"({dur_frames} frames < {SCALE_MIN_ON_FRAMES} min)"
                    )
                elif not gap_ok:
                    # Too close to previous count — noise oscillation
                    scale_event = {
                        'type': 'scale_false_trigger',
                        'time_sec': round(timestamp, 3),
                        'frame': self.frame_idx,
                        'diff': round(scale_diff, 1),
                        'rejected_duration_frames': dur_frames,
                        'reject_reason': 'too_close',
                        'gap_sec': round(gap_frames / self.fps, 2),
                    }
                    print(
                        f"  [t={timestamp:6.2f}s f={self.frame_idx:4d}] "
                        f"SCALE: Rejected — too close to previous "
                        f"({gap_frames / self.fps:.1f}s < "
                        f"{SCALE_MIN_CYCLE_GAP / self.fps:.1f}s min)"
                    )
                self.scale_state = 'empty'
                self.scale_consecutive_on = 0

        if scale_event:
            self.events.append(scale_event)

        # ── TABLE: texture analysis with debounce + gap filtering ──
        x1, y1, x2, y2 = TABLE_ROI
        table_texture_raw = float(np.std(gray[y1:y2, x1:x2]))
        table_texture = self._smooth(self.table_buffer, table_texture_raw)

        table_event = None
        if self.table_state == 'empty':
            if table_texture > TABLE_TEXTURE_THRESHOLD:
                self.table_consecutive_on += 1
                if self.table_consecutive_on >= TABLE_DEBOUNCE_FRAMES:
                    self.table_state = 'covered'
                    self.table_covered_frames = self.table_consecutive_on
                    self.table_texture_history.clear()
                    self.table_peak_texture = table_texture
                    table_event = {
                        'type': 'table_blanket_on',
                        'time_sec': round(timestamp, 3),
                        'frame': self.frame_idx,
                        'texture': round(table_texture, 1),
                    }
            else:
                self.table_consecutive_on = 0

        elif self.table_state == 'covered':
            self.table_covered_frames += 1
            self.table_texture_history.append(table_texture)
            self.table_peak_texture = max(self.table_peak_texture, table_texture)

            if table_texture < TABLE_TEXTURE_THRESHOLD:
                # Falling edge — check duration and gap filters
                gap_frames = self.frame_idx - self.last_table_count_frame
                gap_ok = gap_frames >= TABLE_MIN_CYCLE_GAP
                dur_ok = self.table_covered_frames >= TABLE_MIN_CYCLE_FRAMES

                if dur_ok and gap_ok:
                    # Compute texture slope from history (last 2s)
                    texture_slope = self._compute_texture_slope()

                    self.table_state = 'empty'
                    self.table_count += 1
                    self.last_table_count_frame = self.frame_idx
                    self.table_consecutive_on = 0
                    table_event = {
                        'type': 'table_blanket_off',
                        'time_sec': round(timestamp, 3),
                        'frame': self.frame_idx,
                        'texture': round(table_texture, 1),
                        'table_cycle_number': self.table_count,
                        'duration_frames': self.table_covered_frames,
                        'peak_texture': round(self.table_peak_texture, 1),
                        'texture_slope': round(texture_slope, 2),
                    }
                elif not dur_ok:
                    # Too brief — just reset, don't count
                    self.table_state = 'empty'
                    self.table_consecutive_on = 0
                elif not gap_ok:
                    # Too close to previous — just reset
                    self.table_state = 'empty'
                    self.table_consecutive_on = 0

        if table_event:
            self.events.append(table_event)

        # ── Per-frame record ──
        record = {
            'frame': self.frame_idx,
            'time_sec': round(timestamp, 3),
            'scale_diff': round(scale_diff, 2),
            'scale_state': self.scale_state,
            'table_texture': round(table_texture, 2),
            'table_state': self.table_state,
            'scale_count': self.scale_count,
        }
        self.frame_data.append(record)
        self.frame_idx += 1
        return record

    def _compute_texture_slope(self):
        """Compute texture slope from the end of a table cycle.

        Uses the last TABLE_SLOPE_WINDOW frames of texture history.
        Negative slope = texture dropping = blanket being lifted (going to scale).
        Flat/positive slope = blanket being pulled/thrown off (rejection).

        Returns slope in texture units per second.
        """
        history = list(self.table_texture_history)
        if len(history) < 10:
            return 0.0

        # Use last TABLE_SLOPE_WINDOW frames (or all if fewer)
        window = history[-min(TABLE_SLOPE_WINDOW, len(history)):]

        # Split into first half and second half, compare means
        mid = len(window) // 2
        first_half = np.mean(window[:mid])
        second_half = np.mean(window[mid:])

        # Slope = change per second
        duration_sec = len(window) / self.fps
        if duration_sec > 0:
            return (second_half - first_half) / duration_sec
        return 0.0

    def _reconnect(self, cap):
        """Handle stream reconnection with proper state reset."""
        print("  Stream lost — reconnecting in 3s...")
        time.sleep(3)
        cap.release()
        new_cap = cv2.VideoCapture(self.source)

        if new_cap.isOpened():
            # Flush smoothing buffers
            self.scale_buffer.clear()
            self.table_buffer.clear()

            # Re-enter warmup so state is re-determined from fresh data
            self.scale_state = 'warmup'
            self.warmup_remaining = SCALE_SMOOTH_WINDOW
            self.scale_consecutive_on = 0

            # Recalibrate scale reference
            print("  Reconnected. Recalibrating scale reference...")
            self._calibrate_scale(new_cap)

            self.events.append({
                'type': 'stream_reconnected',
                'time_sec': round(self._get_timestamp(), 3),
                'frame': self.frame_idx,
                'note': 'recalibrated, state reset to warmup',
            })
        else:
            print("  WARNING: Reconnection failed.")

        return new_cap

    def run(self):
        print(f"\n{'='*60}")
        print(f"  Blanket Counter — CH21 Scale + Table Detection")
        print(f"  Source: {self.source}")
        print(f"  Mode: {'LIVE STREAM' if self.live else 'VIDEO FILE'}")
        print(f"{'='*60}\n")

        cap = cv2.VideoCapture(self.source)
        if not cap.isOpened():
            print(f"ERROR: Cannot open source: {self.source}")
            return None

        self.fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration = total_frames / self.fps if self.fps > 0 and total_frames > 0 else 0
        self.start_wall_time = time.time()

        print(f"  Resolution: {width}x{height}, FPS: {self.fps:.2f}")
        if not self.live:
            print(f"  Frames: {total_frames}, Duration: {duration:.2f}s")
        print()

        # Calibrate
        self._calibrate_scale(cap)

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    if self.live:
                        cap = self._reconnect(cap)
                        continue
                    else:
                        break

                if not self._validate_frame(frame):
                    self.frame_idx += 1
                    continue

                timestamp = self._get_timestamp()
                self.process_frame(frame, timestamp)

                if not self.live and self.frame_idx % 100 == 0 and total_frames > 0:
                    pct = self.frame_idx / total_frames * 100
                    print(f"  ... {self.frame_idx}/{total_frames} ({pct:.0f}%) "
                          f"| weighed: {self.scale_count} | table: {self.table_count}")

        except KeyboardInterrupt:
            print("\n  Stopped by user.")
        finally:
            cap.release()

        # Final duration
        if self.live:
            duration = time.time() - self.start_wall_time
        elif duration == 0 and self.frame_idx > 0:
            duration = self.frame_idx / self.fps

        # End-of-video: count blanket still on scale if it was there long enough
        still_on_scale = self.scale_state == 'loaded'
        if still_on_scale:
            dur_frames = self.frame_idx - self.scale_on_frame
            gap_ok = (self.frame_idx - self.last_count_frame) >= SCALE_MIN_CYCLE_GAP
            if dur_frames >= SCALE_MIN_ON_FRAMES and gap_ok:
                self.scale_count += 1
                self.events.append({
                    'type': 'scale_cycle_complete',
                    'time_sec': round(self.frame_idx / self.fps, 3),
                    'frame': self.frame_idx,
                    'blanket_number': self.scale_count,
                    'on_duration_frames': dur_frames,
                    'on_duration_sec': round(dur_frames / self.fps, 2),
                    'note': 'video ended with blanket on scale',
                })
                print(
                    f"  [END] BLANKET #{self.scale_count:3d} — still on scale at video end "
                    f"({dur_frames} frames / {dur_frames / self.fps:.1f}s)"
                )

        # ── Accept/Reject classification via cross-correlation ──
        self._classify_blankets()

        result = {
            'source': str(self.source),
            'video_info': {
                'width': width,
                'height': height,
                'fps': round(self.fps, 2),
                'total_frames': total_frames if not self.live else self.frame_idx,
                'duration_sec': round(duration, 3),
            },
            'detection_config': {
                'scale_roi': list(SCALE_ROI),
                'table_roi': list(TABLE_ROI),
                'scale_on_threshold': SCALE_ON_THRESHOLD,
                'scale_off_threshold': SCALE_OFF_THRESHOLD,
                'scale_smooth_window': SCALE_SMOOTH_WINDOW,
                'scale_debounce_frames': SCALE_DEBOUNCE_FRAMES,
                'table_texture_threshold': TABLE_TEXTURE_THRESHOLD,
                'table_min_cycle_frames': TABLE_MIN_CYCLE_FRAMES,
                'table_min_cycle_gap': TABLE_MIN_CYCLE_GAP,
                'classify_window_sec': CLASSIFY_WINDOW_SEC,
                'texture_slope_lift_threshold': TEXTURE_SLOPE_LIFT_THRESHOLD,
            },
            'results': {
                'blankets_weighed': self.scale_count,
                'still_on_scale': self.scale_state == 'loaded',
                'table_cycles': self.table_count,
                'still_on_table': (self.table_state == 'covered'
                                   and self.table_covered_frames >= TABLE_MIN_CYCLE_FRAMES),
                'accepted': self.accepted_count,
                'rejected': self.rejected_count,
                'total_blankets': self.accepted_count + self.rejected_count,
                'projected_per_hour': (
                    round(self.scale_count / duration * 3600)
                    if duration > 10 else None
                ),
            },
            'events': self.events,
            'frames': list(self.frame_data),
        }

        self._print_summary(result)
        return result

    def _classify_blankets(self):
        """Cross-correlate scale and table events to classify accepted/rejected.

        Logic:
        1. Every scale_cycle_complete IS an accepted blanket.
           Create blanket_accepted for each one.
        2. For each table_blanket_off, check if ANY scale event is nearby.
           If yes → this table cycle is the table-side of that accepted blanket
           (skip it, already counted above).
           If no → this is a REJECTED blanket (tossed without weighing).

        Table texture profile data (peak_texture, texture_slope) is stored as
        metadata on events for analysis but not yet used for classification.
        """
        scale_events = sorted(
            [e for e in self.events if e['type'] == 'scale_cycle_complete'],
            key=lambda e: e['time_sec']
        )
        table_events = sorted(
            [e for e in self.events if e['type'] == 'table_blanket_off'],
            key=lambda e: e['time_sec']
        )
        scale_times = [e['time_sec'] for e in scale_events]

        # Helper: find nearest table cycle to a scale event
        def find_nearby_table(scale_time):
            """Find the nearest table cycle to a scale event."""
            best = None
            best_dist = float('inf')
            for te in table_events:
                delta = te['time_sec'] - scale_time
                if -CLASSIFY_LOOKBACK_SEC <= delta <= CLASSIFY_WINDOW_SEC:
                    dist = abs(delta)
                    if dist < best_dist:
                        best_dist = dist
                        best = te
                elif delta > CLASSIFY_WINDOW_SEC:
                    break
            return best

        # Step 1: All scale events are accepted
        for se in scale_events:
            self.accepted_count += 1
            nearby_table = find_nearby_table(se['time_sec'])
            self.events.append({
                'type': 'blanket_accepted',
                'time_sec': se['time_sec'],
                'frame': se['frame'],
                'blanket_number': self.accepted_count,
                'scale_duration_sec': se.get('on_duration_sec'),
                'scale_diff': se.get('diff'),
                'texture_slope': nearby_table.get('texture_slope') if nearby_table else None,
                'peak_texture': nearby_table.get('peak_texture') if nearby_table else None,
            })

        # Step 2: Table events with no nearby scale event are rejected
        for te in table_events:
            t_off = te['time_sec']
            has_scale = False
            for st in scale_times:
                delta = st - t_off
                if -CLASSIFY_LOOKBACK_SEC <= delta <= CLASSIFY_WINDOW_SEC:
                    has_scale = True
                    break
                if delta > CLASSIFY_WINDOW_SEC:
                    break

            if not has_scale:
                self.rejected_count += 1
                self.events.append({
                    'type': 'blanket_rejected',
                    'time_sec': t_off,
                    'frame': te['frame'],
                    'blanket_number': self.rejected_count,
                    'table_duration_frames': te.get('duration_frames'),
                    'table_texture': te.get('texture'),
                    'peak_texture': te.get('peak_texture'),
                    'texture_slope': te.get('texture_slope'),
                    'reject_reason': 'no_scale',
                })

        print(f"\n  Classification: {self.accepted_count} accepted, "
              f"{self.rejected_count} rejected, "
              f"{self.accepted_count + self.rejected_count} total")

    def _print_summary(self, result):
        r = result['results']
        print(f"\n{'='*60}")
        print(f"  BLANKET COUNT RESULTS")
        print(f"{'='*60}")
        print(f"  Blankets weighed (scale):   {r['blankets_weighed']}")
        print(f"  Still on scale at end:      {'Yes' if r['still_on_scale'] else 'No'}")
        print(f"  Table fold cycles:          {r['table_cycles']}")
        print(f"  Still on table at end:      {'Yes' if r['still_on_table'] else 'No'}")
        print(f"  ──────────────────────────────────")
        print(f"  ACCEPTED (weighed):         {r['accepted']}")
        print(f"  REJECTED (tossed):          {r['rejected']}")
        print(f"  TOTAL BLANKETS:             {r['total_blankets']}")
        if r['projected_per_hour'] is not None:
            print(f"  Projected rate:             ~{r['projected_per_hour']}/hr (accepted)")
        print(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(description="Blanket Counter — CH21 Scale + Table")
    parser.add_argument("videos", nargs="*", help="Video file paths")
    parser.add_argument("--source", help="Single video file or RTSP URL")
    parser.add_argument("--live", action="store_true", help="Live stream mode")
    parser.add_argument("--output", default="blanket_count.json", help="Output JSON file")
    args = parser.parse_args()

    sources = list(args.videos or [])
    if args.source:
        sources.append(args.source)

    if not sources:
        parser.print_help()
        print("\nERROR: No video source specified.")
        sys.exit(1)

    all_results = []
    for src in sources:
        counter = BlanketCounter(source=src, live=args.live)
        result = counter.run()
        if result:
            all_results.append(result)

    output = {
        'generated_at': datetime.now().isoformat(),
        'videos': all_results,
        'total_blankets_weighed': sum(
            r['results']['blankets_weighed'] for r in all_results
        ),
        'total_accepted': sum(r['results']['accepted'] for r in all_results),
        'total_rejected': sum(r['results']['rejected'] for r in all_results),
        'total_blankets': sum(r['results']['total_blankets'] for r in all_results),
    }

    Path(args.output).write_text(json.dumps(output, indent=2))
    print(f"Results saved to: {args.output}")


if __name__ == "__main__":
    main()
