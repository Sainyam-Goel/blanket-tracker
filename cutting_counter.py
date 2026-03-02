"""
CH19 Cutting Counter — Counts blanket cuts from NVR camera feed.

Detection approach: Brightness derivative spike detection.
  When a cut piece slides off the table, the white surface is exposed,
  causing a rapid brightness INCREASE in the table ROI. We detect these
  positive derivative spikes as cut events.

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

# Table ROI — right portion of white table surface
TABLE_ROI = (820, 240, 1020, 360)

# Slide ROI — below table front edge (motion confirmation metadata)
SLIDE_ROI = (720, 370, 960, 520)

# Smoothing
SMOOTH_WINDOW = 10          # frames for brightness running average (~0.4s at 25fps)

# Derivative detection
DERIV_WINDOW = 50           # frames for derivative computation (~2s at 25fps)
DERIV_THRESHOLD = 20.0      # minimum brightness increase over 2s to count as cut
DERIV_CONFIRM_FRAMES = 3    # derivative must stay above threshold for N frames

# Gap filter
MIN_CYCLE_FRAMES = 75       # min 3s gap between cuts

# Break detection
BREAK_BRIGHTNESS = 235      # brightness above this = empty table
BREAK_HOLD_FRAMES = 75      # must sustain for 3s to confirm break
BREAK_EXIT_FRAMES = 25      # must drop below for 1s to exit break

# Output
FRAME_SAMPLE_RATE = 25      # store frame_data every N frames (1 per second)


# ═══════════════════════════════════════════════════════════════
#  CUTTING COUNTER
# ═══════════════════════════════════════════════════════════════

class CuttingCounter:
    def __init__(self, video_path, **config):
        self.video_path = video_path

        # Config overrides
        self.table_roi = config.get("table_roi", TABLE_ROI)
        self.slide_roi = config.get("slide_roi", SLIDE_ROI)
        self.smooth_window = config.get("smooth_window", SMOOTH_WINDOW)
        self.deriv_window = config.get("deriv_window", DERIV_WINDOW)
        self.deriv_threshold = config.get("deriv_threshold", DERIV_THRESHOLD)
        self.deriv_confirm = config.get("deriv_confirm", DERIV_CONFIRM_FRAMES)
        self.min_cycle = config.get("min_cycle", MIN_CYCLE_FRAMES)
        self.break_brightness = config.get("break_brightness", BREAK_BRIGHTNESS)
        self.break_hold = config.get("break_hold", BREAK_HOLD_FRAMES)
        self.break_exit = config.get("break_exit", BREAK_EXIT_FRAMES)
        self.sample_rate = config.get("sample_rate", FRAME_SAMPLE_RATE)
        self.debug = config.get("debug", False)

        # Brightness tracking
        self.brightness_buffer = deque(maxlen=self.smooth_window)
        self.smoothed_history = deque(maxlen=self.deriv_window + 10)

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

        # Results
        self.cuts = []
        self.breaks = []
        self.frame_data = []
        self.fps = 25.0

    def _get_smoothed(self, raw):
        """Add raw to buffer, return running average."""
        self.brightness_buffer.append(raw)
        smoothed = float(np.mean(self.brightness_buffer))
        self.smoothed_history.append(smoothed)
        return smoothed

    def _get_derivative(self):
        """Compute brightness change over deriv_window frames."""
        if len(self.smoothed_history) < self.deriv_window:
            return 0.0
        current = self.smoothed_history[-1]
        past = self.smoothed_history[-self.deriv_window]
        return current - past

    def _compute_slide_motion(self, gray):
        """Frame-to-frame diff in slide ROI."""
        sx1, sy1, sx2, sy2 = self.slide_roi
        slide_region = gray[sy1:sy2, sx1:sx2].astype(np.float32)
        motion = 0.0
        if self.prev_slide_region is not None:
            motion = float(np.mean(np.abs(slide_region - self.prev_slide_region)))
        self.prev_slide_region = slide_region
        return motion

    def _update_break_state(self, smoothed, frame_idx):
        """Detect break periods (empty table)."""
        if not self.in_break:
            if smoothed > self.break_brightness:
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
            if smoothed < self.break_brightness:
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

        # 1. Table brightness
        raw_brightness = float(np.mean(gray[ty1:ty2, tx1:tx2]))
        smoothed = self._get_smoothed(raw_brightness)

        # 2. Derivative
        deriv = self._get_derivative()

        # 3. Slide motion
        slide_motion = self._compute_slide_motion(gray)

        # 4. Break detection
        self._update_break_state(smoothed, frame_idx)

        # 5. Derivative spike detection (suppressed during breaks)
        event = None
        if not self.in_break:
            if deriv >= self.deriv_threshold:
                if not self.spike_detected:
                    # New spike starting
                    self.deriv_confirm_counter += 1
                    if self.deriv_confirm_counter >= self.deriv_confirm:
                        # Check min gap
                        if (frame_idx - self.last_cut_frame) >= self.min_cycle:
                            self.spike_detected = True
                            self.spike_start_frame = frame_idx
                            self.spike_peak_deriv = deriv
                            self.spike_peak_brightness = smoothed
                            self.slide_motion_peak = slide_motion
                else:
                    # Spike ongoing — track peak
                    if deriv > self.spike_peak_deriv:
                        self.spike_peak_deriv = deriv
                    if smoothed > self.spike_peak_brightness:
                        self.spike_peak_brightness = smoothed
                    self.slide_motion_peak = max(self.slide_motion_peak, slide_motion)
            else:
                if self.spike_detected:
                    # Spike ended — emit cut event
                    self.spike_detected = False
                    self.last_cut_frame = self.spike_start_frame
                    t = self.spike_start_frame / self.fps

                    cut_event = {
                        "type": "cut",
                        "frame": self.spike_start_frame,
                        "time_sec": round(t, 2),
                        "peak_deriv": round(self.spike_peak_deriv, 1),
                        "peak_brightness": round(self.spike_peak_brightness, 1),
                        "slide_motion_peak": round(self.slide_motion_peak, 1),
                    }
                    self.cuts.append(cut_event)
                    event = cut_event

                    if self.debug:
                        print(f"  >>> CUT #{len(self.cuts)} at {t:.1f}s "
                              f"(deriv={self.spike_peak_deriv:.1f}, "
                              f"bright={self.spike_peak_brightness:.1f}, "
                              f"slide={self.slide_motion_peak:.1f})")

                    self.spike_start_frame = None
                    self.spike_peak_deriv = 0
                    self.spike_peak_brightness = 0
                    self.slide_motion_peak = 0

                self.deriv_confirm_counter = 0
        else:
            # Reset spike state during breaks
            self.spike_detected = False
            self.deriv_confirm_counter = 0

        # 6. Store frame data (sampled)
        if frame_idx % self.sample_rate == 0:
            self.frame_data.append({
                "frame": frame_idx,
                "time_sec": round(frame_idx / self.fps, 2),
                "brightness": round(raw_brightness, 1),
                "smoothed": round(smoothed, 1),
                "derivative": round(deriv, 1),
                "slide_motion": round(slide_motion, 1),
                "in_break": self.in_break,
            })

        # Debug output (every 1s)
        if self.debug and frame_idx % 25 == 0:
            t = frame_idx / self.fps
            state = "BREAK" if self.in_break else ("SPIKE" if self.spike_detected else "active")
            print(f"  {t:7.1f}s | bright={smoothed:6.1f} | deriv={deriv:+6.1f} "
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

        print(f"CH19 Cutting Counter v2 (derivative-based)")
        print(f"  Video: {self.video_path}")
        print(f"  FPS: {self.fps:.1f}, Frames: {total_frames}, "
              f"Duration: {total_frames/self.fps:.1f}s ({total_frames/self.fps/60:.1f} min)")
        print(f"  Table ROI: {self.table_roi}")
        print(f"  Deriv window: {self.deriv_window} frames, "
              f"Threshold: {self.deriv_threshold}, Confirm: {self.deriv_confirm}")
        print(f"  Min gap: {self.min_cycle} frames ({self.min_cycle/self.fps:.1f}s)")
        print(f"  Break: brightness>{self.break_brightness} for {self.break_hold} frames")
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
            })

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
        if total_cuts > 0:
            print(f"\n  First 10 cuts:")
            for i, cut in enumerate(self.cuts[:10]):
                mins = int(cut['time_sec'] // 60)
                secs = cut['time_sec'] % 60
                print(f"    #{i+1:3d}: {mins:2d}:{secs:05.2f} "
                      f"(deriv={cut['peak_deriv']:+.1f}, bright={cut['peak_brightness']:.0f})")
            if total_cuts > 20:
                print(f"  ... ({total_cuts - 20} more) ...")
                print(f"  Last 10 cuts:")
                for i, cut in enumerate(self.cuts[-10:]):
                    idx = total_cuts - 10 + i
                    mins = int(cut['time_sec'] // 60)
                    secs = cut['time_sec'] % 60
                    print(f"    #{idx+1:3d}: {mins:2d}:{secs:05.2f} "
                          f"(deriv={cut['peak_deriv']:+.1f}, bright={cut['peak_brightness']:.0f})")

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
                "version": "v2-derivative",
            },
            "config": {
                "table_roi": list(self.table_roi),
                "slide_roi": list(self.slide_roi),
                "deriv_window": self.deriv_window,
                "deriv_threshold": self.deriv_threshold,
                "min_cycle_frames": self.min_cycle,
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
        }

        return results

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
    parser = argparse.ArgumentParser(description="CH19 Cutting Counter v2")
    parser.add_argument("video", help="Path to video file or RTSP URL")
    parser.add_argument("--output", "-o", help="Output JSON path")
    parser.add_argument("--debug", "-d", action="store_true", help="Verbose debug output")

    # Config overrides
    parser.add_argument("--deriv-threshold", type=float, default=DERIV_THRESHOLD,
                        help="Derivative threshold for cut detection (default: 20)")
    parser.add_argument("--deriv-window", type=int, default=DERIV_WINDOW,
                        help="Frames for derivative computation (default: 50)")
    parser.add_argument("--smooth", type=int, default=SMOOTH_WINDOW,
                        help="Smoothing window frames (default: 10)")
    parser.add_argument("--min-gap", type=int, default=MIN_CYCLE_FRAMES,
                        help="Minimum frames between cuts (default: 75)")

    args = parser.parse_args()

    counter = CuttingCounter(
        args.video,
        debug=args.debug,
        deriv_threshold=args.deriv_threshold,
        deriv_window=args.deriv_window,
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
