#!/usr/bin/env python3
"""CH27 Ground-Truth Labeling Tool.

Frame-accurate event labeler for the taping camera. Pre-populates suggested
load/toss events from the v2 detection algorithm; the labeler reviews,
edits, deletes false positives, and adds missed events.

Usage
-----
    python3 gt_labeler.py /path/to/video.mp4

Keyboard
--------
    ←/→            Step ±1 frame
    Shift+←/→      Step ±25 frames (1 s)
    ↑/↓            Step ±125 frames (5 s)
    Space          Play / pause at 1×
    Tab            Toggle active table (LEFT ↔ RIGHT)
    A              Mark "load" on active table at current frame
    D              Mark "toss" on active table at current frame
    Enter          Edit note on currently-selected label
    Backspace/Del  Delete currently-selected label
    Click label    Jump video to that frame
    Cmd/Ctrl+S     Save labels to <video>.labels.json
    Cmd/Ctrl+Z     Undo last add/delete
    R              Toggle ROI overlay
    O              Open another video
    ?              Show keyboard help dialog

Output
------
A sidecar JSON `<video_basename>.labels.json` next to the video file. See
the inline JSON_SCHEMA constant for the schema. The classifier-training
pipeline consumes this file directly.
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, simpledialog
from PIL import Image, ImageTk

# Reuse v2 ROI constants for the overlay (visualised but not enforced here)
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
try:
    from taping_counter import (
        LEFT_TABLE_ROI_V2, RIGHT_TABLE_ROI_V2, HEAP_ROI,
        LEFT_AIR_ROI_V2, RIGHT_AIR_ROI_V2,
        TapingCounter,
    )
except Exception as ex:  # pragma: no cover — degrade gracefully
    print(f"[warn] Could not import from taping_counter.py: {ex}")
    LEFT_TABLE_ROI_V2 = (200, 750, 650, 940)
    RIGHT_TABLE_ROI_V2 = (1140, 720, 1750, 970)
    HEAP_ROI = (700, 350, 1220, 750)
    LEFT_AIR_ROI_V2 = (160, 580, 660, 740)
    RIGHT_AIR_ROI_V2 = (1180, 580, 1750, 720)
    TapingCounter = None

# Display canvas size (source 1920×1080 → 1280×720 = 0.667× scale)
DISPLAY_W, DISPLAY_H = 1280, 720
SCALE_X = DISPLAY_W / 1920
SCALE_Y = DISPLAY_H / 1080

# Color scheme (RGB tuples for matplotlib-style + Tk hex strings)
COLOR = {
    "left_load":  "#a7f3d0",   # light green
    "left_toss":  "#10b981",   # bold green
    "right_load": "#bfdbfe",   # light blue
    "right_toss": "#3b82f6",   # bold blue
    "active_left":  "#10b981",
    "active_right": "#3b82f6",
    "v2_auto_unconfirmed": "#fbbf24",  # amber — not yet confirmed
    "manual": "#ffffff",
    "selected_bg": "#1e3a8a",
}

AUTO_SAVE_INTERVAL_MS = 30_000  # 30 s


# ─────────────────────────────────────────────────────────────────
#  Sidecar JSON helpers
# ─────────────────────────────────────────────────────────────────

def sidecar_path(video_path):
    return Path(video_path).with_suffix(".labels.json")


def v2_cache_path(video_path):
    return Path(video_path).with_suffix(".v2_detections.json")


def load_sidecar(video_path):
    p = sidecar_path(video_path)
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except Exception as ex:
        print(f"[warn] failed to read {p}: {ex}")
        return None


def save_sidecar(video_path, data):
    p = sidecar_path(video_path)
    p.write_text(json.dumps(data, indent=2))
    return p


# ─────────────────────────────────────────────────────────────────
#  v2 pre-population (cached)
# ─────────────────────────────────────────────────────────────────

def get_v2_detections(video_path, fps):
    """Return list of (frame, time_sec, table, type) detected by v2.
    Cached to a sidecar so we only run v2 once per video.
    """
    cache = v2_cache_path(video_path)
    if cache.exists():
        try:
            data = json.loads(cache.read_text())
            return data.get("events", [])
        except Exception:
            pass

    if TapingCounter is None:
        print("[warn] TapingCounter not available — skipping pre-population")
        return []

    print(f"[info] Running v2 detector on {video_path} (one-time, may take ~30s)…")
    try:
        counter = TapingCounter(str(video_path), version="v2", debug=False)
        results = counter.run()
        # Cache only the events we need
        ev = []
        for e in results.get("events", []):
            ev.append({
                "frame": int(e["frame"]),
                "time_sec": float(e["time_sec"]),
                "table": e["table"],
                "type": "toss",  # v2 emits toss events
                "peak_signal": e.get("peak_signal"),
                "air_motion_peak": e.get("air_motion_peak"),
            })
        cache.write_text(json.dumps({"events": ev, "v2_run_at": datetime.now().isoformat()}, indent=2))
        print(f"[info] v2 detected {len(ev)} candidate tosses → cached at {cache.name}")
        return ev
    except Exception as ex:
        print(f"[warn] v2 pre-population failed: {ex}")
        return []


# ─────────────────────────────────────────────────────────────────
#  Main app
# ─────────────────────────────────────────────────────────────────

class GTLabeler:
    def __init__(self, root, video_path):
        self.root = root
        self.root.title(f"CH27 GT Labeler — {Path(video_path).name}")
        self.video_path = str(video_path)

        # Open video
        self.cap = cv2.VideoCapture(self.video_path)
        if not self.cap.isOpened():
            messagebox.showerror("Error", f"Cannot open video:\n{self.video_path}")
            root.destroy()
            return
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 25.0
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.duration_sec = self.total_frames / self.fps

        # State
        self.frame_idx = 0
        self.active_table = "left"
        self.show_rois = True
        self.is_playing = False
        self.last_save_time = 0.0
        self.dirty = False

        # Labels & undo
        self.labels = []  # list of {frame, time_sec, table, type, note, source, confirmed}
        self.undo_stack = []  # list of (action, label-snapshot) tuples
        self.selected_idx = None

        # Build UI then load existing sidecar / pre-populate
        self._build_ui()
        self._load_or_prepopulate_labels()
        # Force-read frame 0 so cv2 internal state is fully initialized before
        # we attempt seeking. Without this, the first cap.set+read on HEVC
        # video can return None silently.
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        _ok, _f = self.cap.read()
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        self._render_frame()
        self._refresh_label_list()
        self._update_status()
        # Grab keyboard focus on the root so arrow keys / A / D fire immediately
        self.root.after(100, lambda: self.root.focus_force())

        # Auto-save loop
        self.root.after(AUTO_SAVE_INTERVAL_MS, self._autosave_tick)

    # ── UI construction ────────────────────────────────────────

    def _build_ui(self):
        # Top toolbar
        toolbar = tk.Frame(self.root, bg="#0e1116", padx=6, pady=6)
        toolbar.pack(side=tk.TOP, fill=tk.X)
        tk.Button(toolbar, text="Open…", command=self._open_dialog).pack(side=tk.LEFT, padx=2)
        tk.Button(toolbar, text="Save  (⌘S)", command=self._save).pack(side=tk.LEFT, padx=2)
        tk.Button(toolbar, text="Help (?)", command=self._show_help).pack(side=tk.LEFT, padx=2)
        tk.Label(toolbar, text="  Active:", bg="#0e1116", fg="#9ca3af").pack(side=tk.LEFT, padx=(20, 2))
        self.active_label = tk.Label(toolbar, text="LEFT", bg=COLOR["active_left"], fg="black",
                                      font=("Helvetica", 12, "bold"), padx=10)
        self.active_label.pack(side=tk.LEFT)
        tk.Button(toolbar, text="Toggle Table (Tab)", command=self._toggle_table).pack(side=tk.LEFT, padx=2)
        self.roi_btn = tk.Button(toolbar, text="ROIs: ON (R)", command=self._toggle_rois)
        self.roi_btn.pack(side=tk.LEFT, padx=2)

        # Main area — canvas left, panel right
        main = tk.Frame(self.root, bg="#0e1116")
        main.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        # Video canvas
        canvas_frame = tk.Frame(main, bg="#0e1116")
        canvas_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.canvas = tk.Canvas(canvas_frame, width=DISPLAY_W, height=DISPLAY_H,
                                 bg="black", highlightthickness=0)
        self.canvas.pack(padx=4, pady=4)

        # Time slider
        slider_frame = tk.Frame(canvas_frame, bg="#0e1116")
        slider_frame.pack(fill=tk.X, padx=4)
        self.slider_var = tk.DoubleVar(value=0.0)
        self.slider = tk.Scale(slider_frame, from_=0, to=max(1, self.total_frames - 1),
                                orient=tk.HORIZONTAL, variable=self.slider_var,
                                command=self._on_slider, showvalue=False,
                                bg="#0e1116", fg="#9ca3af", troughcolor="#1f2937",
                                highlightthickness=0)
        self.slider.pack(fill=tk.X)

        # Label panel
        panel = tk.Frame(main, bg="#11141a", width=320)
        panel.pack(side=tk.RIGHT, fill=tk.Y, padx=(2, 0))
        panel.pack_propagate(False)
        tk.Label(panel, text="LABELS", bg="#11141a", fg="#9ca3af",
                  font=("Helvetica", 10, "bold")).pack(anchor=tk.W, padx=8, pady=(8, 4))
        # Listbox + scrollbar
        list_frame = tk.Frame(panel, bg="#11141a")
        list_frame.pack(fill=tk.BOTH, expand=True, padx=4, pady=2)
        scrollbar = tk.Scrollbar(list_frame, orient=tk.VERTICAL)
        self.listbox = tk.Listbox(list_frame, bg="#0e1116", fg="white",
                                    selectbackground=COLOR["selected_bg"],
                                    activestyle="none", font=("Menlo", 11),
                                    yscrollcommand=scrollbar.set)
        scrollbar.config(command=self.listbox.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.listbox.bind("<<ListboxSelect>>", self._on_listbox_select)
        self.listbox.bind("<Double-Button-1>", self._on_listbox_dblclick)

        # Note + actions
        note_frame = tk.Frame(panel, bg="#11141a")
        note_frame.pack(fill=tk.X, padx=8, pady=8)
        tk.Label(note_frame, text="Note:", bg="#11141a", fg="#9ca3af").pack(anchor=tk.W)
        self.note_var = tk.StringVar()
        self.note_entry = tk.Entry(note_frame, textvariable=self.note_var,
                                     bg="#0e1116", fg="white", insertbackground="white")
        self.note_entry.pack(fill=tk.X, pady=2)
        self.note_entry.bind("<Return>", lambda e: self._save_note())
        btn_row = tk.Frame(note_frame, bg="#11141a")
        btn_row.pack(fill=tk.X, pady=4)
        tk.Button(btn_row, text="Save Note", command=self._save_note).pack(side=tk.LEFT)
        tk.Button(btn_row, text="Confirm v2", command=self._confirm_selected).pack(side=tk.LEFT, padx=4)
        tk.Button(btn_row, text="Delete (⌫)", command=self._delete_selected).pack(side=tk.RIGHT)

        # Status bar
        self.status_var = tk.StringVar()
        status = tk.Label(self.root, textvariable=self.status_var,
                           bg="#0e1116", fg="#9ca3af", anchor=tk.W,
                           font=("Menlo", 10))
        status.pack(side=tk.BOTTOM, fill=tk.X, padx=4, pady=2)

        # Key bindings (focus root so they fire regardless of widget focus,
        # except inside the note entry).
        # Tab is bound via _on_tab_key which returns "break" to suppress Tk's
        # default focus traversal — without it, the second Tab press moves
        # widget focus instead of toggling the table.
        for seq, fn in [
            ("<Left>",         lambda e: self._step(-1)),
            ("<Right>",        lambda e: self._step(+1)),
            ("<Shift-Left>",   lambda e: self._step(-int(self.fps))),
            ("<Shift-Right>",  lambda e: self._step(+int(self.fps))),
            ("<Up>",           lambda e: self._step(-int(self.fps * 5))),
            ("<Down>",         lambda e: self._step(+int(self.fps * 5))),
            ("<space>",        lambda e: self._toggle_play()),
            ("<Tab>",          self._on_tab_key),
            ("<Key-Tab>",      self._on_tab_key),
            ("<KeyPress-t>",   lambda e: self._toggle_table()),  # T as alt for Tab
            ("<KeyPress-T>",   lambda e: self._toggle_table()),
            ("<KeyPress-a>",   lambda e: self._mark_event("load")),
            ("<KeyPress-A>",   lambda e: self._mark_event("load")),
            ("<KeyPress-d>",   lambda e: self._mark_event("toss")),
            ("<KeyPress-D>",   lambda e: self._mark_event("toss")),
            ("<KeyPress-r>",   lambda e: self._toggle_rois()),
            ("<KeyPress-R>",   lambda e: self._toggle_rois()),
            ("<KeyPress-o>",   lambda e: self._open_dialog()),
            ("<KeyPress-O>",   lambda e: self._open_dialog()),
            ("<question>",     lambda e: self._show_help()),
            ("<Return>",       lambda e: self._focus_note()),
            ("<BackSpace>",    lambda e: self._delete_selected()),
            ("<Delete>",       lambda e: self._delete_selected()),
            ("<Command-s>",    lambda e: self._save()),
            ("<Control-s>",    lambda e: self._save()),
            ("<Command-z>",    lambda e: self._undo()),
            ("<Control-z>",    lambda e: self._undo()),
        ]:
            self.root.bind(seq, fn)

        # Also bind Tab on every widget that could steal focus, returning "break"
        # so Tk doesn't run its own focus traversal handler.
        for w in (self.canvas, self.listbox, self.slider):
            w.bind("<Tab>", self._on_tab_key)
            w.bind("<Key-Tab>", self._on_tab_key)

        # Don't intercept arrows when focus is in the note entry
        self.note_entry.bind("<Left>", lambda e: None)
        self.note_entry.bind("<Right>", lambda e: None)

    def _on_tab_key(self, _event):
        """Handle Tab: toggle table + suppress Tk's default focus traversal."""
        self._toggle_table()
        return "break"

    # ── Pre-populate / load ──────────────────────────────────────

    def _load_or_prepopulate_labels(self):
        existing = load_sidecar(self.video_path)
        if existing and existing.get("labels"):
            self.labels = existing["labels"]
            print(f"[info] Loaded {len(self.labels)} existing labels from sidecar")
            return

        # Pre-populate from v2
        v2_events = get_v2_detections(self.video_path, self.fps)
        for e in v2_events:
            self.labels.append({
                "frame": e["frame"],
                "time_sec": e["time_sec"],
                "table": e["table"],
                "type": "toss",
                "note": "",
                "source": "v2_auto",
                "confirmed": False,
            })
        self.labels.sort(key=lambda l: l["frame"])

    # ── Frame rendering ──────────────────────────────────────────

    def _render_frame(self):
        # On the FIRST render, just read sequentially (cv2 seek + HEVC sometimes
        # returns junk on the very first cap.read() after instantiation).
        # On subsequent renders, only seek if we're not already at the right frame.
        cur_pos = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))
        if cur_pos != self.frame_idx:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.frame_idx)
        ret, frame = self.cap.read()
        if not ret or frame is None:
            # Show an error placeholder rather than silently leaving the canvas blank
            self.canvas.delete("all")
            self.canvas.create_rectangle(0, 0, DISPLAY_W, DISPLAY_H, fill="#1f2937", outline="")
            self.canvas.create_text(DISPLAY_W // 2, DISPLAY_H // 2,
                                     text=f"⚠ Frame {self.frame_idx} could not be read\n"
                                          f"(cv2.read() returned None)\n"
                                          f"Try arrow keys to advance",
                                     fill="#fbbf24", font=("Helvetica", 14), justify=tk.CENTER)
            return
        # Convert BGR → RGB and resize for display
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb_small = cv2.resize(rgb, (DISPLAY_W, DISPLAY_H), interpolation=cv2.INTER_AREA)

        if self.show_rois:
            self._draw_rois(rgb_small)
        self._draw_overlay(rgb_small)

        img = Image.fromarray(rgb_small)
        self.tk_image = ImageTk.PhotoImage(image=img)
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.tk_image)
        self.slider_var.set(self.frame_idx)

    def _draw_rois(self, rgb):
        """Draw ROI rectangles on the RGB display image (in-place)."""
        rois = [
            ("L-table", LEFT_TABLE_ROI_V2,  (16, 185, 129)),   # green
            ("R-table", RIGHT_TABLE_ROI_V2, (59, 130, 246)),   # blue
            ("L-air",   LEFT_AIR_ROI_V2,    (110, 231, 183)),  # light green
            ("R-air",   RIGHT_AIR_ROI_V2,   (147, 197, 253)),  # light blue
            ("HEAP",    HEAP_ROI,           (180, 180, 180)),  # grey
        ]
        for name, roi, color in rois:
            x1 = int(roi[0] * SCALE_X); y1 = int(roi[1] * SCALE_Y)
            x2 = int(roi[2] * SCALE_X); y2 = int(roi[3] * SCALE_Y)
            cv2.rectangle(rgb, (x1, y1), (x2, y2), color, 2)
            cv2.putText(rgb, name, (x1 + 4, y1 + 16),
                         cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

    def _draw_overlay(self, rgb):
        """Time/frame stamp top-left, active table badge top-right."""
        t_sec = self.frame_idx / self.fps
        text = f"frame {self.frame_idx} - {t_sec:7.2f}s / {self.duration_sec:.0f}s"
        # Black background pill
        cv2.rectangle(rgb, (8, 8), (8 + 380, 38), (0, 0, 0), -1)
        cv2.putText(rgb, text, (16, 30),
                     cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 1, cv2.LINE_AA)
        # Active table badge top-right
        badge_color = (16, 185, 129) if self.active_table == "left" else (59, 130, 246)
        cv2.rectangle(rgb, (DISPLAY_W - 200, 8), (DISPLAY_W - 8, 38), badge_color, -1)
        cv2.putText(rgb, f"ACTIVE: {self.active_table.upper()}",
                     (DISPLAY_W - 192, 30), cv2.FONT_HERSHEY_SIMPLEX,
                     0.65, (0, 0, 0), 2, cv2.LINE_AA)

    # ── Navigation ───────────────────────────────────────────────

    def _step(self, delta):
        self.is_playing = False
        new_idx = max(0, min(self.total_frames - 1, self.frame_idx + delta))
        if new_idx != self.frame_idx:
            self.frame_idx = new_idx
            self._render_frame()
            self._update_status()

    def _toggle_play(self):
        self.is_playing = not self.is_playing
        if self.is_playing:
            self._play_tick()

    def _play_tick(self):
        if not self.is_playing:
            return
        self.frame_idx = min(self.frame_idx + 1, self.total_frames - 1)
        if self.frame_idx >= self.total_frames - 1:
            self.is_playing = False
        self._render_frame()
        self._update_status()
        delay_ms = max(1, int(1000 / self.fps))
        self.root.after(delay_ms, self._play_tick)

    def _on_slider(self, val):
        try:
            new_idx = int(float(val))
        except ValueError:
            return
        if new_idx != self.frame_idx:
            self.frame_idx = new_idx
            self._render_frame()
            self._update_status()

    # ── Table toggle / ROI overlay ──────────────────────────────

    def _toggle_table(self):
        self.active_table = "right" if self.active_table == "left" else "left"
        bg = COLOR["active_left"] if self.active_table == "left" else COLOR["active_right"]
        self.active_label.config(text=self.active_table.upper(), bg=bg)
        self._render_frame()
        self._update_status()

    def _toggle_rois(self):
        self.show_rois = not self.show_rois
        self.roi_btn.config(text=f"ROIs: {'ON' if self.show_rois else 'OFF'} (R)")
        self._render_frame()

    # ── Label add / delete / select ─────────────────────────────

    def _mark_event(self, evt_type):
        # If the focused widget is the note entry, don't treat A/D as event keys
        if self.root.focus_get() is self.note_entry:
            return
        label = {
            "frame": self.frame_idx,
            "time_sec": round(self.frame_idx / self.fps, 3),
            "table": self.active_table,
            "type": evt_type,
            "note": "",
            "source": "manual",
            "confirmed": True,
        }
        self.labels.append(label)
        self.labels.sort(key=lambda l: l["frame"])
        self.undo_stack.append(("add", label))
        self.dirty = True
        self._refresh_label_list()
        # Select the just-added label
        for i, l in enumerate(self.labels):
            if l is label:
                self._select_label(i)
                break
        self._update_status()

    def _delete_selected(self):
        if self.selected_idx is None:
            return
        # If focus is in the note entry, let backspace edit text instead
        if self.root.focus_get() is self.note_entry:
            return
        i = self.selected_idx
        if i < 0 or i >= len(self.labels):
            return
        removed = self.labels.pop(i)
        self.undo_stack.append(("delete", removed))
        self.dirty = True
        self.selected_idx = None
        self._refresh_label_list()
        self._update_status()

    def _undo(self):
        if not self.undo_stack:
            return
        action, payload = self.undo_stack.pop()
        if action == "add":
            try:
                self.labels.remove(payload)
            except ValueError:
                pass
        elif action == "delete":
            self.labels.append(payload)
            self.labels.sort(key=lambda l: l["frame"])
        self.dirty = True
        self._refresh_label_list()
        self._update_status()

    def _confirm_selected(self):
        if self.selected_idx is None:
            return
        l = self.labels[self.selected_idx]
        if l.get("source") == "v2_auto":
            l["confirmed"] = True
            self.dirty = True
            self._refresh_label_list()
            self._update_status()

    def _save_note(self):
        if self.selected_idx is None:
            return
        self.labels[self.selected_idx]["note"] = self.note_var.get().strip()
        self.dirty = True
        self._refresh_label_list()
        self._update_status()
        # Return focus to root for keyboard nav
        self.root.focus_set()

    def _focus_note(self):
        if self.selected_idx is not None:
            self.note_entry.focus_set()
            self.note_entry.icursor(tk.END)

    # ── Label list rendering / selection ────────────────────────

    def _label_display(self, label):
        t = label["time_sec"]
        tag = label["table"][0].upper() + label["type"][0].upper()  # LL, LT, RL, RT
        marker = "○" if (label.get("source") == "v2_auto" and not label.get("confirmed")) else "●"
        note = f" — {label['note']}" if label.get("note") else ""
        return f"{marker} {t:7.2f}s {tag} f{label['frame']:>5d}{note}"

    def _refresh_label_list(self):
        cur_sel = self.selected_idx
        self.listbox.delete(0, tk.END)
        for i, l in enumerate(self.labels):
            self.listbox.insert(tk.END, self._label_display(l))
            # Color per row
            if l["table"] == "left":
                fg = COLOR["left_toss"] if l["type"] == "toss" else COLOR["left_load"]
            else:
                fg = COLOR["right_toss"] if l["type"] == "toss" else COLOR["right_load"]
            if l.get("source") == "v2_auto" and not l.get("confirmed"):
                fg = COLOR["v2_auto_unconfirmed"]
            self.listbox.itemconfig(i, fg=fg)
        if cur_sel is not None and 0 <= cur_sel < len(self.labels):
            self.listbox.selection_set(cur_sel)
            self.listbox.see(cur_sel)

    def _select_label(self, idx):
        self.selected_idx = idx
        self.listbox.selection_clear(0, tk.END)
        self.listbox.selection_set(idx)
        self.listbox.see(idx)
        # Populate note entry
        self.note_var.set(self.labels[idx].get("note", ""))

    def _on_listbox_select(self, _event):
        sel = self.listbox.curselection()
        if not sel:
            return
        i = sel[0]
        self.selected_idx = i
        self.note_var.set(self.labels[i].get("note", ""))

    def _on_listbox_dblclick(self, _event):
        sel = self.listbox.curselection()
        if not sel:
            return
        i = sel[0]
        self.selected_idx = i
        self.frame_idx = self.labels[i]["frame"]
        self._render_frame()
        self._update_status()

    # ── Save / autosave / status ────────────────────────────────

    def _build_sidecar_data(self):
        return {
            "video": Path(self.video_path).name,
            "video_path": str(Path(self.video_path).resolve()),
            "fps": self.fps,
            "total_frames": self.total_frames,
            "duration_sec": round(self.duration_sec, 2),
            "labeled_at": datetime.now().isoformat(timespec="seconds"),
            "labeler_notes": "",
            "labels": self.labels,
        }

    def _save(self):
        data = self._build_sidecar_data()
        p = save_sidecar(self.video_path, data)
        self.last_save_time = time.time()
        self.dirty = False
        print(f"[info] saved {len(self.labels)} labels → {p}")
        self._update_status()

    def _autosave_tick(self):
        if self.dirty:
            self._save()
        self.root.after(AUTO_SAVE_INTERVAL_MS, self._autosave_tick)

    def _update_status(self):
        t_sec = self.frame_idx / self.fps
        n_total = len(self.labels)
        n_left = sum(1 for l in self.labels if l["table"] == "left")
        n_right = n_total - n_left
        n_unconf = sum(1 for l in self.labels
                        if l.get("source") == "v2_auto" and not l.get("confirmed"))
        if self.last_save_time > 0:
            ago = int(time.time() - self.last_save_time)
            saved_str = f"saved {ago}s ago"
        else:
            saved_str = "unsaved"
        if self.dirty:
            saved_str = "● UNSAVED CHANGES (Cmd+S)"
        self.status_var.set(
            f" frame {self.frame_idx:>5d} · {t_sec:6.2f}s / {self.duration_sec:.0f}s   "
            f"·   ACTIVE: {self.active_table.upper()}   "
            f"·   {n_total} labels (L={n_left}, R={n_right}, "
            f"{n_unconf} unconfirmed)   ·   {saved_str}"
        )

    def _open_dialog(self):
        path = filedialog.askopenfilename(
            title="Open video",
            filetypes=[("Video files", "*.mp4 *.mov *.avi *.mkv"), ("All files", "*.*")],
        )
        if path:
            # Save current first if dirty
            if self.dirty:
                if messagebox.askyesno("Save?", "Save current labels before opening new video?"):
                    self._save()
            # Replace state with a new app rooted at same window
            self.cap.release()
            self.video_path = path
            self.cap = cv2.VideoCapture(path)
            self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 25.0
            self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self.duration_sec = self.total_frames / self.fps
            self.frame_idx = 0
            self.labels = []
            self.undo_stack = []
            self.selected_idx = None
            self.last_save_time = 0.0
            self.dirty = False
            self.root.title(f"CH27 GT Labeler — {Path(path).name}")
            self.slider.config(to=max(1, self.total_frames - 1))
            self._load_or_prepopulate_labels()
            self._render_frame()
            self._refresh_label_list()
            self._update_status()

    def _show_help(self):
        msg = (
            "CH27 GT Labeler — Keyboard\n\n"
            "Frame nav:\n"
            "  ←/→            Step ±1 frame\n"
            "  Shift+←/→      Step ±1 second\n"
            "  ↑/↓            Step ±5 seconds\n"
            "  Space          Play/pause\n\n"
            "Labeling:\n"
            "  Tab            Toggle active table (LEFT ↔ RIGHT)\n"
            "  A              Mark LOAD on active table\n"
            "  D              Mark TOSS on active table\n\n"
            "Editing:\n"
            "  Click label    Select; Double-click jumps video\n"
            "  Enter          Edit note on selected label\n"
            "  Backspace      Delete selected label\n"
            "  Cmd/Ctrl+Z     Undo last add/delete\n\n"
            "File:\n"
            "  Cmd/Ctrl+S     Save\n"
            "  O              Open another video\n\n"
            "View:\n"
            "  R              Toggle ROI overlay\n"
            "  ?              Show this dialog\n"
        )
        messagebox.showinfo("Keyboard help", msg)


# ─────────────────────────────────────────────────────────────────
#  Entry point
# ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="CH27 GT Labeling Tool")
    parser.add_argument("video", nargs="?", help="Video file to label")
    args = parser.parse_args()

    if not args.video:
        # Pop a file dialog before creating the app
        root = tk.Tk()
        root.withdraw()
        args.video = filedialog.askopenfilename(
            title="Open video",
            filetypes=[("Video files", "*.mp4 *.mov *.avi *.mkv")],
        )
        root.destroy()
        if not args.video:
            print("No video selected; exiting.")
            return

    if not os.path.exists(args.video):
        print(f"Video not found: {args.video}")
        return

    root = tk.Tk()
    root.configure(bg="#0e1116")
    GTLabeler(root, args.video)
    root.mainloop()


if __name__ == "__main__":
    main()
