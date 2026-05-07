#!/usr/bin/env python3
"""Browser-based CH27 ground-truth labeler.

This keeps the same sidecar schema as gt_labeler.py, but uses the browser's
native MP4 player for video display. It exists because some macOS Tk builds can
create valid image widgets while painting the image area blank.

Usage:
    python3 gt_labeler_web.py gt_clips/gt_clip1_morning.mp4
"""

import argparse
import json
import mimetypes
import os
import threading
import time
import webbrowser
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import cv2

from gt_labeler import get_v2_detections, load_sidecar, save_sidecar, sidecar_path
from taping_counter import (
    HEAP_ROI,
    LEFT_AIR_ROI_V2,
    LEFT_TABLE_ROI_V2,
    RIGHT_AIR_ROI_V2,
    RIGHT_TABLE_ROI_V2,
)


HTML = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>CH27 GT Labeler</title>
  <style>
    * { box-sizing: border-box; }
    body {
      margin: 0;
      background: #0e1116;
      color: #e5e7eb;
      font: 14px/1.35 -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }
    button, input {
      font: inherit;
      color: #f9fafb;
      background: #242933;
      border: 1px solid #38404c;
      border-radius: 6px;
      padding: 7px 10px;
    }
    button { cursor: pointer; }
    button:hover { background: #303744; }
    .toolbar {
      height: 48px;
      display: flex;
      align-items: center;
      gap: 6px;
      padding: 6px 8px;
      border-bottom: 1px solid #202632;
      background: #111720;
    }
    .badge {
      margin-left: 14px;
      padding: 7px 14px;
      border-radius: 6px;
      color: #081016;
      font-weight: 800;
      background: #10b981;
    }
    .badge.right { background: #3b82f6; }
    .shell {
      height: calc(100vh - 76px);
      display: grid;
      grid-template-columns: minmax(640px, 1fr) 340px;
      gap: 0;
      min-height: 0;
    }
    .stage {
      min-width: 0;
      min-height: 0;
      padding: 8px;
      display: grid;
      grid-template-rows: minmax(0, 1fr) 34px;
      gap: 8px;
    }
    .video-wrap {
      position: relative;
      min-height: 0;
      background: #000;
      border: 1px solid #202632;
      overflow: hidden;
      display: flex;
      align-items: center;
      justify-content: center;
    }
    video {
      width: 100%;
      height: 100%;
      object-fit: contain;
      background: #000;
    }
    canvas.overlay {
      position: absolute;
      inset: 0;
      width: 100%;
      height: 100%;
      pointer-events: none;
    }
    .timeline {
      display: flex;
      gap: 8px;
      align-items: center;
      color: #a7b0be;
      font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
    }
    input[type=range] { flex: 1; padding: 0; }
    .panel {
      border-left: 1px solid #202632;
      background: #11141a;
      min-height: 0;
      display: grid;
      grid-template-rows: auto minmax(0, 1fr) auto;
    }
    .panel h2 {
      margin: 0;
      padding: 12px;
      font-size: 13px;
      letter-spacing: 0;
      color: #9ca3af;
      border-bottom: 1px solid #202632;
    }
    .labels {
      overflow: auto;
      padding: 6px;
      font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
    }
    .row {
      display: grid;
      grid-template-columns: 1fr auto;
      gap: 8px;
      padding: 6px 7px;
      border-radius: 5px;
      cursor: pointer;
      color: #d1d5db;
    }
    .row:hover { background: #1b2431; }
    .row.selected { background: #1e3a8a; color: #fff; }
    .left.load { color: #10b981; }
    .left.toss { color: #a7f3d0; }
    .right.load { color: #3b82f6; }
    .right.toss { color: #bfdbfe; }
    .auto:not(.confirmed) { color: #fbbf24; }
    .note {
      border-top: 1px solid #202632;
      padding: 8px;
      display: grid;
      gap: 7px;
    }
    .note input { width: 100%; background: #0e1116; }
    .status {
      height: 28px;
      padding: 5px 8px;
      color: #a7b0be;
      background: #0b0f14;
      border-top: 1px solid #202632;
      font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
      white-space: nowrap;
      overflow: hidden;
      text-overflow: ellipsis;
    }
  </style>
</head>
<body>
  <div class="toolbar">
    <button id="saveBtn">Save (⌘S)</button>
    <button id="playBtn">Play (Space)</button>
    <button id="roiBtn">ROIs: ON (R)</button>
    <button id="helpBtn">Help (?)</button>
    <span id="activeBadge" class="badge">ACTIVE: LEFT</span>
  </div>
  <div class="shell">
    <main class="stage">
      <div class="video-wrap" id="wrap">
        <video id="video" preload="auto" src="/video"></video>
        <canvas id="overlay" class="overlay"></canvas>
      </div>
      <div class="timeline">
        <span id="frameText">f0</span>
        <input id="slider" type="range" min="0" max="1" value="0" />
        <span id="timeText">0.00s</span>
      </div>
    </main>
    <aside class="panel">
      <h2>LABELS</h2>
      <div id="labels" class="labels"></div>
      <div class="note">
        <input id="noteInput" placeholder="Note for selected label" />
        <div>
          <button id="noteBtn">Save Note</button>
          <button id="confirmBtn">Confirm v2</button>
          <button id="deleteBtn">Delete</button>
        </div>
      </div>
    </aside>
  </div>
  <div id="status" class="status">Loading...</div>

<script>
const state = {
  labels: [],
  selected: null,
  activeTable: "left",
  showRois: true,
  dirty: false,
  suggestionsLoading: false,
  metadata: null,
  rois: null
};

const video = document.getElementById("video");
const overlay = document.getElementById("overlay");
const slider = document.getElementById("slider");
const labelsEl = document.getElementById("labels");
const statusEl = document.getElementById("status");
const activeBadge = document.getElementById("activeBadge");
const noteInput = document.getElementById("noteInput");

function fps() { return state.metadata?.fps || 25; }
function currentFrame() { return Math.round(video.currentTime * fps()); }
function frameTime(frame) { return frame / fps(); }
function seekFrame(frame) {
  frame = Math.max(0, Math.min(state.metadata.total_frames - 1, frame));
  video.currentTime = frameTime(frame);
}
function fmtTime(sec) { return `${sec.toFixed(2)}s`; }

async function api(path, opts = {}) {
  const res = await fetch(path, opts);
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

async function loadState() {
  const data = await api("/api/state");
  state.metadata = data.metadata;
  state.rois = data.rois;
  state.labels = data.labels || [];
  state.suggestionsLoading = data.suggestions_loading;
  slider.max = Math.max(1, state.metadata.total_frames - 1);
  renderLabels();
  drawOverlay();
  updateStatus();
}

async function pollSuggestions() {
  const data = await api("/api/state");
  if (!state.dirty && state.labels.length === 0 && data.labels.length > 0) {
    state.labels = data.labels;
    renderLabels();
  }
  state.suggestionsLoading = data.suggestions_loading;
  updateStatus();
  if (state.suggestionsLoading) setTimeout(pollSuggestions, 1000);
}

function renderLabels() {
  labelsEl.innerHTML = "";
  state.labels.sort((a, b) => a.frame - b.frame || a.table.localeCompare(b.table));
  state.labels.forEach((l, i) => {
    const row = document.createElement("div");
    row.className = `row ${l.table} ${l.type} ${l.source === "v2_auto" ? "auto" : ""} ${l.confirmed ? "confirmed" : ""} ${i === state.selected ? "selected" : ""}`;
    const marker = l.source === "v2_auto" && !l.confirmed ? "○" : "●";
    const tag = `${l.table[0].toUpperCase()}${l.type[0].toUpperCase()}`;
    row.innerHTML = `<span>${marker} ${l.time_sec.toFixed(2).padStart(7)}s ${tag} f${String(l.frame).padStart(5)}${l.note ? " - " + escapeHtml(l.note) : ""}</span><span></span>`;
    row.onclick = () => selectLabel(i, false);
    row.ondblclick = () => selectLabel(i, true);
    labelsEl.appendChild(row);
  });
}

function escapeHtml(s) {
  return s.replace(/[&<>"']/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;","\"":"&quot;","'":"&#39;"}[c]));
}

function selectLabel(i, jump) {
  state.selected = i;
  noteInput.value = state.labels[i]?.note || "";
  renderLabels();
  if (jump) seekFrame(state.labels[i].frame);
}

function mark(type) {
  const frame = currentFrame();
  const label = {
    frame,
    time_sec: Number(frameTime(frame).toFixed(3)),
    table: state.activeTable,
    type,
    note: "",
    source: "manual",
    confirmed: true
  };
  state.labels.push(label);
  state.dirty = true;
  renderLabels();
  selectLabel(state.labels.indexOf(label), false);
  updateStatus();
}

function deleteSelected() {
  if (state.selected === null) return;
  state.labels.splice(state.selected, 1);
  state.selected = null;
  noteInput.value = "";
  state.dirty = true;
  renderLabels();
  updateStatus();
}

function confirmSelected() {
  if (state.selected === null) return;
  state.labels[state.selected].confirmed = true;
  state.dirty = true;
  renderLabels();
  updateStatus();
}

function saveNote() {
  if (state.selected === null) return;
  state.labels[state.selected].note = noteInput.value.trim();
  state.dirty = true;
  renderLabels();
  updateStatus();
}

async function saveLabels() {
  const payload = {
    labels: state.labels,
    labeler_notes: ""
  };
  await api("/api/labels", {
    method: "POST",
    headers: {"Content-Type": "application/json"},
    body: JSON.stringify(payload)
  });
  state.dirty = false;
  updateStatus("saved");
}

function toggleTable() {
  state.activeTable = state.activeTable === "left" ? "right" : "left";
  activeBadge.textContent = `ACTIVE: ${state.activeTable.toUpperCase()}`;
  activeBadge.className = `badge ${state.activeTable === "right" ? "right" : ""}`;
  drawOverlay();
  updateStatus();
}

function togglePlay() {
  if (video.paused) video.play();
  else video.pause();
}

function drawOverlay() {
  const rect = video.getBoundingClientRect();
  const wrapRect = document.getElementById("wrap").getBoundingClientRect();
  overlay.width = wrapRect.width;
  overlay.height = wrapRect.height;
  const ctx = overlay.getContext("2d");
  ctx.clearRect(0, 0, overlay.width, overlay.height);
  if (!state.showRois || !state.rois || !state.metadata) return;

  const videoAspect = state.metadata.width / state.metadata.height;
  const boxAspect = wrapRect.width / wrapRect.height;
  let drawW, drawH, offX, offY;
  if (boxAspect > videoAspect) {
    drawH = wrapRect.height;
    drawW = drawH * videoAspect;
    offX = (wrapRect.width - drawW) / 2;
    offY = 0;
  } else {
    drawW = wrapRect.width;
    drawH = drawW / videoAspect;
    offX = 0;
    offY = (wrapRect.height - drawH) / 2;
  }
  const sx = drawW / state.metadata.width;
  const sy = drawH / state.metadata.height;
  const specs = [
    ["L-table", state.rois.left_table, "#10b981"],
    ["R-table", state.rois.right_table, "#3b82f6"],
    ["L-air", state.rois.left_air, "#6ee7b7"],
    ["R-air", state.rois.right_air, "#93c5fd"],
    ["HEAP", state.rois.heap, "#c4c4c4"]
  ];
  for (const [name, roi, color] of specs) {
    const [x1, y1, x2, y2] = roi;
    ctx.strokeStyle = color;
    ctx.lineWidth = 2;
    ctx.strokeRect(offX + x1 * sx, offY + y1 * sy, (x2 - x1) * sx, (y2 - y1) * sy);
    ctx.fillStyle = color;
    ctx.font = "13px sans-serif";
    ctx.fillText(name, offX + x1 * sx + 4, offY + y1 * sy + 15);
  }
}

function updateStatus(extra = "") {
  if (!state.metadata) return;
  const frame = currentFrame();
  slider.value = frame;
  document.getElementById("frameText").textContent = `f${frame}`;
  document.getElementById("timeText").textContent = `${fmtTime(video.currentTime)} / ${Math.round(state.metadata.duration_sec)}s`;
  document.getElementById("playBtn").textContent = video.paused ? "Play (Space)" : "Pause (Space)";
  const left = state.labels.filter(l => l.table === "left").length;
  const right = state.labels.length - left;
  const unconf = state.labels.filter(l => l.source === "v2_auto" && !l.confirmed).length;
  const saved = state.dirty ? "UNSAVED CHANGES" : (extra || "saved");
  const loading = state.suggestionsLoading ? " · loading v2 suggestions" : "";
  statusEl.textContent = `frame ${frame} · ${fmtTime(video.currentTime)} · ${video.paused ? "PAUSED" : "PLAYING"} · ACTIVE: ${state.activeTable.toUpperCase()} · ${state.labels.length} labels (L=${left}, R=${right}, ${unconf} unconfirmed) · ${saved}${loading}`;
}

document.getElementById("saveBtn").onclick = saveLabels;
document.getElementById("playBtn").onclick = togglePlay;
document.getElementById("roiBtn").onclick = () => {
  state.showRois = !state.showRois;
  document.getElementById("roiBtn").textContent = `ROIs: ${state.showRois ? "ON" : "OFF"} (R)`;
  drawOverlay();
};
document.getElementById("helpBtn").onclick = () => alert("Keys: Left/Right frame, Shift+Left/Right 1s, Up/Down 5s, Space play, Tab/T switch table, A load in the lower table ROI, D toss through the upper air ROI, R ROIs, Cmd/Ctrl+S save, Delete selected.");
document.getElementById("noteBtn").onclick = saveNote;
document.getElementById("confirmBtn").onclick = confirmSelected;
document.getElementById("deleteBtn").onclick = deleteSelected;
slider.oninput = () => seekFrame(Number(slider.value));
video.ontimeupdate = () => { updateStatus(); drawOverlay(); };
video.onplay = updateStatus;
video.onpause = updateStatus;
window.onresize = drawOverlay;

document.addEventListener("keydown", e => {
  if (document.activeElement === noteInput && !["Escape"].includes(e.key)) return;
  if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === "s") { e.preventDefault(); saveLabels(); return; }
  if (e.key === " ") { e.preventDefault(); togglePlay(); return; }
  if (e.key === "Tab" || e.key.toLowerCase() === "t") { e.preventDefault(); toggleTable(); return; }
  if (e.key.toLowerCase() === "a") { e.preventDefault(); mark("load"); return; }
  if (e.key.toLowerCase() === "d") { e.preventDefault(); mark("toss"); return; }
  if (e.key.toLowerCase() === "r") { e.preventDefault(); document.getElementById("roiBtn").click(); return; }
  if (e.key === "Backspace" || e.key === "Delete") { e.preventDefault(); deleteSelected(); return; }
  if (e.key === "Enter") { e.preventDefault(); noteInput.focus(); return; }
  if (e.key === "ArrowRight") { e.preventDefault(); seekFrame(currentFrame() + (e.shiftKey ? Math.round(fps()) : 1)); return; }
  if (e.key === "ArrowLeft") { e.preventDefault(); seekFrame(currentFrame() - (e.shiftKey ? Math.round(fps()) : 1)); return; }
  if (e.key === "ArrowUp") { e.preventDefault(); seekFrame(currentFrame() - Math.round(fps() * 5)); return; }
  if (e.key === "ArrowDown") { e.preventDefault(); seekFrame(currentFrame() + Math.round(fps() * 5)); return; }
});

loadState().then(() => {
  pollSuggestions();
  video.currentTime = 0;
  updateStatus();
  drawOverlay();
}).catch(err => {
  statusEl.textContent = err.message;
});
</script>
</body>
</html>
"""


class LabelerState:
    def __init__(self, video_path):
        self.video_path = Path(video_path).resolve()
        self.labels = []
        self.suggestions_loading = False
        self.lock = threading.Lock()

        cap = cv2.VideoCapture(str(self.video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {self.video_path}")
        self.fps = float(cap.get(cv2.CAP_PROP_FPS) or 25.0)
        self.total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        self.width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 1920)
        self.height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 1080)
        cap.release()
        self.duration_sec = self.total_frames / self.fps if self.fps else 0

        existing = load_sidecar(self.video_path)
        if existing and existing.get("labels"):
            self.labels = existing["labels"]
        else:
            self._load_or_start_suggestions()

    def _load_or_start_suggestions(self):
        cache = self.video_path.with_suffix(".v2_detections.json")
        if cache.exists():
            events = json.loads(cache.read_text()).get("events", [])
            self.labels = [self._event_to_label(e) for e in events]
            return

        self.suggestions_loading = True

        def worker():
            events = get_v2_detections(self.video_path, self.fps)
            labels = [self._event_to_label(e) for e in events]
            with self.lock:
                if not self.labels:
                    self.labels = labels
                self.suggestions_loading = False

        threading.Thread(target=worker, daemon=True).start()

    @staticmethod
    def _event_to_label(e):
        return {
            "frame": int(e["frame"]),
            "time_sec": float(e["time_sec"]),
            "table": e["table"],
            "type": "toss",
            "note": "",
            "source": "v2_auto",
            "confirmed": False,
        }

    def payload(self):
        with self.lock:
            labels = list(self.labels)
            loading = self.suggestions_loading
        return {
            "metadata": {
                "video": self.video_path.name,
                "video_path": str(self.video_path),
                "fps": self.fps,
                "total_frames": self.total_frames,
                "duration_sec": round(self.duration_sec, 2),
                "width": self.width,
                "height": self.height,
            },
            "rois": {
                "left_table": LEFT_TABLE_ROI_V2,
                "right_table": RIGHT_TABLE_ROI_V2,
                "left_air": LEFT_AIR_ROI_V2,
                "right_air": RIGHT_AIR_ROI_V2,
                "heap": HEAP_ROI,
            },
            "labels": labels,
            "suggestions_loading": loading,
        }

    def save(self, labels, notes=""):
        with self.lock:
            self.labels = labels
        data = {
            "video": self.video_path.name,
            "video_path": str(self.video_path),
            "fps": self.fps,
            "total_frames": self.total_frames,
            "duration_sec": round(self.duration_sec, 2),
            "labeled_at": datetime.now().isoformat(timespec="seconds"),
            "labeler_notes": notes,
            "labels": labels,
        }
        return save_sidecar(self.video_path, data)


def make_handler(state):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):
            return

        def _json(self, obj, status=200):
            body = json.dumps(obj).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            path = urlparse(self.path).path
            if path == "/":
                body = HTML.encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return
            if path == "/api/state":
                self._json(state.payload())
                return
            if path == "/video":
                self._serve_video()
                return
            self.send_error(404)

        def do_POST(self):
            path = urlparse(self.path).path
            if path != "/api/labels":
                self.send_error(404)
                return
            length = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(length) or b"{}")
            labels = payload.get("labels", [])
            saved = state.save(labels, payload.get("labeler_notes", ""))
            self._json({"ok": True, "path": str(saved), "count": len(labels)})

        def _serve_video(self):
            video = state.video_path
            size = video.stat().st_size
            range_header = self.headers.get("Range")
            content_type = mimetypes.guess_type(video.name)[0] or "video/mp4"

            start, end = 0, size - 1
            status = 200
            if range_header:
                unit, _, rng = range_header.partition("=")
                if unit == "bytes":
                    left, _, right = rng.partition("-")
                    start = int(left) if left else 0
                    end = int(right) if right else size - 1
                    end = min(end, size - 1)
                    status = 206

            length = end - start + 1
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Accept-Ranges", "bytes")
            self.send_header("Content-Length", str(length))
            if status == 206:
                self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
            self.end_headers()
            with video.open("rb") as f:
                f.seek(start)
                remaining = length
                while remaining:
                    chunk = f.read(min(1024 * 1024, remaining))
                    if not chunk:
                        break
                    self.wfile.write(chunk)
                    remaining -= len(chunk)

    return Handler


def main():
    parser = argparse.ArgumentParser(description="Browser-based CH27 GT labeler")
    parser.add_argument("video", help="Video file to label")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args()

    state = LabelerState(args.video)
    server = ThreadingHTTPServer(("127.0.0.1", args.port), make_handler(state))
    url = f"http://127.0.0.1:{args.port}/"
    print(f"Serving labeler at {url}")
    print(f"Labels will save to {sidecar_path(state.video_path)}")
    if not args.no_browser:
        threading.Timer(0.5, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
