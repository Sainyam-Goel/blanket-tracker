#!/usr/bin/env python3
"""Browser-based ROI calibrator for CH27 taping counter.

Draw and adjust ROI rectangles on video frames, export coordinates.

Usage:
    python3 roi_calibrator_web.py gt_clips/gt_clip1_morning.mp4
"""

import argparse
import json
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import cv2

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))

from taping_counter import (
    LEFT_TABLE_ROI_V2, RIGHT_TABLE_ROI_V2,
    LEFT_AIR_ROI_V2, RIGHT_AIR_ROI_V2, HEAP_ROI,
)


HTML = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>CH27 ROI Calibrator</title>
<style>
* { box-sizing: border-box; margin: 0; padding: 0; }
body {
  background: #0a0b10; color: #e2e4e9;
  font: 13px/1.4 -apple-system, BlinkMacSystemFont, sans-serif;
  display: grid; grid-template-rows: auto 1fr; height: 100vh;
}
.toolbar {
  display: flex; gap: 8px; align-items: center; padding: 8px 12px;
  background: #11141c; border-bottom: 1px solid #1e2430; flex-wrap: wrap;
}
.toolbar button, .toolbar select {
  font: inherit; padding: 6px 12px; border-radius: 6px;
  background: #1a1e28; color: #e2e4e9; border: 1px solid #2a3040; cursor: pointer;
}
.toolbar button:hover { background: #242a38; }
.toolbar select { min-width: 140px; }
.spacer { flex: 1; }
.main { display: flex; min-height: 0; overflow: hidden; }
.stage {
  flex: 1; min-width: 0; padding: 12px;
  display: flex; align-items: flex-start; justify-content: center;
  overflow: auto;
}
.stage img {
  max-width: 100%; height: auto; object-fit: contain;
}
.stage-inner { position: relative; display: inline-block; line-height: 0; }
.stage-inner img { display: block; }
.stage-inner canvas { position: absolute; top: 0; left: 0; }
.panel {
  width: 300px; flex-shrink: 0;
  border-left: 1px solid #1e2430; background: #0e1018;
  overflow-y: auto; padding: 12px;
}
.panel h3 {
  font-size: 11px; text-transform: uppercase; letter-spacing: 0.5px;
  color: #6b7a93; margin-bottom: 12px;
}
.roi-card {
  background: #131620; border: 1px solid #1e2430; border-radius: 8px;
  padding: 10px 12px; margin-bottom: 8px; cursor: pointer;
}
.roi-card:hover { border-color: #3b5078; }
.roi-card.selected { border-color: #3b82f6; background: #161e2c; }
.roi-name {
  font-weight: 600; margin-bottom: 4px; font-size: 13px;
}
.roi-coords {
  font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
  font-size: 11px; color: #8b9ab8;
}
.roi-color { display: inline-block; width: 8px; height: 8px; border-radius: 50%; margin-right: 6px; }
.roi-actions { margin-top: 6px; }
.roi-actions button {
  font-size: 11px; padding: 2px 8px; background: #1a1e28;
  color: #e2e4e9; border: 1px solid #2a3040; border-radius: 4px; cursor: pointer;
}
.roi-actions button:hover { background: #ef4444; border-color: #ef4444; }
.export-area {
  margin-top: 16px; padding-top: 12px; border-top: 1px solid #1e2430;
}
.export-area pre {
  background: #080a10; border: 1px solid #1e2430; border-radius: 6px;
  padding: 10px; font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
  font-size: 11px; color: #34d399; overflow-x: auto; white-space: pre-wrap;
  max-height: 200px; overflow-y: auto;
}
.hint {
  font-size: 11px; color: #6b7a93; margin-top: 8px; line-height: 1.6;
}
</style>
</head>
<body>
<div class="toolbar">
  <select id="frameSelect" onchange="loadFrame()">
    <option value="0">Frame 0 (0s)</option>
    <option value="1000">Frame 1000 (40s)</option>
    <option value="2500">Frame 2500 (100s)</option>
    <option value="3750" selected>Frame 3750 (150s)</option>
    <option value="5000">Frame 5000 (200s)</option>
    <option value="6250">Frame 6250 (250s)</option>
  </select>
  <button onclick="loadFrame()">Reload</button>
  <span class="spacer"></span>
  <button id="btnCopy" onclick="copyToClipboard()">Copy Code</button>
  <button id="btnSave" onclick="saveROIs()" style="background:#10b981;border-color:#10b981;">Save JSON</button>
</div>
<div class="main">
  <div class="stage">
    <div class="stage-inner">
      <img id="frameImg" src="/frame?f=3750" />
      <canvas id="overlayCanvas"></canvas>
    </div>
  </div>
  <div class="panel">
    <h3>ROI Rectangles</h3>
    <div id="roiList"></div>
    <div class="export-area">
      <h3>Export</h3>
      <pre id="exportText"></pre>
      <button style="margin-top:8px;width:100%;" onclick="copyToClipboard()">Copy to Clipboard</button>
    </div>
    <div class="hint">
      <b>Click+drag</b> empty area to draw new rect<br>
      <b>Click</b> rect or card to select<br>
      <b>Drag corners</b> (white squares) to resize<br>
      <b>Drag inside</b> to move selected rect<br>
      <b>Delete</b> to remove selected
    </div>
  </div>
</div>

<script>
var rois = %%ROIS_JSON%%;
var selected = null;
var drawing = false, dragging = false, dragCorner = -1;
var dragStart = null, dragOrig = null, dragCurrent = null;

var colors = {
  left_table: "#10b981", right_table: "#3b82f6",
  left_air: "#6ee7b7", right_air: "#93c5fd", heap: "#9ca3af"
};

function $(id) { return document.getElementById(id); }

function syncCanvas() {
  var img = $("frameImg");
  var cv = $("overlayCanvas");
  cv.width = img.naturalWidth;
  cv.height = img.naturalHeight;
  cv.style.width = img.clientWidth + "px";
  cv.style.height = img.clientHeight + "px";
}

function renderCards() {
  var list = $("roiList"), html = "";
  for (var name in rois) {
    var r = rois[name];
    var c = colors[name] || "#fff";
    html += '<div class="roi-card' + (selected === name ? " selected" : "") + '" onclick="selectROI(\'' + name + '\')">';
    html += '<div class="roi-name"><span class="roi-color" style="background:' + c + '"></span>' + name.toUpperCase() + '</div>';
    html += '<div class="roi-coords">(' + r[0] + ", " + r[1] + ") &rarr; (" + r[2] + ", " + r[3] + ")</div>";
    html += '<div class="roi-actions"><button onclick="event.stopPropagation();deleteROI(\'' + name + '\')">Delete</button></div>';
    html += '</div>';
  }
  list.innerHTML = html;
  updateExport();
}

function selectROI(name) { selected = name; renderCards(); drawRois(); }
function deleteROI(name) { if (confirm("Delete " + name + "?")) { delete rois[name]; if (selected === name) selected = null; renderCards(); drawRois(); } }

function drawRois() {
  var img = $("frameImg");
  var cv = $("overlayCanvas");
  syncCanvas();
  var ctx = cv.getContext("2d");
  ctx.clearRect(0, 0, cv.width, cv.height);

  for (var name in rois) {
    var r = rois[name], c = colors[name] || "#aaa", sel = name === selected;
    ctx.fillStyle = c + "1A";
    ctx.fillRect(r[0], r[1], r[2]-r[0], r[3]-r[1]);
    ctx.strokeStyle = c;
    ctx.lineWidth = sel ? 2.5 : 2;
    ctx.strokeRect(r[0], r[1], r[2]-r[0], r[3]-r[1]);
    ctx.fillStyle = c;
    ctx.font = "bold 11px sans-serif";
    ctx.fillText(name.toUpperCase(), r[0] + 4, r[1] + 15);

    if (sel) {
      var h = [[r[0],r[1]],[r[2],r[1]],[r[0],r[3]],[r[2],r[3]]];
      for (var i = 0; i < 4; i++) {
        ctx.fillStyle = "#fff"; ctx.strokeStyle = c; ctx.lineWidth = 1.5;
        ctx.fillRect(h[i][0]-4, h[i][1]-4, 8, 8); ctx.strokeRect(h[i][0]-4, h[i][1]-4, 8, 8);
      }
    }
  }

  if (drawing && dragStart && dragCurrent) {
    var x1 = Math.min(dragStart.x, dragCurrent.x), y1 = Math.min(dragStart.y, dragCurrent.y);
    var x2 = Math.max(dragStart.x, dragCurrent.x), y2 = Math.max(dragStart.y, dragCurrent.y);
    ctx.strokeStyle = "#fbbf24"; ctx.lineWidth = 2;
    ctx.setLineDash([6, 4]); ctx.strokeRect(x1, y1, x2-x1, y2-y1); ctx.setLineDash([]);
  }
}

function coords(e) {
  var img = $("frameImg");
  var r = img.getBoundingClientRect();
  return {
    x: Math.round((e.clientX - r.left) * img.naturalWidth / r.width),
    y: Math.round((e.clientY - r.top) * img.naturalHeight / r.height)
  };
}

function cornerAt(cx, cy) {
  if (!selected || !rois[selected]) return -1;
  var r = rois[selected];
  var ps = [[r[0],r[1]],[r[2],r[1]],[r[0],r[3]],[r[2],r[3]]];
  for (var i = 0; i < 4; i++)
    if (Math.abs(cx - ps[i][0]) <= 10 && Math.abs(cy - ps[i][1]) <= 10) return i;
  return -1;
}

function inside(cx, cy, r) {
  return cx >= r[0] && cx <= r[2] && cy >= r[1] && cy <= r[3];
}

var cv = $("overlayCanvas");

cv.addEventListener("mousedown", function(e) {
  var p = coords(e);
  var c = cornerAt(p.x, p.y);
  if (c >= 0) {
    dragging = true; dragCorner = c;
    dragStart = p; dragOrig = rois[selected].slice();
    return;
  }
  for (var name in rois) {
    if (inside(p.x, p.y, rois[name])) {
      selected = name; renderCards(); drawRois();
      dragging = true; dragCorner = -1;
      dragStart = p; dragOrig = rois[name].slice();
      return;
    }
  }
  drawing = true; dragStart = p; dragCurrent = p; selected = null; renderCards();
});

cv.addEventListener("mousemove", function(e) {
  dragCurrent = coords(e);
  if (!drawing && !dragging) return;
  if (drawing) { drawRois(); return; }
  if (dragging && selected && rois[selected]) {
    var dx = dragCurrent.x - dragStart.x, dy = dragCurrent.y - dragStart.y;
    var r = dragOrig.slice();
    if (dragCorner === 0) { r[0] += dx; r[1] += dy; }
    else if (dragCorner === 1) { r[2] += dx; r[1] += dy; }
    else if (dragCorner === 2) { r[0] += dx; r[3] += dy; }
    else if (dragCorner === 3) { r[2] += dx; r[3] += dy; }
    else { r[0] += dx; r[1] += dy; r[2] += dx; r[3] += dy; }
    rois[selected] = r;
    drawRois(); renderCards();
  }
});

cv.addEventListener("mouseup", function() {
  if (drawing) {
    drawing = false;
    var x1 = Math.min(dragStart.x, dragCurrent.x), y1 = Math.min(dragStart.y, dragCurrent.y);
    var x2 = Math.max(dragStart.x, dragCurrent.x), y2 = Math.max(dragStart.y, dragCurrent.y);
    if (Math.abs(x2-x1) > 8 && Math.abs(y2-y1) > 8) {
      var name = prompt("ROI name:", "");
      if (name) { rois[name] = [x1, y1, x2, y2]; selected = name; }
    }
    dragStart = null; dragCurrent = null;
    renderCards(); drawRois();
  }
  if (dragging) { dragging = false; dragCorner = -1; updateExport(); }
});

document.addEventListener("keydown", function(e) {
  if (e.key === "Delete" && selected && !e.target.closest("input,textarea")) deleteROI(selected);
});

function updateExport() {
  var lines = [];
  for (var name in rois)
    lines.push(name.toUpperCase() + "_ROI_V2 = (" + rois[name].join(", ") + ")");
  $("exportText").textContent = lines.join("\n");
}

function copyToClipboard() {
  navigator.clipboard.writeText($("exportText").textContent).then(function() {
    $("btnCopy").textContent = "Copied!";
    setTimeout(function() { $("btnCopy").textContent = "Copy Code"; }, 1500);
  });
}

function saveROIs() {
  fetch("/api/save", {
    method: "POST",
    headers: {"Content-Type": "application/json"},
    body: JSON.stringify({rois: rois})
  }).then(function() {
    $("btnSave").textContent = "Saved!";
    setTimeout(function() { $("btnSave").textContent = "Save JSON"; }, 1500);
  });
}

function loadFrame() {
  $("frameImg").src = "/frame?f=" + $("frameSelect").value;
}

$("frameImg").addEventListener("load", function() { syncCanvas(); drawRois(); });
window.addEventListener("resize", function() { if ($("frameImg").naturalWidth) { syncCanvas(); drawRois(); } });
renderCards();
</script>
</body>
</html>
"""


def serve_frame(video_path, frame_num):
    cap = cv2.VideoCapture(str(video_path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        return None
    _, jpg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return jpg.tobytes()


def make_handler(video_path):
    rois_state = {
        "left_table":  list(LEFT_TABLE_ROI_V2),
        "right_table": list(RIGHT_TABLE_ROI_V2),
        "left_air":    list(LEFT_AIR_ROI_V2),
        "right_air":   list(RIGHT_AIR_ROI_V2),
        "heap":        list(HEAP_ROI),
    }
    html = HTML.replace("%%ROIS_JSON%%", json.dumps(rois_state))

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
            qs = parse_qs(urlparse(self.path).query)
            if path == "/":
                body = html.encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return
            if path == "/frame":
                f = int(qs.get("f", ["0"])[0])
                data = serve_frame(video_path, f)
                if data:
                    self.send_response(200)
                    self.send_header("Content-Type", "image/jpeg")
                    self.send_header("Content-Length", str(len(data)))
                    self.end_headers()
                    self.wfile.write(data)
                else:
                    self.send_error(404)
                return
            self.send_error(404)

        def do_POST(self):
            path = urlparse(self.path).path
            if path == "/api/save":
                length = int(self.headers.get("Content-Length", "0"))
                payload = json.loads(self.rfile.read(length) or "{}")
                new_rois = payload.get("rois", {})
                out = {}
                for name, rect in new_rois.items():
                    out[name] = [int(round(v)) for v in rect]
                out_path = Path(video_path).with_suffix(".rois.json")
                out_path.write_text(json.dumps(out, indent=2))
                self._json({"ok": True, "path": str(out_path), "rois": out})
                return
            self.send_error(404)

    return Handler


def main():
    parser = argparse.ArgumentParser(description="Browser-based CH27 ROI calibrator")
    parser.add_argument("video", help="Video file")
    parser.add_argument("--port", type=int, default=8780)
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args()

    if not Path(args.video).exists():
        print(f"Video not found: {args.video}")
        sys.exit(1)

    server = ThreadingHTTPServer(("127.0.0.1", args.port), make_handler(args.video))
    url = f"http://127.0.0.1:{args.port}/"
    print(f"ROI Calibrator at {url}")
    print(f"Save to: {Path(args.video).with_suffix('.rois.json')}")
    if not args.no_browser:
        import webbrowser, threading
        threading.Timer(0.5, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nDone.")


if __name__ == "__main__":
    main()
