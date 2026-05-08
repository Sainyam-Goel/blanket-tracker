#!/usr/bin/env python3
"""Generate the unified Blanket Tracker dashboard — one HTML, three camera tabs.

Reads:
    cutting/cutting_fullday.json     (CH19)
    passing/blanket_fullday.json     (CH21)
    taping/taping_fullday_v11.json   (CH27)

Writes:
    index.html    (the live root for sainyam-goel.github.io/blanket-tracker/)

Each camera has its own full-page view (KPIs + charts + tables) inside the
single HTML; a tab bar at the top swaps which page is visible. Styling is
the CH27-dark-card pattern the user prefers.

Usage:
    python3 generate_unified_dashboard.py
"""
from __future__ import annotations

import json
import os
import sys
from datetime import datetime
from pathlib import Path

BASE = Path(__file__).parent
CH19_JSON = BASE / "cutting" / "cutting_fullday.json"
CH21_JSON = BASE / "passing" / "blanket_fullday.json"
CH27_JSON = Path(os.environ.get("CH27_JSON", str(BASE / "taping" / "taping_fullday_v11.json")))
OUTPUT_HTML = BASE / "index.html"

# Production dates per camera (from NVR filenames). Could be parsed automatically
# but they don't change often and the user prefers an explicit value.
CH19_DATE = "27/02/2026"
CH21_DATE = "27/02/2026"
CH27_DATE = "28/04/2026"

CH19_START_HOUR = 11
CH21_START_HOUR = 11
CH27_START_HOUR = 9


# ──────────────────────────────────────────────────────────────────
# Data loaders
# ──────────────────────────────────────────────────────────────────

def load_ch19():
    if not CH19_JSON.exists():
        return None
    data = json.loads(CH19_JSON.read_text())
    fd = data.get("frame_data", [])
    step = max(1, len(fd) // 1500)
    # Slim events to fields the dashboard uses
    evt_keep = {"type", "frame", "time_sec", "confidence", "peak_deriv", "segment"}
    events = [{k: e[k] for k in evt_keep if k in e} for e in data.get("events", [])]
    fd_keep = {"frame", "time_sec", "brightness", "smoothed", "derivative", "in_break"}
    fd_slim = [{k: r[k] for k in fd_keep if k in r} for r in fd[::step]]
    return {
        "metadata": data.get("metadata", {}),
        "summary": data.get("summary", {}),
        "config": data.get("config", {}),
        "segments": data.get("segments", []),
        "events": events,
        "breaks": data.get("breaks", []),
        "frame_data": fd_slim,
        "date": CH19_DATE,
        "start_clock_hour": CH19_START_HOUR,
    }


def load_ch21():
    if not CH21_JSON.exists():
        return None
    raw = json.loads(CH21_JSON.read_text())
    all_events = []
    all_frames = []
    total_dur = 0.0
    first_cfg = None
    seg_info = []
    for i, v in enumerate(raw.get("videos", [])):
        offset = v.get("time_offset_sec", total_dur)
        all_events.extend(v.get("events", []))
        frames = v.get("frames", [])
        step = max(1, len(frames) // 200)
        all_frames.extend(frames[::step])
        seg_dur = v.get("video_info", {}).get("duration_sec", 0)
        seg_results = v.get("results", {})
        seg_info.append({
            "index": i,
            "offset_sec": round(offset, 2),
            "duration_sec": round(seg_dur, 2),
            "accepted": seg_results.get("accepted", 0),
            "rejected": seg_results.get("rejected", 0),
        })
        if first_cfg is None:
            first_cfg = v.get("detection_config", {})
        total_dur += seg_dur
    all_events.sort(key=lambda e: e.get("time_sec", 0))
    all_frames.sort(key=lambda f: f.get("time_sec", 0))
    # Slim event fields
    evt_keep = {"type", "time_sec", "frame", "diff", "note"}
    events_slim = [{k: e[k] for k in evt_keep if k in e} for e in all_events]
    fr_keep = {"frame", "time_sec", "scale_diff", "scale_state", "table_texture", "table_state"}
    frames_slim = [{k: r[k] for k in fr_keep if k in r} for r in all_frames]
    return {
        "video_info": {
            "duration_sec": total_dur,
            "fps": 25.0,
            "total_segments": len(raw.get("videos", [])),
        },
        "detection_config": first_cfg or {},
        "results": {
            "accepted": raw.get("total_accepted", 0),
            "rejected": raw.get("total_rejected", 0),
            "total_blankets": raw.get("total_blankets", 0),
            "table_blanket_off": raw.get("total_table_blanket_off", 0),
        },
        "segments": seg_info,
        "events": events_slim,
        "frames": frames_slim,
        "date": CH21_DATE,
        "start_clock_hour": CH21_START_HOUR,
    }


def load_ch27():
    if not CH27_JSON.exists():
        return None
    data = json.loads(CH27_JSON.read_text())
    keep = {
        "table", "time_sec", "frame", "cycle_duration_sec", "air_motion_peak",
        "v4_prob", "v4_eff_thresh", "long_cycle", "via_overlap_detector",
        "via_secondary_path", "segment", "blob_max_area",
    }
    slim = [{k: e[k] for k in keep if k in e} for e in data.get("events", [])]
    return {
        "metadata": data.get("metadata", {}),
        "summary": data.get("summary", {}),
        "segments": data.get("segments", []),
        "events": slim,
        "suppressed_count": data.get("summary", {}).get("suppressed_count",
                                                          len(data.get("suppressed_candidates", []))),
        "date": CH27_DATE,
        "start_clock_hour": CH27_START_HOUR,
    }


# ──────────────────────────────────────────────────────────────────
# HTML template — single file with three camera pages and a tab bar
# ──────────────────────────────────────────────────────────────────

TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Blanket Production Tracker — Panipat Factory</title>
<link href="https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
<style>
  :root {
    --bg: #0a0a0f;
    --surface: #12121a;
    --card: #1a1a26;
    --border: #2a2a3d;
    --text: #e8e8f0;
    --muted: #6b7280;
    --green: #10b981;
    --red:   #ef4444;
    /* Camera accents */
    --ch19: #f59e0b;
    --ch21: #3b82f6;
    --ch27-l: #7c3aed;
    --ch27-r: #06b6d4;
  }
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body {
    background: var(--bg); color: var(--text);
    font-family: 'Syne', sans-serif; min-height: 100vh; overflow-x: hidden;
  }
  body::before {
    content: ''; position: fixed; inset: 0; pointer-events: none; z-index: 0;
    background-image:
      linear-gradient(rgba(124,58,237,0.04) 1px, transparent 1px),
      linear-gradient(90deg, rgba(6,182,212,0.04) 1px, transparent 1px);
    background-size: 40px 40px;
  }
  .container {
    position: relative; z-index: 1;
    max-width: 1400px; margin: 0 auto; padding: 2rem;
  }

  /* ── Header ───────────────────────────────────────── */
  header {
    display: flex; align-items: flex-start; justify-content: space-between;
    flex-wrap: wrap; gap: 1rem;
    margin-bottom: 1.5rem; padding-bottom: 1.5rem;
    border-bottom: 1px solid var(--border);
  }
  .logo-area h1 {
    font-size: 2.1rem; font-weight: 800; letter-spacing: -0.04em;
    background: linear-gradient(135deg, #f59e0b, #3b82f6, #7c3aed, #06b6d4);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text;
  }
  .logo-area p {
    color: var(--muted); font-size: 0.85rem;
    font-family: 'JetBrains Mono', monospace; margin-top: 0.3rem;
  }
  .header-badges { display: flex; gap: 0.5rem; flex-wrap: wrap; }
  .duration-badge {
    display: flex; align-items: center; gap: 0.5rem;
    padding: 0.4rem 0.9rem; border-radius: 100px;
    font-size: 0.75rem; font-family: 'JetBrains Mono', monospace;
    background: rgba(232,232,240,0.05);
    border: 1px solid rgba(232,232,240,0.2);
    color: var(--text); font-weight: 600;
  }

  /* ── Tab bar ──────────────────────────────────────── */
  .tab-bar {
    display: flex; gap: 0.5rem;
    margin-bottom: 1.5rem;
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 14px; padding: 0.5rem;
  }
  .tab {
    flex: 1; padding: 0.9rem 1rem;
    background: transparent; border: 1px solid transparent;
    border-radius: 10px; cursor: pointer;
    color: var(--muted);
    font-family: 'JetBrains Mono', monospace; font-size: 0.85rem; font-weight: 600;
    text-transform: uppercase; letter-spacing: 0.05em;
    display: flex; align-items: center; justify-content: center; gap: 0.6rem;
    transition: color 0.2s, background 0.2s, border-color 0.2s;
  }
  .tab:hover { color: var(--text); background: rgba(255,255,255,0.03); }
  .tab .tab-dot {
    width: 8px; height: 8px; border-radius: 50%; background: currentColor;
  }
  .tab .tab-count {
    font-size: 0.7rem; color: var(--muted);
    font-weight: 400; letter-spacing: 0;
  }
  .tab[data-camera="ch19"].active { color: var(--ch19); background: rgba(245,158,11,0.10); border-color: rgba(245,158,11,0.35); }
  .tab[data-camera="ch21"].active { color: var(--ch21); background: rgba(59,130,246,0.10); border-color: rgba(59,130,246,0.35); }
  .tab[data-camera="ch27"].active { color: var(--ch27-l); background: rgba(124,58,237,0.10); border-color: rgba(124,58,237,0.35); }
  .tab[data-camera="ch19"].active .tab-count { color: var(--ch19); }
  .tab[data-camera="ch21"].active .tab-count { color: var(--ch21); }
  .tab[data-camera="ch27"].active .tab-count { color: var(--ch27-l); }

  /* ── Camera page ──────────────────────────────────── */
  .camera-page { display: none; }
  .camera-page.active { display: block; animation: fadeIn 0.25s ease; }
  @keyframes fadeIn { from { opacity: 0; transform: translateY(4px); } to { opacity: 1; transform: translateY(0); } }

  .page-header {
    display: flex; align-items: flex-start; justify-content: space-between;
    flex-wrap: wrap; gap: 1rem; margin-bottom: 1.5rem;
  }
  .page-title {
    font-size: 1.4rem; font-weight: 700; letter-spacing: -0.02em;
  }
  .page-subtitle {
    color: var(--muted); font-size: 0.85rem;
    font-family: 'JetBrains Mono', monospace; margin-top: 0.25rem;
  }

  /* Themed accents per page */
  .camera-page[data-camera="ch19"] .page-title { color: var(--ch19); }
  .camera-page[data-camera="ch21"] .page-title { color: var(--ch21); }
  .camera-page[data-camera="ch27"] .page-title { background: linear-gradient(135deg, #7c3aed, #06b6d4); -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text; }

  /* ── KPI grid ─────────────────────────────────────── */
  .kpi-grid {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 1rem; margin-bottom: 1.5rem;
  }
  .kpi-card {
    background: var(--surface); border: 1px solid var(--border);
    border-radius: 14px; padding: 1.2rem;
    position: relative; overflow: hidden;
    transition: transform 0.2s, border-color 0.2s;
  }
  .kpi-card:hover { transform: translateY(-2px); }
  .kpi-card::before {
    content: ''; position: absolute; top: 0; left: 0; right: 0; height: 2px;
    opacity: 0.7;
  }
  .camera-page[data-camera="ch19"] .kpi-card::before { background: linear-gradient(90deg, #f59e0b, #fbbf24); }
  .camera-page[data-camera="ch19"] .kpi-card:hover { border-color: rgba(245,158,11,0.4); }
  .camera-page[data-camera="ch21"] .kpi-card::before { background: linear-gradient(90deg, #3b82f6, #06b6d4); }
  .camera-page[data-camera="ch21"] .kpi-card:hover { border-color: rgba(59,130,246,0.4); }
  .camera-page[data-camera="ch27"] .kpi-card::before { background: linear-gradient(90deg, #7c3aed, #06b6d4); }
  .camera-page[data-camera="ch27"] .kpi-card:hover { border-color: rgba(124,58,237,0.4); }
  .camera-page[data-camera="ch27"] .kpi-card.left::before  { background: linear-gradient(90deg, #7c3aed, #a78bfa); }
  .camera-page[data-camera="ch27"] .kpi-card.right::before { background: linear-gradient(90deg, #06b6d4, #22d3ee); }
  .kpi-card.green::before { background: linear-gradient(90deg, #10b981, #059669) !important; }
  .kpi-card.red::before   { background: linear-gradient(90deg, #ef4444, #dc2626) !important; }

  .kpi-label {
    font-size: 0.7rem; text-transform: uppercase; letter-spacing: 0.1em;
    color: var(--muted); font-family: 'JetBrains Mono', monospace; margin-bottom: 0.5rem;
  }
  .kpi-value {
    font-size: 2.2rem; font-weight: 800; letter-spacing: -0.04em; line-height: 1;
    color: var(--text);
  }
  .kpi-value.amber { color: var(--ch19); }
  .kpi-value.blue  { color: var(--ch21); }
  .kpi-value.left  { color: var(--ch27-l); }
  .kpi-value.right { color: var(--ch27-r); }
  .kpi-value.green { color: var(--green); }
  .kpi-value.red   { color: var(--red); }
  .kpi-sub {
    font-size: 0.7rem; color: var(--muted);
    font-family: 'JetBrains Mono', monospace; margin-top: 0.4rem;
  }

  /* ── Panels ───────────────────────────────────────── */
  .panel {
    background: var(--surface); border: 1px solid var(--border);
    border-radius: 14px; padding: 1.5rem; margin-bottom: 1.5rem;
  }
  .panel-header {
    display: flex; justify-content: space-between; align-items: center;
    margin-bottom: 1rem; padding-bottom: 0.8rem;
    border-bottom: 1px solid var(--border);
  }
  .panel-title {
    font-size: 0.85rem; text-transform: uppercase; letter-spacing: 0.08em;
    font-family: 'JetBrains Mono', monospace; font-weight: 600; color: var(--text);
    display: flex; align-items: center; gap: 0.5rem;
  }
  .panel-title .dot {
    width: 8px; height: 8px; border-radius: 50%;
  }
  .camera-page[data-camera="ch19"] .panel-title .dot { background: var(--ch19); }
  .camera-page[data-camera="ch21"] .panel-title .dot { background: var(--ch21); }
  .camera-page[data-camera="ch27"] .panel-title .dot { background: var(--ch27-l); }
  .panel-tag {
    font-size: 0.7rem; font-family: 'JetBrains Mono', monospace; color: var(--muted);
    background: rgba(124,58,237,0.05);
    padding: 0.25rem 0.7rem; border-radius: 100px;
    border: 1px solid rgba(124,58,237,0.15);
  }
  canvas { width: 100%; display: block; }

  .panel-grid-2 {
    display: grid; grid-template-columns: 1fr 1fr;
    gap: 1.5rem; margin-bottom: 1.5rem;
  }

  /* ── Tables ───────────────────────────────────────── */
  table { width: 100%; border-collapse: collapse; }
  thead th {
    text-align: left; padding: 0.7rem 0.8rem;
    font-size: 0.7rem; text-transform: uppercase; letter-spacing: 0.08em;
    font-family: 'JetBrains Mono', monospace; color: var(--muted);
    border-bottom: 1px solid var(--border); font-weight: 600;
  }
  thead th.num { text-align: right; }
  tbody td {
    padding: 0.6rem 0.8rem; font-size: 0.9rem;
    border-bottom: 1px solid rgba(42,42,61,0.4);
    font-family: 'JetBrains Mono', monospace;
  }
  tbody td.num { text-align: right; }
  tbody td.label { color: var(--text); font-weight: 500; }
  .camera-page[data-camera="ch19"] tbody tr:hover { background: rgba(245,158,11,0.04); }
  .camera-page[data-camera="ch21"] tbody tr:hover { background: rgba(59,130,246,0.04); }
  .camera-page[data-camera="ch27"] tbody tr:hover { background: rgba(124,58,237,0.04); }
  .bar-cell { padding: 0.6rem 0.8rem; min-width: 200px; }
  .bar-row {
    display: flex; height: 18px; border-radius: 4px; overflow: hidden;
    background: rgba(42,42,61,0.3);
  }
  .bar-l { background: linear-gradient(90deg, #7c3aed, #a78bfa); }
  .bar-r { background: linear-gradient(90deg, #06b6d4, #22d3ee); }
  .bar-acc { background: linear-gradient(90deg, #10b981, #059669); }
  .bar-rej { background: linear-gradient(90deg, #ef4444, #dc2626); }
  .bar-amber { background: linear-gradient(90deg, #f59e0b, #d97706); }

  /* ── Model banner (CH27) ──────────────────────────── */
  .model-banner {
    display: flex; flex-wrap: wrap; gap: 0.6rem 1.2rem; align-items: center;
    background: rgba(124,58,237,0.05);
    border: 1px solid rgba(124,58,237,0.2);
    border-radius: 12px; padding: 0.9rem 1.2rem;
    margin-bottom: 1.5rem;
    font-family: 'JetBrains Mono', monospace; font-size: 0.78rem;
    color: var(--text);
  }
  .model-banner .pill {
    background: rgba(16,185,129,0.12); color: var(--green);
    padding: 0.18rem 0.55rem; border-radius: 100px;
    font-weight: 600; border: 1px solid rgba(16,185,129,0.25);
  }
  .model-banner .pill.muted { background: rgba(107,114,128,0.12); color: var(--muted); border-color: rgba(107,114,128,0.25); }
  .model-banner .label { color: var(--muted); margin-right: 0.3rem; }

  /* ── Footer ───────────────────────────────────────── */
  footer {
    text-align: center; padding: 2rem 0 1rem;
    color: var(--muted); font-size: 0.75rem;
    font-family: 'JetBrains Mono', monospace;
    border-top: 1px solid var(--border); margin-top: 2rem;
  }

  @media (max-width: 900px) {
    .kpi-grid { grid-template-columns: repeat(2, 1fr); }
    .panel-grid-2 { grid-template-columns: 1fr; }
    .tab { font-size: 0.75rem; padding: 0.7rem 0.5rem; }
  }
</style>
</head>
<body>
<div class="container">
  <header>
    <div class="logo-area">
      <h1>BLANKET PRODUCTION TRACKER</h1>
      <p>Panipat factory · Mr. Goyal · CV pipeline across 3 cameras</p>
    </div>
    <div class="header-badges">
      <div class="duration-badge" id="badge-summary">--</div>
    </div>
  </header>

  <!-- Tab bar -->
  <div class="tab-bar" id="tab-bar">
    <button class="tab" data-camera="ch27">
      <span class="tab-dot"></span>
      CH27 · Taping
      <span class="tab-count" id="tab-count-ch27"></span>
    </button>
    <button class="tab" data-camera="ch21">
      <span class="tab-dot"></span>
      CH21 · Passing
      <span class="tab-count" id="tab-count-ch21"></span>
    </button>
    <button class="tab" data-camera="ch19">
      <span class="tab-dot"></span>
      CH19 · Cutting
      <span class="tab-count" id="tab-count-ch19"></span>
    </button>
  </div>

  <!-- ────────────────────────  CH19 PAGE  ──────────────────────── -->
  <div class="camera-page" data-camera="ch19">
    <div class="page-header">
      <div>
        <div class="page-title">CH19 · Cutting Table</div>
        <div class="page-subtitle">4-worker brightness-derivative cut counter · multi-scale d35+d25 · echo suppression</div>
      </div>
      <div class="header-badges">
        <div class="duration-badge" id="ch19-date">--</div>
        <div class="duration-badge" id="ch19-duration">--</div>
        <div class="duration-badge" id="ch19-version">--</div>
      </div>
    </div>

    <div class="kpi-grid">
      <div class="kpi-card">
        <div class="kpi-label">Total Cuts</div>
        <div class="kpi-value amber" id="ch19-kpi-total">0</div>
        <div class="kpi-sub" id="ch19-kpi-total-sub">over the day</div>
      </div>
      <div class="kpi-card">
        <div class="kpi-label">Cuts / Min</div>
        <div class="kpi-value" id="ch19-kpi-rate">0</div>
        <div class="kpi-sub">active time only</div>
      </div>
      <div class="kpi-card">
        <div class="kpi-label">Avg Cycle</div>
        <div class="kpi-value" id="ch19-kpi-cycle">0</div>
        <div class="kpi-sub">seconds between cuts</div>
      </div>

      <div class="kpi-card green">
        <div class="kpi-label">High Confidence</div>
        <div class="kpi-value green" id="ch19-kpi-high">0</div>
        <div class="kpi-sub" id="ch19-kpi-high-sub">% of total</div>
      </div>
      <div class="kpi-card">
        <div class="kpi-label">Active Time</div>
        <div class="kpi-value" id="ch19-kpi-active">0</div>
        <div class="kpi-sub" id="ch19-kpi-active-sub">working hours</div>
      </div>
      <div class="kpi-card">
        <div class="kpi-label">Breaks</div>
        <div class="kpi-value" id="ch19-kpi-breaks">0</div>
        <div class="kpi-sub" id="ch19-kpi-breaks-sub">idle min total</div>
      </div>
    </div>

    <div class="panel">
      <div class="panel-header">
        <div class="panel-title"><span class="dot"></span> Cumulative Cuts Over the Day</div>
        <div class="panel-tag">Running total · break gaps shaded</div>
      </div>
      <canvas id="ch19-chart-cumulative" style="height: 300px;"></canvas>
    </div>

    <div class="panel-grid-2">
      <div class="panel">
        <div class="panel-header">
          <div class="panel-title"><span class="dot"></span> Hourly Cuts</div>
          <div class="panel-tag">Per clock hour</div>
        </div>
        <canvas id="ch19-chart-hourly" style="height: 280px;"></canvas>
      </div>
      <div class="panel">
        <div class="panel-header">
          <div class="panel-title"><span class="dot"></span> Confidence Mix</div>
          <div class="panel-tag">High / medium / low per hour</div>
        </div>
        <canvas id="ch19-chart-confidence" style="height: 280px;"></canvas>
      </div>
    </div>

    <div class="panel">
      <div class="panel-header">
        <div class="panel-title"><span class="dot"></span> Brightness Signal</div>
        <div class="panel-tag">Smoothed table-brightness · cuts marked at peaks</div>
      </div>
      <canvas id="ch19-chart-signal" style="height: 220px;"></canvas>
    </div>

    <div class="panel">
      <div class="panel-header">
        <div class="panel-title"><span class="dot"></span> Hourly Breakdown</div>
        <div class="panel-tag" id="ch19-hourly-tag">Per-hour stats</div>
      </div>
      <table>
        <thead>
          <tr>
            <th>Hour</th>
            <th class="num">Cuts</th>
            <th class="num">High</th>
            <th class="num">Med</th>
            <th class="num">Low</th>
            <th class="num">High %</th>
            <th class="bar-cell">Distribution</th>
          </tr>
        </thead>
        <tbody id="ch19-hourly-tbody"></tbody>
      </table>
    </div>
  </div>

  <!-- ────────────────────────  CH21 PAGE  ──────────────────────── -->
  <div class="camera-page" data-camera="ch21">
    <div class="page-header">
      <div>
        <div class="page-title">CH21 · Passing Station</div>
        <div class="page-subtitle">Weighing scale chokepoint · accept tossed LEFT, reject tossed RIGHT</div>
      </div>
      <div class="header-badges">
        <div class="duration-badge" id="ch21-date">--</div>
        <div class="duration-badge" id="ch21-duration">--</div>
      </div>
    </div>

    <div class="kpi-grid">
      <div class="kpi-card">
        <div class="kpi-label">Total Blankets</div>
        <div class="kpi-value blue" id="ch21-kpi-total">0</div>
        <div class="kpi-sub">accepted + rejected</div>
      </div>
      <div class="kpi-card green">
        <div class="kpi-label">Accepted</div>
        <div class="kpi-value green" id="ch21-kpi-acc">0</div>
        <div class="kpi-sub">weighed on scale</div>
      </div>
      <div class="kpi-card red">
        <div class="kpi-label">Rejected</div>
        <div class="kpi-value red" id="ch21-kpi-rej">0</div>
        <div class="kpi-sub">tossed right</div>
      </div>

      <div class="kpi-card">
        <div class="kpi-label">Reject Rate</div>
        <div class="kpi-value" id="ch21-kpi-rej-pct">0</div>
        <div class="kpi-sub">of total blankets</div>
      </div>
      <div class="kpi-card">
        <div class="kpi-label">Throughput</div>
        <div class="kpi-value" id="ch21-kpi-rate">0</div>
        <div class="kpi-sub">blankets / hour</div>
      </div>
      <div class="kpi-card">
        <div class="kpi-label">Peak (5-min)</div>
        <div class="kpi-value" id="ch21-kpi-peak">0</div>
        <div class="kpi-sub">scaled to /hr equivalent</div>
      </div>
    </div>

    <div class="panel">
      <div class="panel-header">
        <div class="panel-title"><span class="dot"></span> Cumulative Finishing Curve</div>
        <div class="panel-tag">Total · Accepted · Rejected over the day</div>
      </div>
      <canvas id="ch21-chart-cumulative" style="height: 300px;"></canvas>
    </div>

    <div class="panel-grid-2">
      <div class="panel">
        <div class="panel-header">
          <div class="panel-title"><span class="dot"></span> Hourly Throughput</div>
          <div class="panel-tag">Per clock hour</div>
        </div>
        <canvas id="ch21-chart-hourly" style="height: 280px;"></canvas>
      </div>
      <div class="panel">
        <div class="panel-header">
          <div class="panel-title" style="color: var(--red)"><span class="dot" style="background: var(--red);"></span> Reject Rate Trend</div>
          <div class="panel-tag">% rejected per hour</div>
        </div>
        <canvas id="ch21-chart-reject" style="height: 280px;"></canvas>
      </div>
    </div>

    <div class="panel">
      <div class="panel-header">
        <div class="panel-title"><span class="dot"></span> Scale + Table Signal</div>
        <div class="panel-tag">Scale-diff and table-texture over the day</div>
      </div>
      <canvas id="ch21-chart-signal" style="height: 240px;"></canvas>
    </div>

    <div class="panel">
      <div class="panel-header">
        <div class="panel-title"><span class="dot"></span> Hourly Breakdown</div>
        <div class="panel-tag">Per-hour stats</div>
      </div>
      <table>
        <thead>
          <tr>
            <th>Hour</th>
            <th class="num">Accepted</th>
            <th class="num">Rejected</th>
            <th class="num">Total</th>
            <th class="num">Reject %</th>
            <th class="bar-cell">Distribution</th>
          </tr>
        </thead>
        <tbody id="ch21-hourly-tbody"></tbody>
      </table>
    </div>
  </div>

  <!-- ────────────────────────  CH27 PAGE  ──────────────────────── -->
  <div class="camera-page" data-camera="ch27">
    <div class="page-header">
      <div>
        <div class="page-title">CH27 · Taping Station</div>
        <div class="page-subtitle">Two parallel taping tables · XGBoost pulse classifier (v11)</div>
      </div>
      <div class="header-badges">
        <div class="duration-badge" id="ch27-date">--</div>
        <div class="duration-badge" id="ch27-duration">--</div>
        <div class="duration-badge" id="ch27-left-count">--</div>
        <div class="duration-badge" id="ch27-right-count">--</div>
      </div>
    </div>

    <div class="model-banner">
      <span><span class="label">Model</span><span class="pill" id="ch27-banner-model">v11</span></span>
      <span><span class="label">Per-clip F1</span><span class="pill">0.963</span></span>
      <span><span class="label">Features</span><span class="pill muted">30</span></span>
      <span><span class="label">Training clips</span><span class="pill muted">20</span></span>
      <span id="ch27-banner-extra" class="label" style="flex:1; min-width: 200px; text-align:right;"></span>
    </div>

    <div class="kpi-grid">
      <div class="kpi-card">
        <div class="kpi-label">Total Cycles</div>
        <div class="kpi-value" id="ch27-kpi-total" style="color: var(--ch19);">0</div>
        <div class="kpi-sub" id="ch27-kpi-total-sub">L + R combined</div>
      </div>
      <div class="kpi-card left">
        <div class="kpi-label">LEFT Table</div>
        <div class="kpi-value left" id="ch27-kpi-left">0</div>
        <div class="kpi-sub" id="ch27-kpi-left-sub">cycles</div>
      </div>
      <div class="kpi-card right">
        <div class="kpi-label">RIGHT Table</div>
        <div class="kpi-value right" id="ch27-kpi-right">0</div>
        <div class="kpi-sub" id="ch27-kpi-right-sub">cycles</div>
      </div>

      <div class="kpi-card">
        <div class="kpi-label">Throughput</div>
        <div class="kpi-value" id="ch27-kpi-rate">0</div>
        <div class="kpi-sub">cycles / hour</div>
      </div>
      <div class="kpi-card">
        <div class="kpi-label">Peak (5-min)</div>
        <div class="kpi-value" id="ch27-kpi-peak">0</div>
        <div class="kpi-sub">scaled to /hr equivalent</div>
      </div>
      <div class="kpi-card">
        <div class="kpi-label">Median Cycle</div>
        <div class="kpi-value" id="ch27-kpi-median">0</div>
        <div class="kpi-sub">load → toss duration</div>
      </div>
    </div>

    <div class="panel">
      <div class="panel-header">
        <div class="panel-title"><span class="dot"></span> Cumulative Cycles — LEFT vs RIGHT</div>
        <div class="panel-tag">Running totals over the day</div>
      </div>
      <canvas id="ch27-chart-cumulative" style="height: 320px;"></canvas>
    </div>

    <div class="panel-grid-2">
      <div class="panel">
        <div class="panel-header">
          <div class="panel-title"><span class="dot"></span> Per-Segment Throughput</div>
          <div class="panel-tag">Real clock time · gaps shown explicitly</div>
        </div>
        <canvas id="ch27-chart-hourly" style="height: 280px;"></canvas>
      </div>
      <div class="panel">
        <div class="panel-header">
          <div class="panel-title"><span class="dot" style="background:var(--ch19)"></span> Cycle-Duration Distribution</div>
          <div class="panel-tag" id="ch27-duration-tag">Histogram</div>
        </div>
        <canvas id="ch27-chart-duration" style="height: 280px;"></canvas>
      </div>
    </div>

    <div class="panel">
      <div class="panel-header">
        <div class="panel-title"><span class="dot" style="background: var(--green);"></span> Classifier Confidence Distribution</div>
        <div class="panel-tag">Stacked by table · XGBoost predict_proba</div>
      </div>
      <canvas id="ch27-chart-confidence" style="height: 240px;"></canvas>
    </div>

    <div class="panel">
      <div class="panel-header">
        <div class="panel-title"><span class="dot"></span> Per-Segment Breakdown</div>
        <div class="panel-tag" id="ch27-segment-tag">Per-NVR-file stats</div>
      </div>
      <table>
        <thead>
          <tr>
            <th>#</th><th>Window</th>
            <th class="num">Duration</th>
            <th class="num">LEFT</th><th class="num">RIGHT</th><th class="num">Total</th>
            <th class="num">Rate/hr</th>
            <th class="num">Median Cycle</th><th class="num">Avg Conf</th>
            <th class="bar-cell">Distribution</th>
          </tr>
        </thead>
        <tbody id="ch27-segment-tbody"></tbody>
      </table>
    </div>
  </div>

  <footer>
    <p>Blanket Production Tracker · 3-camera CV system · Panipat factory</p>
    <p style="margin-top: 0.4rem;">Generated <span id="gen-time"></span></p>
  </footer>
</div>

<script>
const D = __DATA__;

// ══════════════════════════════════════════════════════════════
// SHARED HELPERS
// ══════════════════════════════════════════════════════════════
const fmtDur = (s) => s >= 3600 ? (s/3600).toFixed(1) + 'hr' : Math.round(s/60) + 'min';
const median = (arr) => {
  if (!arr.length) return 0;
  const s = [...arr].sort((a,b)=>a-b);
  const m = Math.floor(s.length/2);
  return s.length % 2 ? s[m] : (s[m-1] + s[m]) / 2;
};

function setupCanvas(id, h) {
  const c = document.getElementById(id);
  if (!c) return null;
  c.style.height = h + 'px';
  const dpr = window.devicePixelRatio || 1;
  const r = c.getBoundingClientRect();
  const w = Math.max(100, r.width);
  c.width = w * dpr;
  c.height = h * dpr;
  const ctx = c.getContext('2d');
  ctx.scale(dpr, dpr);
  return { ctx, W: w, H: h };
}

function drawGridH(ctx, pad, cW, cH, max, divs=4) {
  ctx.strokeStyle = 'rgba(42,42,61,0.4)';
  ctx.lineWidth = 1;
  ctx.font = '10px JetBrains Mono, monospace';
  ctx.fillStyle = 'rgba(107,114,128,0.7)';
  ctx.textAlign = 'right';
  for (let i = 0; i <= divs; i++) {
    const y = pad.top + (cH * i / divs);
    ctx.beginPath();
    ctx.moveTo(pad.left, y);
    ctx.lineTo(pad.left + cW, y);
    ctx.stroke();
    const v = max * (1 - i/divs);
    ctx.fillText(v >= 100 ? Math.round(v).toString() : v.toFixed(v >= 10 ? 0 : 1), pad.left - 4, y + 3);
  }
}

function drawHourAxis(ctx, pad, cW, H, dur, startHour, numHours) {
  ctx.font = '10px JetBrains Mono, monospace';
  ctx.fillStyle = 'rgba(107,114,128,0.8)';
  ctx.textAlign = 'center';
  for (let h = 0; h <= numHours; h++) {
    const tSec = h * 3600;
    if (tSec > dur) continue;
    const x = pad.left + (tSec / dur) * cW;
    const lbl = String((startHour + h) % 24).padStart(2,'0') + ':00';
    ctx.fillText(lbl, x, H - 6);
  }
}

function animateCount(el, target, duration = 1000, decimals = 0, suffix = '') {
  if (!el) return;
  const start = performance.now();
  function tick(now) {
    const progress = Math.min((now - start) / duration, 1);
    const eased = 1 - Math.pow(1 - progress, 3);
    const current = target * eased;
    const text = decimals > 0 ? current.toFixed(decimals) : Math.round(current);
    el.textContent = (progress >= 1) ? (text + suffix) : text;
    if (progress < 1) requestAnimationFrame(tick);
  }
  requestAnimationFrame(tick);
}

// ══════════════════════════════════════════════════════════════
// TAB SWITCHING
// ══════════════════════════════════════════════════════════════
let currentCamera = null;
let renderedCameras = new Set();

// Full render = KPIs + charts + tables (run once per camera)
const FULL_RENDERERS = {
  ch19: () => D.ch19 && renderCH19(),
  ch21: () => D.ch21 && renderCH21(),
  ch27: () => D.ch27 && renderCH27(),
};

// Charts-only redraw — used on tab re-show and resize. Doesn't touch KPI
// numbers (which would otherwise re-animate from zero each time).
const CHART_REDRAWERS = {
  ch19: () => {
    if (!D.ch19) return;
    drawCH19Cumulative(); drawCH19Hourly(); drawCH19Confidence(); drawCH19Signal();
  },
  ch21: () => {
    if (!D.ch21) return;
    drawCH21Cumulative(); drawCH21Hourly(); drawCH21RejectTrend(); drawCH21Signal();
  },
  ch27: () => {
    if (!D.ch27) return;
    drawCH27Cumulative(); drawCH27Hourly(); drawCH27Duration(); drawCH27Confidence();
  },
};

function switchCamera(name) {
  if (currentCamera === name) return;
  currentCamera = name;
  document.querySelectorAll('.tab').forEach(t => t.classList.toggle('active', t.dataset.camera === name));
  document.querySelectorAll('.camera-page').forEach(p => p.classList.toggle('active', p.dataset.camera === name));
  if (!renderedCameras.has(name) && FULL_RENDERERS[name]) {
    FULL_RENDERERS[name]();
    renderedCameras.add(name);
  } else if (CHART_REDRAWERS[name]) {
    // The page was just made visible — canvases that were 0-width while
    // hidden need a resize-and-redraw. KPIs stay at their final values.
    requestAnimationFrame(() => requestAnimationFrame(CHART_REDRAWERS[name]));
  }
}

document.querySelectorAll('.tab').forEach(tab => {
  tab.addEventListener('click', () => switchCamera(tab.dataset.camera));
});

// ── Header summary badge + tab counts ─────────────────────────
const summaryParts = [];
if (D.ch19) summaryParts.push(D.ch19.summary.total_cuts + ' cuts');
if (D.ch21) summaryParts.push(D.ch21.results.total_blankets + ' blankets');
if (D.ch27) summaryParts.push((D.ch27.summary.total_cycles || 0) + ' cycles');
document.getElementById('badge-summary').textContent = summaryParts.join(' · ');
document.getElementById('gen-time').textContent = new Date(D.generated_at).toLocaleString();

if (D.ch19) document.getElementById('tab-count-ch19').textContent = '· ' + D.ch19.summary.total_cuts;
if (D.ch21) document.getElementById('tab-count-ch21').textContent = '· ' + D.ch21.results.total_blankets;
if (D.ch27) document.getElementById('tab-count-ch27').textContent = '· ' + (D.ch27.summary.total_cycles || 0);

// Disable tabs without data
['ch19', 'ch21', 'ch27'].forEach(name => {
  if (!D[name]) {
    const t = document.querySelector(`.tab[data-camera="${name}"]`);
    if (t) { t.disabled = true; t.style.opacity = 0.35; t.style.cursor = 'not-allowed'; }
  }
});

// ══════════════════════════════════════════════════════════════
// CH19 — CUTTING
// ══════════════════════════════════════════════════════════════
function renderCH19() {
  const ch = D.ch19;
  const dur = ch.metadata.duration_sec;
  const startHour = ch.start_clock_hour;
  const numHours = Math.ceil(dur / 3600);
  const events = ch.events;
  const summary = ch.summary;
  const breaks = ch.breaks;

  // ── Header badges ──
  document.getElementById('ch19-date').textContent = ch.date;
  document.getElementById('ch19-duration').textContent = fmtDur(dur);
  document.getElementById('ch19-version').textContent =
    (ch.metadata.version || 'v5-robust');

  // ── KPIs ──
  const high = summary.confidence_high || 0;
  const total = summary.total_cuts;
  const highPct = total > 0 ? (high / total * 100) : 0;
  const breakMin = (summary.break_time_sec || 0) / 60;
  const breakCount = Math.floor(breaks.length / 2);

  animateCount(document.getElementById('ch19-kpi-total'), total);
  animateCount(document.getElementById('ch19-kpi-rate'), summary.cuts_per_minute || 0, 1000, 1);
  animateCount(document.getElementById('ch19-kpi-cycle'), summary.avg_cycle_sec || 0, 1000, 1, 's');
  animateCount(document.getElementById('ch19-kpi-high'), highPct, 1000, 1, '%');
  animateCount(document.getElementById('ch19-kpi-active'), (summary.active_time_sec || 0) / 3600, 1000, 1, 'hr');
  animateCount(document.getElementById('ch19-kpi-breaks'), breakCount);

  document.getElementById('ch19-kpi-high-sub').textContent =
    high + ' of ' + total + ' cuts';
  document.getElementById('ch19-kpi-active-sub').textContent =
    fmtDur(summary.active_time_sec || 0);
  document.getElementById('ch19-kpi-breaks-sub').textContent =
    breakMin.toFixed(1) + ' min idle';
  document.getElementById('ch19-kpi-total-sub').textContent =
    'over ' + fmtDur(dur);

  // ── Cumulative chart ──
  drawCH19Cumulative();
  drawCH19Hourly();
  drawCH19Confidence();
  drawCH19Signal();
  renderCH19HourlyTable();
}

function drawCH19Cumulative() {
  const setup = setupCanvas('ch19-chart-cumulative', 300);
  if (!setup) return;
  const { ctx, W, H } = setup;
  const ch = D.ch19;
  const dur = ch.metadata.duration_sec;
  const startHour = ch.start_clock_hour;
  const numHours = Math.ceil(dur / 3600);
  const events = ch.events;
  const breaks = ch.breaks || [];

  const pad = { top: 25, right: 70, bottom: 35, left: 55 };
  const cW = W - pad.left - pad.right;
  const cH = H - pad.top - pad.bottom;

  // Cumulative cuts series
  const pts = [{t:0, v:0}];
  let count = 0;
  for (const e of events) { count++; pts.push({t: e.time_sec, v: count}); }
  pts.push({t: dur, v: count});
  const vMax = Math.max(count, 1);

  drawGridH(ctx, pad, cW, cH, vMax, 5);
  drawHourAxis(ctx, pad, cW, H, dur, startHour, numHours);

  // Break shading
  for (let i = 0; i < breaks.length - 1; i += 2) {
    const a = breaks[i], b = breaks[i+1];
    if (a.type === 'break_start' && b.type === 'break_end') {
      const x1 = pad.left + (a.time_sec / dur) * cW;
      const x2 = pad.left + (b.time_sec / dur) * cW;
      ctx.fillStyle = 'rgba(245,158,11,0.07)';
      ctx.fillRect(x1, pad.top, Math.max(1, x2-x1), cH);
    }
  }

  // Area fill
  const grad = ctx.createLinearGradient(0, pad.top, 0, pad.top + cH);
  grad.addColorStop(0, 'rgba(245,158,11,0.30)');
  grad.addColorStop(1, 'rgba(245,158,11,0.02)');
  ctx.beginPath();
  pts.forEach((p, i) => {
    const x = pad.left + (p.t / dur) * cW;
    const y = pad.top + (1 - p.v / vMax) * cH;
    i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
  });
  ctx.lineTo(pad.left + cW, pad.top + cH);
  ctx.lineTo(pad.left, pad.top + cH);
  ctx.closePath();
  ctx.fillStyle = grad;
  ctx.fill();

  // Line
  ctx.strokeStyle = 'rgb(245,158,11)';
  ctx.lineWidth = 2.5;
  ctx.lineJoin = 'round';
  ctx.beginPath();
  pts.forEach((p, i) => {
    const x = pad.left + (p.t / dur) * cW;
    const y = pad.top + (1 - p.v / vMax) * cH;
    i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
  });
  ctx.stroke();

  // End label
  ctx.font = '11px JetBrains Mono, monospace';
  ctx.textAlign = 'left';
  ctx.fillStyle = 'rgb(245,158,11)';
  ctx.fillText(count + ' cuts', pad.left + cW + 6, pad.top + (1 - count/vMax) * cH + 4);
}

function drawCH19Hourly() {
  const setup = setupCanvas('ch19-chart-hourly', 280);
  if (!setup) return;
  const { ctx, W, H } = setup;
  const ch = D.ch19;
  const dur = ch.metadata.duration_sec;
  const startHour = ch.start_clock_hour;
  const numHours = Math.ceil(dur / 3600);
  const events = ch.events;

  const pad = { top: 25, right: 15, bottom: 32, left: 50 };
  const cW = W - pad.left - pad.right;
  const cH = H - pad.top - pad.bottom;

  const cuts = new Array(numHours).fill(0);
  for (const e of events) {
    const h = Math.min(Math.floor(e.time_sec/3600), numHours-1);
    cuts[h]++;
  }
  const maxBar = Math.max(...cuts, 1);

  drawGridH(ctx, pad, cW, cH, maxBar, 4);

  const groupW = cW / numHours;
  const barW = groupW * 0.62;
  const barPad = (groupW - barW) / 2;

  for (let h = 0; h < numHours; h++) {
    if (cuts[h] === 0) continue;
    const x = pad.left + h * groupW + barPad;
    const barH = (cuts[h] / maxBar) * cH;
    const grad = ctx.createLinearGradient(0, pad.top + cH - barH, 0, pad.top + cH);
    grad.addColorStop(0, 'rgba(245,158,11,0.95)');
    grad.addColorStop(1, 'rgba(245,158,11,0.55)');
    ctx.fillStyle = grad;
    ctx.beginPath();
    ctx.roundRect(x, pad.top + cH - barH, barW, barH, [4,4,0,0]);
    ctx.fill();

    ctx.fillStyle = 'rgba(232,232,240,0.85)';
    ctx.font = '10px JetBrains Mono, monospace';
    ctx.textAlign = 'center';
    ctx.fillText(cuts[h], x + barW/2, pad.top + cH - barH - 5);
  }

  // X labels
  ctx.font = '10px JetBrains Mono, monospace';
  ctx.fillStyle = 'rgba(107,114,128,0.8)';
  ctx.textAlign = 'center';
  for (let h = 0; h < numHours; h++) {
    const x = pad.left + h * groupW + groupW/2;
    const lbl = String((startHour + h) % 24).padStart(2,'0') + ':00';
    ctx.fillText(lbl, x, H - 8);
  }
}

function drawCH19Confidence() {
  const setup = setupCanvas('ch19-chart-confidence', 280);
  if (!setup) return;
  const { ctx, W, H } = setup;
  const ch = D.ch19;
  const dur = ch.metadata.duration_sec;
  const startHour = ch.start_clock_hour;
  const numHours = Math.ceil(dur / 3600);
  const events = ch.events;

  const pad = { top: 25, right: 15, bottom: 32, left: 50 };
  const cW = W - pad.left - pad.right;
  const cH = H - pad.top - pad.bottom;

  const high = new Array(numHours).fill(0);
  const med  = new Array(numHours).fill(0);
  const low  = new Array(numHours).fill(0);
  for (const e of events) {
    const h = Math.min(Math.floor(e.time_sec/3600), numHours-1);
    if (e.confidence === 'high') high[h]++;
    else if (e.confidence === 'medium') med[h]++;
    else low[h]++;
  }
  const maxBar = Math.max(...high.map((a,i)=> a + med[i] + low[i]), 1);

  drawGridH(ctx, pad, cW, cH, maxBar, 4);

  const groupW = cW / numHours;
  const barW = groupW * 0.62;
  const barPad = (groupW - barW) / 2;

  for (let h = 0; h < numHours; h++) {
    const x = pad.left + h * groupW + barPad;
    const hH = (high[h] / maxBar) * cH;
    const mH = (med[h] / maxBar) * cH;
    const lH = (low[h] / maxBar) * cH;

    if (high[h] > 0) {
      ctx.fillStyle = 'rgba(16,185,129,0.85)';
      ctx.beginPath();
      const r = (med[h] + low[h]) > 0 ? [0,0,0,0] : [4,4,0,0];
      ctx.roundRect(x, pad.top + cH - hH, barW, hH, r);
      ctx.fill();
    }
    if (med[h] > 0) {
      ctx.fillStyle = 'rgba(245,158,11,0.85)';
      ctx.beginPath();
      const r = low[h] > 0 ? [0,0,0,0] : [4,4,0,0];
      ctx.roundRect(x, pad.top + cH - hH - mH, barW, mH, r);
      ctx.fill();
    }
    if (low[h] > 0) {
      ctx.fillStyle = 'rgba(239,68,68,0.85)';
      ctx.beginPath();
      ctx.roundRect(x, pad.top + cH - hH - mH - lH, barW, lH, [4,4,0,0]);
      ctx.fill();
    }

    const total = high[h] + med[h] + low[h];
    if (total > 0) {
      ctx.fillStyle = 'rgba(232,232,240,0.85)';
      ctx.font = '10px JetBrains Mono, monospace';
      ctx.textAlign = 'center';
      ctx.fillText(total, x + barW/2, pad.top + cH - hH - mH - lH - 5);
    }
  }

  // X labels
  ctx.font = '10px JetBrains Mono, monospace';
  ctx.fillStyle = 'rgba(107,114,128,0.8)';
  ctx.textAlign = 'center';
  for (let h = 0; h < numHours; h++) {
    const x = pad.left + h * groupW + groupW/2;
    const lbl = String((startHour + h) % 24).padStart(2,'0') + ':00';
    ctx.fillText(lbl, x, H - 8);
  }

  // Legend
  ctx.font = '10px JetBrains Mono, monospace';
  ctx.textAlign = 'left';
  let lx = pad.left + 5, ly = pad.top - 8;
  const items = [['high', 'rgba(16,185,129,0.85)'], ['med', 'rgba(245,158,11,0.85)'], ['low', 'rgba(239,68,68,0.85)']];
  for (const [lbl, color] of items) {
    ctx.fillStyle = color;
    ctx.fillRect(lx, ly - 8, 11, 8);
    ctx.fillStyle = 'rgba(200,200,220,0.8)';
    ctx.fillText(lbl, lx + 16, ly - 1);
    lx += 55;
  }
}

function drawCH19Signal() {
  const setup = setupCanvas('ch19-chart-signal', 220);
  if (!setup) return;
  const { ctx, W, H } = setup;
  const ch = D.ch19;
  const dur = ch.metadata.duration_sec;
  const startHour = ch.start_clock_hour;
  const numHours = Math.ceil(dur / 3600);
  const fd = ch.frame_data || [];

  const pad = { top: 22, right: 15, bottom: 32, left: 50 };
  const cW = W - pad.left - pad.right;
  const cH = H - pad.top - pad.bottom;
  if (!fd.length) return;

  const maxVal = 255;

  // Break shading
  for (const f of fd) { /* iterate once for consistent style */ }
  let inBreak = false, breakStart = 0;
  for (const f of fd) {
    const wasIn = inBreak;
    inBreak = f.in_break === true;
    if (inBreak && !wasIn) breakStart = f.time_sec;
    if (!inBreak && wasIn) {
      const x1 = pad.left + (breakStart / dur) * cW;
      const x2 = pad.left + (f.time_sec / dur) * cW;
      ctx.fillStyle = 'rgba(245,158,11,0.07)';
      ctx.fillRect(x1, pad.top, Math.max(1, x2-x1), cH);
    }
  }

  drawGridH(ctx, pad, cW, cH, maxVal, 4);
  drawHourAxis(ctx, pad, cW, H, dur, startHour, numHours);

  // Sample
  const maxPts = 400;
  const step = fd.length > maxPts ? Math.ceil(fd.length / maxPts) : 1;
  const sampled = fd.filter((_, i) => i % step === 0 || i === fd.length - 1);

  // Brightness line (smoothed)
  ctx.strokeStyle = 'rgb(245,158,11)';
  ctx.lineWidth = 1.5;
  ctx.beginPath();
  sampled.forEach((f, i) => {
    const v = f.smoothed != null ? f.smoothed : (f.brightness || 0);
    const x = pad.left + (f.time_sec / dur) * cW;
    const y = pad.top + (1 - Math.min(v, maxVal) / maxVal) * cH;
    i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
  });
  ctx.stroke();

  // Legend
  ctx.font = '10px JetBrains Mono, monospace';
  ctx.textAlign = 'left';
  let lx = pad.left + 5, ly = pad.top + 14;
  ctx.fillStyle = 'rgb(245,158,11)';
  ctx.fillRect(lx, ly - 7, 12, 3);
  ctx.fillStyle = 'rgba(200,200,220,0.85)';
  ctx.fillText('Smoothed brightness', lx + 16, ly - 3);
  lx += 165;
  ctx.fillStyle = 'rgba(245,158,11,0.18)';
  ctx.fillRect(lx, ly - 8, 14, 7);
  ctx.fillStyle = 'rgba(200,200,220,0.85)';
  ctx.fillText('Break period', lx + 18, ly - 3);
}

function renderCH19HourlyTable() {
  const ch = D.ch19;
  const dur = ch.metadata.duration_sec;
  const startHour = ch.start_clock_hour;
  const numHours = Math.ceil(dur / 3600);
  const events = ch.events;

  const high = new Array(numHours).fill(0);
  const med  = new Array(numHours).fill(0);
  const low  = new Array(numHours).fill(0);
  for (const e of events) {
    const h = Math.min(Math.floor(e.time_sec/3600), numHours-1);
    if (e.confidence === 'high') high[h]++;
    else if (e.confidence === 'medium') med[h]++;
    else low[h]++;
  }
  const totals = high.map((a,i)=> a + med[i] + low[i]);
  const maxTotal = Math.max(...totals, 1);

  let html = '';
  for (let h = 0; h < numHours; h++) {
    const t = totals[h];
    if (t === 0 && h === numHours - 1 && (h * 3600 + 3600) > dur) continue;
    const pctHigh = t > 0 ? (high[h]/t*100).toFixed(0) : '—';
    const lbl = String((startHour + h) % 24).padStart(2,'0') + ':00';
    const lblNext = String((startHour + h + 1) % 24).padStart(2,'0') + ':00';
    const wHigh = (high[h] / maxTotal) * 100;
    html += `<tr>
      <td class="label">${lbl} – ${lblNext}</td>
      <td class="num">${t}</td>
      <td class="num" style="color: var(--green)">${high[h]}</td>
      <td class="num" style="color: var(--ch19)">${med[h]}</td>
      <td class="num" style="color: var(--red)">${low[h]}</td>
      <td class="num">${pctHigh}${pctHigh !== '—' ? '%' : ''}</td>
      <td class="bar-cell">
        <div class="bar-row">
          <div style="width:${wHigh}%; background: linear-gradient(90deg, #10b981, #059669);"></div>
        </div>
      </td>
    </tr>`;
  }
  document.getElementById('ch19-hourly-tbody').innerHTML = html;
  document.getElementById('ch19-hourly-tag').textContent =
    numHours + ' hourly buckets · ' + ch.summary.total_cuts + ' total cuts';
}

// ══════════════════════════════════════════════════════════════
// CH21 — PASSING
// ══════════════════════════════════════════════════════════════
function renderCH21() {
  const ch = D.ch21;
  const dur = ch.video_info.duration_sec;
  const events = ch.events;
  const accepted = ch.results.accepted;
  const rejected = ch.results.rejected;
  const total = ch.results.total_blankets;

  const acceptedEvents = events.filter(e => e.type === 'blanket_accepted');
  const rejectedEvents = events.filter(e => e.type === 'blanket_rejected');
  const allFinished = [...acceptedEvents, ...rejectedEvents].sort((a,b)=>a.time_sec-b.time_sec);

  // Header
  document.getElementById('ch21-date').textContent = ch.date;
  document.getElementById('ch21-duration').textContent = fmtDur(dur) + ' · ' + ch.video_info.total_segments + ' segments';

  // KPIs
  const rejPct = total > 0 ? (rejected / total) * 100 : 0;
  const rate = total / Math.max(0.001, dur / 3600);
  let avgCycle = 0;
  if (allFinished.length > 1) {
    let g = 0;
    for (let i = 1; i < allFinished.length; i++)
      g += allFinished[i].time_sec - allFinished[i-1].time_sec;
    avgCycle = g / (allFinished.length - 1);
  }
  let peak = 0;
  for (let i = 0; i < allFinished.length; i++) {
    const wEnd = allFinished[i].time_sec + 300;
    let count = 0;
    for (let j = i; j < allFinished.length && allFinished[j].time_sec <= wEnd; j++) count++;
    if (count > peak) peak = count;
  }
  peak = Math.round(peak * 12);

  animateCount(document.getElementById('ch21-kpi-total'), total);
  animateCount(document.getElementById('ch21-kpi-acc'), accepted);
  animateCount(document.getElementById('ch21-kpi-rej'), rejected);
  animateCount(document.getElementById('ch21-kpi-rej-pct'), rejPct, 1000, 1, '%');
  animateCount(document.getElementById('ch21-kpi-rate'), Math.round(rate));
  animateCount(document.getElementById('ch21-kpi-peak'), peak);

  drawCH21Cumulative();
  drawCH21Hourly();
  drawCH21RejectTrend();
  drawCH21Signal();
  renderCH21HourlyTable();
}

function drawCH21Cumulative() {
  const setup = setupCanvas('ch21-chart-cumulative', 300);
  if (!setup) return;
  const { ctx, W, H } = setup;
  const ch = D.ch21;
  const dur = ch.video_info.duration_sec;
  const startHour = ch.start_clock_hour;
  const numHours = Math.ceil(dur / 3600);
  const events = ch.events;
  const acceptedEvents = events.filter(e => e.type === 'blanket_accepted');
  const rejectedEvents = events.filter(e => e.type === 'blanket_rejected');
  const allFinished = [...acceptedEvents, ...rejectedEvents].sort((a,b)=>a.time_sec-b.time_sec);

  const pad = { top: 25, right: 80, bottom: 35, left: 55 };
  const cW = W - pad.left - pad.right;
  const cH = H - pad.top - pad.bottom;

  const totalPts = [{t:0, v:0}], accPts = [{t:0, v:0}], rejPts = [{t:0, v:0}];
  let aT=0, aA=0, aR=0;
  for (const e of allFinished) {
    if (e.type === 'blanket_accepted') aA++;
    else if (e.type === 'blanket_rejected') aR++;
    aT++;
    totalPts.push({t:e.time_sec, v:aT});
    accPts.push({t:e.time_sec, v:aA});
    rejPts.push({t:e.time_sec, v:aR});
  }
  totalPts.push({t:dur, v:aT}); accPts.push({t:dur, v:aA}); rejPts.push({t:dur, v:aR});
  const vMax = Math.max(aT, 1);

  drawGridH(ctx, pad, cW, cH, vMax, 5);
  drawHourAxis(ctx, pad, cW, H, dur, startHour, numHours);

  // Total area
  const grad = ctx.createLinearGradient(0, pad.top, 0, pad.top + cH);
  grad.addColorStop(0, 'rgba(59,130,246,0.30)');
  grad.addColorStop(1, 'rgba(59,130,246,0.02)');
  ctx.beginPath();
  totalPts.forEach((p, i) => {
    const x = pad.left + (p.t / dur) * cW;
    const y = pad.top + (1 - p.v / vMax) * cH;
    i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
  });
  ctx.lineTo(pad.left + cW, pad.top + cH);
  ctx.lineTo(pad.left, pad.top + cH);
  ctx.closePath();
  ctx.fillStyle = grad;
  ctx.fill();

  function plot(pts, color, lw) {
    ctx.strokeStyle = color;
    ctx.lineWidth = lw;
    ctx.lineJoin = 'round';
    ctx.beginPath();
    pts.forEach((p, i) => {
      const x = pad.left + (p.t / dur) * cW;
      const y = pad.top + (1 - p.v / vMax) * cH;
      i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
    });
    ctx.stroke();
  }
  plot(totalPts, 'rgb(59,130,246)', 2.5);
  plot(accPts, 'rgb(16,185,129)', 1.8);
  plot(rejPts, 'rgb(239,68,68)', 1.5);

  ctx.font = '11px JetBrains Mono, monospace';
  ctx.textAlign = 'left';
  const lblX = pad.left + cW + 6;
  ctx.fillStyle = 'rgb(59,130,246)'; ctx.fillText(aT + ' total', lblX, pad.top + (1 - aT/vMax) * cH + 4);
  ctx.fillStyle = 'rgb(16,185,129)'; ctx.fillText(aA + ' acc',   lblX, pad.top + (1 - aA/vMax) * cH + 4);
  ctx.fillStyle = 'rgb(239,68,68)';  ctx.fillText(aR + ' rej',   lblX, pad.top + (1 - aR/vMax) * cH + 4);
}

function drawCH21Hourly() {
  const setup = setupCanvas('ch21-chart-hourly', 280);
  if (!setup) return;
  const { ctx, W, H } = setup;
  const ch = D.ch21;
  const dur = ch.video_info.duration_sec;
  const startHour = ch.start_clock_hour;
  const numHours = Math.ceil(dur / 3600);
  const events = ch.events;
  const accepted = events.filter(e => e.type === 'blanket_accepted');
  const rejected = events.filter(e => e.type === 'blanket_rejected');

  const pad = { top: 25, right: 15, bottom: 32, left: 50 };
  const cW = W - pad.left - pad.right;
  const cH = H - pad.top - pad.bottom;

  const acc = new Array(numHours).fill(0);
  const rej = new Array(numHours).fill(0);
  for (const e of accepted) { const h = Math.min(Math.floor(e.time_sec/3600), numHours-1); acc[h]++; }
  for (const e of rejected) { const h = Math.min(Math.floor(e.time_sec/3600), numHours-1); rej[h]++; }
  const maxBar = Math.max(...acc.map((a,i) => a+rej[i]), 1);

  drawGridH(ctx, pad, cW, cH, maxBar, 4);

  const groupW = cW / numHours;
  const barW = groupW * 0.62;
  const barPad = (groupW - barW) / 2;

  for (let h = 0; h < numHours; h++) {
    const x = pad.left + h * groupW + barPad;
    const accH = (acc[h] / maxBar) * cH;
    const rejH = (rej[h] / maxBar) * cH;
    if (acc[h] > 0) {
      const g = ctx.createLinearGradient(0, pad.top + cH - accH, 0, pad.top + cH);
      g.addColorStop(0, 'rgba(16,185,129,0.95)');
      g.addColorStop(1, 'rgba(16,185,129,0.55)');
      ctx.fillStyle = g;
      ctx.beginPath();
      const r = rej[h] > 0 ? [0,0,0,0] : [4,4,0,0];
      ctx.roundRect(x, pad.top + cH - accH, barW, accH, r);
      ctx.fill();
    }
    if (rej[h] > 0) {
      const g = ctx.createLinearGradient(0, pad.top + cH - accH - rejH, 0, pad.top + cH - accH);
      g.addColorStop(0, 'rgba(239,68,68,0.95)');
      g.addColorStop(1, 'rgba(239,68,68,0.55)');
      ctx.fillStyle = g;
      ctx.beginPath();
      ctx.roundRect(x, pad.top + cH - accH - rejH, barW, rejH, [4,4,0,0]);
      ctx.fill();
    }
    const total = acc[h] + rej[h];
    if (total > 0) {
      ctx.fillStyle = 'rgba(232,232,240,0.85)';
      ctx.font = '10px JetBrains Mono, monospace';
      ctx.textAlign = 'center';
      ctx.fillText(total, x + barW/2, pad.top + cH - accH - rejH - 5);
    }
  }

  // X labels
  ctx.font = '10px JetBrains Mono, monospace';
  ctx.fillStyle = 'rgba(107,114,128,0.8)';
  ctx.textAlign = 'center';
  for (let h = 0; h < numHours; h++) {
    const x = pad.left + h * groupW + groupW/2;
    const lbl = String((startHour + h) % 24).padStart(2,'0') + ':00';
    ctx.fillText(lbl, x, H - 8);
  }

  // Legend
  ctx.font = '10px JetBrains Mono, monospace';
  ctx.textAlign = 'left';
  let lx = pad.left + 5, ly = pad.top - 8;
  ctx.fillStyle = 'rgba(16,185,129,0.85)';
  ctx.fillRect(lx, ly - 8, 11, 8);
  ctx.fillStyle = 'rgba(200,200,220,0.8)';
  ctx.fillText('Accepted', lx + 16, ly - 1);
  lx += 80;
  ctx.fillStyle = 'rgba(239,68,68,0.85)';
  ctx.fillRect(lx, ly - 8, 11, 8);
  ctx.fillStyle = 'rgba(200,200,220,0.8)';
  ctx.fillText('Rejected', lx + 16, ly - 1);
}

function drawCH21RejectTrend() {
  const setup = setupCanvas('ch21-chart-reject', 280);
  if (!setup) return;
  const { ctx, W, H } = setup;
  const ch = D.ch21;
  const dur = ch.video_info.duration_sec;
  const startHour = ch.start_clock_hour;
  const numHours = Math.ceil(dur / 3600);
  const events = ch.events;
  const total = ch.results.total_blankets;
  const rejected = ch.results.rejected;

  const pad = { top: 25, right: 50, bottom: 32, left: 50 };
  const cW = W - pad.left - pad.right;
  const cH = H - pad.top - pad.bottom;

  const acc = new Array(numHours).fill(0);
  const rej = new Array(numHours).fill(0);
  for (const e of events.filter(e=>e.type==='blanket_accepted')) {
    const h = Math.min(Math.floor(e.time_sec/3600), numHours-1); acc[h]++;
  }
  for (const e of events.filter(e=>e.type==='blanket_rejected')) {
    const h = Math.min(Math.floor(e.time_sec/3600), numHours-1); rej[h]++;
  }
  const pcts = acc.map((a, i) => {
    const t = a + rej[i];
    return t > 0 ? (rej[i] / t) * 100 : 0;
  });
  const maxPct = Math.max(40, Math.ceil(Math.max(...pcts) / 10) * 10);

  drawGridH(ctx, pad, cW, cH, maxPct, 4);

  const overallPct = total > 0 ? (rejected / total) * 100 : 0;
  const yOver = pad.top + (1 - overallPct/maxPct) * cH;
  ctx.strokeStyle = 'rgba(239,68,68,0.35)';
  ctx.setLineDash([5, 4]);
  ctx.lineWidth = 1;
  ctx.beginPath(); ctx.moveTo(pad.left, yOver); ctx.lineTo(pad.left + cW, yOver); ctx.stroke();
  ctx.setLineDash([]);
  ctx.fillStyle = 'rgba(239,68,68,0.7)';
  ctx.font = '10px JetBrains Mono, monospace';
  ctx.textAlign = 'left';
  ctx.fillText('avg ' + overallPct.toFixed(1) + '%', pad.left + cW + 4, yOver + 4);

  const groupW = cW / numHours;
  const barW = groupW * 0.55;
  const barPad = (groupW - barW) / 2;
  for (let h = 0; h < numHours; h++) {
    const v = pcts[h];
    if (v === 0) continue;
    const x = pad.left + h * groupW + barPad;
    const barH = (v / maxPct) * cH;
    const grad = ctx.createLinearGradient(0, pad.top + cH - barH, 0, pad.top + cH);
    grad.addColorStop(0, 'rgba(239,68,68,0.85)');
    grad.addColorStop(1, 'rgba(239,68,68,0.35)');
    ctx.fillStyle = grad;
    ctx.beginPath();
    ctx.roundRect(x, pad.top + cH - barH, barW, barH, [4,4,0,0]);
    ctx.fill();
    ctx.fillStyle = 'rgba(232,232,240,0.85)';
    ctx.font = '10px JetBrains Mono, monospace';
    ctx.textAlign = 'center';
    ctx.fillText(v.toFixed(0) + '%', x + barW/2, pad.top + cH - barH - 5);
  }

  // X labels
  ctx.font = '10px JetBrains Mono, monospace';
  ctx.fillStyle = 'rgba(107,114,128,0.8)';
  ctx.textAlign = 'center';
  for (let h = 0; h < numHours; h++) {
    const x = pad.left + h * groupW + groupW/2;
    const lbl = String((startHour + h) % 24).padStart(2,'0') + ':00';
    ctx.fillText(lbl, x, H - 8);
  }
}

function drawCH21Signal() {
  const setup = setupCanvas('ch21-chart-signal', 240);
  if (!setup) return;
  const { ctx, W, H } = setup;
  const ch = D.ch21;
  const dur = ch.video_info.duration_sec;
  const startHour = ch.start_clock_hour;
  const numHours = Math.ceil(dur / 3600);
  const fr = ch.frames || [];

  const pad = { top: 22, right: 15, bottom: 32, left: 50 };
  const cW = W - pad.left - pad.right;
  const cH = H - pad.top - pad.bottom;
  if (!fr.length) return;

  const maxVal = 120;
  const threshold = (ch.detection_config && ch.detection_config.scale_on_threshold) || 25;

  // Loaded shading
  let inLoaded = false, loadStart = 0;
  for (const f of fr) {
    const was = inLoaded;
    inLoaded = f.scale_state === 'loaded';
    if (inLoaded && !was) loadStart = f.time_sec;
    if (!inLoaded && was) {
      const x1 = pad.left + (loadStart / dur) * cW;
      const x2 = pad.left + (f.time_sec / dur) * cW;
      ctx.fillStyle = 'rgba(59,130,246,0.07)';
      ctx.fillRect(x1, pad.top, Math.max(1, x2-x1), cH);
    }
  }

  drawGridH(ctx, pad, cW, cH, maxVal, 4);
  drawHourAxis(ctx, pad, cW, H, dur, startHour, numHours);

  const thY = pad.top + (1 - threshold/maxVal) * cH;
  ctx.setLineDash([6, 4]);
  ctx.strokeStyle = 'rgba(245,158,11,0.55)';
  ctx.lineWidth = 1.4;
  ctx.beginPath();
  ctx.moveTo(pad.left, thY); ctx.lineTo(pad.left + cW, thY); ctx.stroke();
  ctx.setLineDash([]);
  ctx.fillStyle = 'rgba(245,158,11,0.8)';
  ctx.font = '9px JetBrains Mono, monospace';
  ctx.textAlign = 'right';
  ctx.fillText('scale ON=' + threshold, pad.left + cW - 6, thY - 4);

  const maxPts = 400;
  const step = fr.length > maxPts ? Math.ceil(fr.length / maxPts) : 1;
  const pts = fr.filter((_, i) => i % step === 0 || i === fr.length - 1);

  const grad = ctx.createLinearGradient(0, pad.top, 0, pad.top + cH);
  grad.addColorStop(0, 'rgba(59,130,246,0.30)');
  grad.addColorStop(1, 'rgba(59,130,246,0.02)');
  ctx.beginPath();
  pts.forEach((f, i) => {
    const x = pad.left + (f.time_sec / dur) * cW;
    const y = pad.top + (1 - Math.min(f.table_texture || 0, maxVal) / maxVal) * cH;
    i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
  });
  ctx.lineTo(pad.left + cW, pad.top + cH);
  ctx.lineTo(pad.left, pad.top + cH);
  ctx.closePath();
  ctx.fillStyle = grad;
  ctx.fill();

  ctx.beginPath();
  ctx.strokeStyle = 'rgb(59,130,246)';
  ctx.lineWidth = 1.5;
  pts.forEach((f, i) => {
    const x = pad.left + (f.time_sec / dur) * cW;
    const y = pad.top + (1 - Math.min(f.table_texture || 0, maxVal) / maxVal) * cH;
    i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
  });
  ctx.stroke();

  ctx.beginPath();
  ctx.strokeStyle = 'rgba(245,158,11,0.65)';
  ctx.lineWidth = 1;
  pts.forEach((f, i) => {
    const x = pad.left + (f.time_sec / dur) * cW;
    const y = pad.top + (1 - Math.min(f.scale_diff || 0, maxVal) / maxVal) * cH;
    i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
  });
  ctx.stroke();

  // Legend
  ctx.font = '10px JetBrains Mono, monospace';
  ctx.textAlign = 'left';
  let lx = pad.left + 5, ly = pad.top + 14;
  ctx.fillStyle = 'rgb(59,130,246)';
  ctx.fillRect(lx, ly - 7, 12, 3);
  ctx.fillStyle = 'rgba(200,200,220,0.85)';
  ctx.fillText('Table texture', lx + 16, ly - 3);
  lx += 105;
  ctx.fillStyle = 'rgba(245,158,11,0.65)';
  ctx.fillRect(lx, ly - 7, 12, 3);
  ctx.fillStyle = 'rgba(200,200,220,0.85)';
  ctx.fillText('Scale diff', lx + 16, ly - 3);
  lx += 90;
  ctx.fillStyle = 'rgba(59,130,246,0.18)';
  ctx.fillRect(lx, ly - 8, 14, 7);
  ctx.fillStyle = 'rgba(200,200,220,0.85)';
  ctx.fillText('Scale loaded', lx + 18, ly - 3);
}

function renderCH21HourlyTable() {
  const ch = D.ch21;
  const dur = ch.video_info.duration_sec;
  const startHour = ch.start_clock_hour;
  const numHours = Math.ceil(dur / 3600);
  const events = ch.events;
  const accepted = events.filter(e => e.type === 'blanket_accepted');
  const rejected = events.filter(e => e.type === 'blanket_rejected');

  const acc = new Array(numHours).fill(0);
  const rej = new Array(numHours).fill(0);
  for (const e of accepted) { const h = Math.min(Math.floor(e.time_sec/3600), numHours-1); acc[h]++; }
  for (const e of rejected) { const h = Math.min(Math.floor(e.time_sec/3600), numHours-1); rej[h]++; }
  const maxTotal = Math.max(...acc.map((a,i) => a + rej[i]), 1);

  let html = '';
  for (let h = 0; h < numHours; h++) {
    const a = acc[h], r = rej[h], t = a + r;
    if (t === 0 && h === numHours - 1 && (h * 3600 + 3600) > dur) continue;
    const pct = t > 0 ? (r/t*100).toFixed(1) : '—';
    const aPct = (a / maxTotal) * 100;
    const rPct = (r / maxTotal) * 100;
    const lbl = String((startHour + h) % 24).padStart(2,'0') + ':00';
    const lblNext = String((startHour + h + 1) % 24).padStart(2,'0') + ':00';
    html += `<tr>
      <td class="label">${lbl} – ${lblNext}</td>
      <td class="num" style="color:var(--green)">${a}</td>
      <td class="num" style="color:var(--red)">${r}</td>
      <td class="num">${t}</td>
      <td class="num">${pct}${pct !== '—' ? '%' : ''}</td>
      <td class="bar-cell">
        <div class="bar-row">
          <div class="bar-acc" style="width:${aPct}%"></div>
          <div class="bar-rej" style="width:${rPct}%"></div>
        </div>
      </td>
    </tr>`;
  }
  document.getElementById('ch21-hourly-tbody').innerHTML = html;
}

// ══════════════════════════════════════════════════════════════
// CH27 — TAPING
// ══════════════════════════════════════════════════════════════
let ch27ClockBuckets = [];

function buildCH27ClockBuckets() {
  const ch = D.ch27;
  const segs = ch.segments;
  const events = ch.events;
  const startHour = ch.start_clock_hour;
  const numHours = Math.ceil(ch.metadata.duration_sec / 3600);

  if (!segs.length) return [];
  let minClock = startHour, maxClock = startHour + numHours;
  for (const s of segs) {
    const m = (s.label || '').match(/^(\d{2}):(\d{2})-(\d{2}):(\d{2})$/);
    if (!m) continue;
    const sH = +m[1], eH = +m[3], eM = +m[4];
    minClock = Math.min(minClock, sH);
    maxClock = Math.max(maxClock, eM > 0 ? eH + 1 : eH);
  }
  const buckets = [];
  for (let h = minClock; h < maxClock; h++) {
    buckets.push({
      clockHour: h,
      label: String(h).padStart(2,'0') + ':00',
      l: 0, r: 0, durs: [], probs: [], hasData: false,
    });
  }
  const segByIdx = {};
  for (const s of segs) segByIdx[s.segment_index ?? -1] = s;
  for (const e of events) {
    const seg = segByIdx[e.segment ?? -1];
    if (!seg) continue;
    const m = (seg.label || '').match(/^(\d{2}):(\d{2})-(\d{2}):(\d{2})$/);
    if (!m) continue;
    const segStartSec = (+m[1]) * 3600 + (+m[2]) * 60;
    const evtClockSec = segStartSec + (e.time_sec - (seg.offset_sec || 0));
    const evtClockHour = Math.floor(evtClockSec / 3600);
    const idx = evtClockHour - minClock;
    if (idx < 0 || idx >= buckets.length) continue;
    if (e.table === 'left')  buckets[idx].l++;
    if (e.table === 'right') buckets[idx].r++;
    if (e.cycle_duration_sec && e.cycle_duration_sec < 600)
      buckets[idx].durs.push(e.cycle_duration_sec);
    if (e.v4_prob != null) buckets[idx].probs.push(e.v4_prob);
    buckets[idx].hasData = true;
  }
  for (const b of buckets) {
    for (const s of segs) {
      const m = (s.label || '').match(/^(\d{2}):(\d{2})-(\d{2}):(\d{2})$/);
      if (!m) continue;
      const sH = +m[1], eH = +m[3], eM = +m[4];
      const segEndH = eM > 0 ? eH + 1 : eH;
      if (b.clockHour >= sH && b.clockHour < segEndH) { b.hasData = true; break; }
    }
  }
  return buckets;
}

function renderCH27() {
  const ch = D.ch27;
  const events = ch.events;
  const segs = ch.segments;
  const summary = ch.summary || {};
  const meta = ch.metadata || {};
  const dur = meta.duration_sec || 0;
  const startHour = ch.start_clock_hour;
  const left = events.filter(e => e.table === 'left');
  const right = events.filter(e => e.table === 'right');

  events.sort((a,b) => a.time_sec - b.time_sec);
  ch27ClockBuckets = buildCH27ClockBuckets();

  // Header
  document.getElementById('ch27-date').textContent = ch.date;
  document.getElementById('ch27-duration').textContent = fmtDur(dur) + ' · ' + segs.length + ' segments';
  document.getElementById('ch27-left-count').textContent = left.length + ' LEFT';
  document.getElementById('ch27-right-count').textContent = right.length + ' RIGHT';

  const ver = meta.version || 'v11';
  document.getElementById('ch27-banner-model').textContent = ver.split(' ')[0];
  document.getElementById('ch27-banner-extra').textContent = '· ' + ver;

  // KPIs
  const total = events.length;
  const rate = total / Math.max(0.001, dur / 3600);
  const allDurs = events.map(e => e.cycle_duration_sec || 0).filter(d => d > 0 && d < 600);
  const medCycle = median(allDurs);
  let peak = 0;
  for (let i = 0; i < events.length; i++) {
    const wEnd = events[i].time_sec + 300;
    let count = 0;
    for (let j = i; j < events.length && events[j].time_sec <= wEnd; j++) count++;
    if (count > peak) peak = count;
  }
  peak = Math.round(peak * 12);

  animateCount(document.getElementById('ch27-kpi-total'), total);
  animateCount(document.getElementById('ch27-kpi-left'), left.length);
  animateCount(document.getElementById('ch27-kpi-right'), right.length);
  animateCount(document.getElementById('ch27-kpi-rate'), Math.round(rate));
  animateCount(document.getElementById('ch27-kpi-peak'), peak);
  animateCount(document.getElementById('ch27-kpi-median'), medCycle, 1000, 1, 's');

  document.getElementById('ch27-kpi-left-sub').textContent  = 'cycles · ' + Math.round(left.length / Math.max(0.001, dur/3600))  + '/hr';
  document.getElementById('ch27-kpi-right-sub').textContent = 'cycles · ' + Math.round(right.length / Math.max(0.001, dur/3600)) + '/hr';
  document.getElementById('ch27-kpi-total-sub').textContent =
    'L + R · long cycles: ' + events.filter(e => e.long_cycle).length;

  drawCH27Cumulative();
  drawCH27Hourly();
  drawCH27Duration();
  drawCH27Confidence();
  renderCH27SegmentTable();
}

function drawCH27Cumulative() {
  const setup = setupCanvas('ch27-chart-cumulative', 320);
  if (!setup) return;
  const { ctx, W, H } = setup;
  const ch = D.ch27;
  const events = ch.events;
  const dur = ch.metadata.duration_sec;
  const startHour = ch.start_clock_hour;
  const numHours = Math.ceil(dur / 3600);

  const pad = { top: 25, right: 80, bottom: 35, left: 55 };
  const cW = W - pad.left - pad.right;
  const cH = H - pad.top - pad.bottom;

  const lPts = [{t:0,v:0}], rPts = [{t:0,v:0}];
  let lc=0, rc=0;
  for (const e of events) {
    if (e.table === 'left')  lc++;
    if (e.table === 'right') rc++;
    if (e.table === 'left')  lPts.push({t:e.time_sec, v:lc});
    if (e.table === 'right') rPts.push({t:e.time_sec, v:rc});
  }
  lPts.push({t:dur, v:lc}); rPts.push({t:dur, v:rc});
  const vMax = Math.max(lc, rc, 1);

  drawGridH(ctx, pad, cW, cH, vMax, 5);
  drawHourAxis(ctx, pad, cW, H, dur, startHour, numHours);

  function plot(pts, color, lw, fillGrad) {
    if (fillGrad) {
      const grad = ctx.createLinearGradient(0, pad.top, 0, pad.top + cH);
      grad.addColorStop(0, fillGrad[0]);
      grad.addColorStop(1, fillGrad[1]);
      ctx.beginPath();
      pts.forEach((p, i) => {
        const x = pad.left + (p.t / dur) * cW;
        const y = pad.top + (1 - p.v / vMax) * cH;
        i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
      });
      ctx.lineTo(pad.left + cW, pad.top + cH);
      ctx.lineTo(pad.left, pad.top + cH);
      ctx.closePath();
      ctx.fillStyle = grad;
      ctx.fill();
    }
    ctx.strokeStyle = color;
    ctx.lineWidth = lw;
    ctx.lineJoin = 'round';
    ctx.beginPath();
    pts.forEach((p, i) => {
      const x = pad.left + (p.t / dur) * cW;
      const y = pad.top + (1 - p.v / vMax) * cH;
      i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
    });
    ctx.stroke();
  }
  plot(lPts, 'rgb(124,58,237)', 2.4, ['rgba(124,58,237,0.22)', 'rgba(124,58,237,0.02)']);
  plot(rPts, 'rgb(6,182,212)', 2.4, null);

  ctx.font = '11px JetBrains Mono, monospace';
  ctx.textAlign = 'left';
  const lblX = pad.left + cW + 6;
  ctx.fillStyle = 'rgb(124,58,237)'; ctx.fillText(lc + ' LEFT',  lblX, pad.top + (1 - lc/vMax) * cH + 4);
  ctx.fillStyle = 'rgb(6,182,212)';  ctx.fillText(rc + ' RIGHT', lblX, pad.top + (1 - rc/vMax) * cH + 4);
}

function drawCH27Hourly() {
  const setup = setupCanvas('ch27-chart-hourly', 280);
  if (!setup) return;
  const { ctx, W, H } = setup;
  const pad = { top: 25, right: 15, bottom: 32, left: 50 };
  const cW = W - pad.left - pad.right;
  const cH = H - pad.top - pad.bottom;

  const N = ch27ClockBuckets.length;
  if (N === 0) return;
  const maxBar = Math.max(...ch27ClockBuckets.map(b => b.l + b.r), 1);

  drawGridH(ctx, pad, cW, cH, maxBar, 4);

  const groupW = cW / N;
  const barW = groupW * 0.62;
  const barPad = (groupW - barW) / 2;

  for (let i = 0; i < N; i++) {
    const b = ch27ClockBuckets[i];
    const x = pad.left + i * groupW + barPad;

    if (!b.hasData) {
      ctx.save();
      ctx.strokeStyle = 'rgba(107,114,128,0.25)';
      ctx.setLineDash([3, 3]);
      ctx.lineWidth = 1;
      ctx.strokeRect(x, pad.top + cH - 28, barW, 28);
      ctx.setLineDash([]);
      ctx.fillStyle = 'rgba(107,114,128,0.45)';
      ctx.font = '9px JetBrains Mono, monospace';
      ctx.textAlign = 'center';
      ctx.fillText('no data', x + barW/2, pad.top + cH - 12);
      ctx.restore();
      continue;
    }

    const lh = (b.l / maxBar) * cH;
    const rh = (b.r / maxBar) * cH;

    if (b.l > 0) {
      const g = ctx.createLinearGradient(0, pad.top + cH - lh, 0, pad.top + cH);
      g.addColorStop(0, 'rgba(124,58,237,0.95)');
      g.addColorStop(1, 'rgba(124,58,237,0.55)');
      ctx.fillStyle = g;
      ctx.beginPath();
      const r = b.r > 0 ? [0,0,0,0] : [4,4,0,0];
      ctx.roundRect(x, pad.top + cH - lh, barW, lh, r);
      ctx.fill();
    }
    if (b.r > 0) {
      const g = ctx.createLinearGradient(0, pad.top + cH - lh - rh, 0, pad.top + cH - lh);
      g.addColorStop(0, 'rgba(6,182,212,0.95)');
      g.addColorStop(1, 'rgba(6,182,212,0.55)');
      ctx.fillStyle = g;
      ctx.beginPath();
      ctx.roundRect(x, pad.top + cH - lh - rh, barW, rh, [4,4,0,0]);
      ctx.fill();
    }
    const total = b.l + b.r;
    if (total > 0) {
      ctx.fillStyle = 'rgba(232,232,240,0.85)';
      ctx.font = '10px JetBrains Mono, monospace';
      ctx.textAlign = 'center';
      ctx.fillText(total, x + barW/2, pad.top + cH - lh - rh - 5);
    }
  }

  ctx.font = '10px JetBrains Mono, monospace';
  ctx.textAlign = 'center';
  for (let i = 0; i < N; i++) {
    const x = pad.left + i * groupW + groupW/2;
    ctx.fillStyle = ch27ClockBuckets[i].hasData
      ? 'rgba(107,114,128,0.85)'
      : 'rgba(107,114,128,0.4)';
    ctx.fillText(ch27ClockBuckets[i].label, x, H - 8);
  }

  ctx.font = '10px JetBrains Mono, monospace';
  ctx.textAlign = 'left';
  let lx = pad.left + 5, ly = pad.top - 8;
  ctx.fillStyle = 'rgba(124,58,237,0.85)';
  ctx.fillRect(lx, ly - 8, 11, 8);
  ctx.fillStyle = 'rgba(200,200,220,0.8)';
  ctx.fillText('LEFT', lx + 16, ly - 1);
  lx += 65;
  ctx.fillStyle = 'rgba(6,182,212,0.85)';
  ctx.fillRect(lx, ly - 8, 11, 8);
  ctx.fillStyle = 'rgba(200,200,220,0.8)';
  ctx.fillText('RIGHT', lx + 16, ly - 1);
}

function drawCH27Duration() {
  const setup = setupCanvas('ch27-chart-duration', 280);
  if (!setup) return;
  const { ctx, W, H } = setup;
  const ch = D.ch27;
  const events = ch.events;

  const pad = { top: 25, right: 15, bottom: 38, left: 50 };
  const cW = W - pad.left - pad.right;
  const cH = H - pad.top - pad.bottom;

  const numBins = 30;
  const lBins = new Array(numBins + 1).fill(0);
  const rBins = new Array(numBins + 1).fill(0);
  for (const e of events) {
    const d = e.cycle_duration_sec || 0;
    if (d <= 0) continue;
    const idx = d >= numBins ? numBins : Math.floor(d);
    if (e.table === 'left')  lBins[idx]++;
    if (e.table === 'right') rBins[idx]++;
  }
  const totals = lBins.map((a, i) => a + rBins[i]);
  const maxBar = Math.max(...totals, 1);

  drawGridH(ctx, pad, cW, cH, maxBar, 4);

  const groupW = cW / (numBins + 1);
  const barW = groupW * 0.78;
  for (let i = 0; i <= numBins; i++) {
    const x = pad.left + i * groupW + (groupW - barW)/2;
    const lh = (lBins[i] / maxBar) * cH;
    const rh = (rBins[i] / maxBar) * cH;
    if (lBins[i] > 0) {
      ctx.fillStyle = 'rgba(124,58,237,0.85)';
      ctx.beginPath();
      ctx.roundRect(x, pad.top + cH - lh, barW, lh, [2,2,0,0]);
      ctx.fill();
    }
    if (rBins[i] > 0) {
      ctx.fillStyle = 'rgba(6,182,212,0.85)';
      ctx.beginPath();
      ctx.roundRect(x, pad.top + cH - lh - rh, barW, rh, [2,2,0,0]);
      ctx.fill();
    }
  }

  ctx.font = '10px JetBrains Mono, monospace';
  ctx.fillStyle = 'rgba(107,114,128,0.8)';
  ctx.textAlign = 'center';
  for (let i = 0; i <= numBins; i += 5) {
    const x = pad.left + i * groupW + groupW/2;
    const lbl = i === numBins ? '30+' : i.toString();
    ctx.fillText(lbl + 's', x, H - 18);
  }
  ctx.fillStyle = 'rgba(107,114,128,0.65)';
  ctx.font = '9px JetBrains Mono, monospace';
  ctx.fillText('cycle duration (seconds)', pad.left + cW/2, H - 4);

  const validDurs = events.map(e => e.cycle_duration_sec).filter(d => d > 0 && d < 600);
  const med = median(validDurs);
  if (med > 0 && med < numBins) {
    const x = pad.left + med * groupW + groupW/2;
    ctx.setLineDash([4,3]);
    ctx.strokeStyle = 'rgba(245,158,11,0.7)';
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    ctx.moveTo(x, pad.top); ctx.lineTo(x, pad.top + cH); ctx.stroke();
    ctx.setLineDash([]);
    ctx.fillStyle = 'rgba(245,158,11,0.9)';
    ctx.font = '10px JetBrains Mono, monospace';
    ctx.textAlign = 'left';
    ctx.fillText('median ' + med.toFixed(1) + 's', x + 4, pad.top + 12);
  }

  document.getElementById('ch27-duration-tag').textContent =
    'Cycles ≤ 30s shown · ' + events.filter(e => e.cycle_duration_sec > 30).length + ' longer in overflow';
}

function drawCH27Confidence() {
  const setup = setupCanvas('ch27-chart-confidence', 240);
  if (!setup) return;
  const { ctx, W, H } = setup;
  const ch = D.ch27;
  const events = ch.events;

  const pad = { top: 25, right: 15, bottom: 36, left: 50 };
  const cW = W - pad.left - pad.right;
  const cH = H - pad.top - pad.bottom;

  const numBins = 20;
  const lo = 0.5, hi = 1.0;
  const span = hi - lo;
  const lBins = new Array(numBins).fill(0);
  const rBins = new Array(numBins).fill(0);
  for (const e of events) {
    const p = e.v4_prob;
    if (p == null || p < lo || p > hi) continue;
    const idx = Math.min(numBins-1, Math.floor((p - lo) / span * numBins));
    if (e.table === 'left')  lBins[idx]++;
    if (e.table === 'right') rBins[idx]++;
  }
  const totals = lBins.map((a, i) => a + rBins[i]);
  const maxBar = Math.max(...totals, 1);

  drawGridH(ctx, pad, cW, cH, maxBar, 4);

  const groupW = cW / numBins;
  const barW = groupW * 0.86;
  for (let i = 0; i < numBins; i++) {
    const x = pad.left + i * groupW + (groupW - barW)/2;
    const lh = (lBins[i] / maxBar) * cH;
    const rh = (rBins[i] / maxBar) * cH;
    if (lBins[i] > 0) {
      ctx.fillStyle = 'rgba(124,58,237,0.85)';
      ctx.beginPath();
      ctx.roundRect(x, pad.top + cH - lh, barW, lh, [2,2,0,0]);
      ctx.fill();
    }
    if (rBins[i] > 0) {
      ctx.fillStyle = 'rgba(6,182,212,0.85)';
      ctx.beginPath();
      ctx.roundRect(x, pad.top + cH - lh - rh, barW, rh, [2,2,0,0]);
      ctx.fill();
    }
  }

  function drawThresh(p, color, label) {
    if (p < lo || p > hi) return;
    const x = pad.left + (p - lo) / span * cW;
    ctx.setLineDash([4,3]);
    ctx.strokeStyle = color;
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    ctx.moveTo(x, pad.top); ctx.lineTo(x, pad.top + cH); ctx.stroke();
    ctx.setLineDash([]);
    ctx.fillStyle = color;
    ctx.font = '10px JetBrains Mono, monospace';
    ctx.textAlign = 'left';
    ctx.fillText(label, x + 3, pad.top + 12);
  }
  drawThresh(0.62, 'rgba(124,58,237,0.95)', 'L th=0.62');
  drawThresh(0.68, 'rgba(6,182,212,0.95)', 'R th=0.68');

  ctx.font = '10px JetBrains Mono, monospace';
  ctx.fillStyle = 'rgba(107,114,128,0.8)';
  ctx.textAlign = 'center';
  for (let i = 0; i <= numBins; i += 4) {
    const x = pad.left + i * groupW;
    const v = lo + (i / numBins) * span;
    ctx.fillText(v.toFixed(2), x, H - 18);
  }
  ctx.fillStyle = 'rgba(107,114,128,0.65)';
  ctx.font = '9px JetBrains Mono, monospace';
  ctx.fillText('classifier probability', pad.left + cW/2, H - 4);
}

function renderCH27SegmentTable() {
  const ch = D.ch27;
  const events = ch.events;
  const segs = ch.segments;
  const dur = ch.metadata.duration_sec;
  const tbody = document.getElementById('ch27-segment-tbody');

  const dursBySeg = {}, probsBySeg = {};
  for (const e of events) {
    const sIdx = e.segment ?? -1;
    (dursBySeg[sIdx] ||= []);
    (probsBySeg[sIdx] ||= []);
    if (e.cycle_duration_sec && e.cycle_duration_sec < 600) dursBySeg[sIdx].push(e.cycle_duration_sec);
    if (e.v4_prob != null) probsBySeg[sIdx].push(e.v4_prob);
  }

  const maxTotal = Math.max(...segs.map(s => (s.left_cycles||0) + (s.right_cycles||0)), 1);

  let html = '';
  for (const seg of segs) {
    const sIdx = seg.segment_index ?? -1;
    const l = seg.left_cycles || 0;
    const r = seg.right_cycles || 0;
    const t = l + r;
    const rateHr = seg.duration_sec > 0 ? Math.round(t / (seg.duration_sec/3600)) : '—';
    const segDurs  = dursBySeg[sIdx]  || [];
    const segProbs = probsBySeg[sIdx] || [];
    const med = segDurs.length ? median(segDurs).toFixed(1) : '—';
    const avgC = segProbs.length
      ? (segProbs.reduce((a,b)=>a+b,0) / segProbs.length).toFixed(2)
      : '—';
    const lPct = (l / maxTotal) * 100;
    const rPct = (r / maxTotal) * 100;
    html += `<tr>
      <td class="label">#${sIdx + 1}</td>
      <td>${seg.label || '—'}</td>
      <td class="num">${fmtDur(seg.duration_sec || 0)}</td>
      <td class="num" style="color:var(--ch27-l)">${l}</td>
      <td class="num" style="color:var(--ch27-r)">${r}</td>
      <td class="num">${t}</td>
      <td class="num">${rateHr}</td>
      <td class="num">${med}${med !== '—' ? 's' : ''}</td>
      <td class="num">${avgC}</td>
      <td class="bar-cell">
        <div class="bar-row">
          <div class="bar-l" style="width:${lPct}%"></div>
          <div class="bar-r" style="width:${rPct}%"></div>
        </div>
      </td>
    </tr>`;
  }

  const missingHours = ch27ClockBuckets.filter(b => !b.hasData).map(b => b.label);
  if (missingHours.length) {
    html += `<tr style="opacity:0.55">
      <td class="label">—</td>
      <td>${missingHours.join(', ')}</td>
      <td class="num" colspan="8" style="color:var(--muted); text-align:left; padding-left: 0.8rem;">
        no NVR footage — gap in source data
      </td>
    </tr>`;
  }

  tbody.innerHTML = html;
  document.getElementById('ch27-segment-tag').textContent =
    segs.length + ' files · ' + fmtDur(dur) + ' analyzed' +
    (missingHours.length ? ' · gap: ' + missingHours.join(', ') : '');
}

// ── Initial: pick default tab and render ──────────────────────
const defaultCamera = D.ch27 ? 'ch27' : (D.ch21 ? 'ch21' : 'ch19');
requestAnimationFrame(() => switchCamera(defaultCamera));

// ── Resize ────────────────────────────────────────────────────
let resizeTimer;
window.addEventListener('resize', () => {
  clearTimeout(resizeTimer);
  resizeTimer = setTimeout(() => {
    if (currentCamera && CHART_REDRAWERS[currentCamera]) CHART_REDRAWERS[currentCamera]();
  }, 120);
});
</script>
</body>
</html>"""


# ──────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────

def main():
    print("Loading per-camera full-day data...")
    ch19 = load_ch19()
    ch21 = load_ch21()
    ch27 = load_ch27()

    if ch19 is None and ch21 is None and ch27 is None:
        print("ERROR: no data found. Run the per-camera full-day pipelines first.")
        sys.exit(1)

    if ch19: print(f"  CH19: {ch19['summary'].get('total_cuts', 0)} cuts · "
                   f"{ch19['metadata'].get('duration_sec', 0)/3600:.2f}h · "
                   f"{len(ch19['events'])} events · {len(ch19['frame_data'])} frames")
    else:    print("  CH19: missing — skipped")

    if ch21: print(f"  CH21: {ch21['results']['total_blankets']} blankets "
                   f"({ch21['results']['accepted']} acc / {ch21['results']['rejected']} rej) · "
                   f"{ch21['video_info']['duration_sec']/3600:.2f}h · "
                   f"{len(ch21['events'])} events · {len(ch21['frames'])} frames")
    else:    print("  CH21: missing — skipped")

    if ch27: print(f"  CH27: {ch27['summary'].get('total_cycles', 0)} cycles "
                   f"(L={ch27['summary'].get('left_cycles', 0)} / R={ch27['summary'].get('right_cycles', 0)}) · "
                   f"{ch27['metadata'].get('duration_sec', 0)/3600:.2f}h · "
                   f"{len(ch27['events'])} events")
    else:    print("  CH27: missing — skipped")

    bundle = {
        "generated_at": datetime.now().isoformat(),
        "ch19": ch19,
        "ch21": ch21,
        "ch27": ch27,
    }
    payload = json.dumps(bundle, separators=(",", ":"))
    html = TEMPLATE.replace("__DATA__", payload)
    OUTPUT_HTML.write_text(html)

    size_mb = len(html) / (1024 * 1024)
    print(f"\nWrote {OUTPUT_HTML} ({size_mb:.2f} MB)")


if __name__ == "__main__":
    main()
