#!/usr/bin/env python3
"""Generate standalone CH27 (Taping Station) dashboard from full-day data.

Reads `taping/taping_fullday_v11.json` (or override via CH27_JSON env var) and
produces a self-contained HTML file with KPI cards, cumulative L/R curves,
hourly throughput bars, cycle-duration histogram, classifier-confidence
distribution, and per-hour / per-segment tables.

Usage:
    python3 generate_ch27_dashboard.py
        → writes ch27_dashboard.html
"""

import json
import math
import os
import sys
from datetime import datetime
from pathlib import Path

BASE = Path(__file__).parent
DEFAULT_JSON = BASE / "taping" / "taping_fullday_v11.json"
CH27_JSON = Path(os.environ.get("CH27_JSON", str(DEFAULT_JSON)))
OUTPUT_HTML = BASE / "ch27_dashboard.html"

# Day starts at 09:00 (per CH27 segment labels)
START_CLOCK_HOUR = 9


def load_and_compact():
    """Load the full-day JSON; compact events + segments for embedding."""
    with open(CH27_JSON) as f:
        ch27 = json.load(f)

    events = ch27.get("events", [])
    summary = ch27.get("summary", {})
    metadata = ch27.get("metadata", {})
    segments = ch27.get("segments", [])
    suppressed = ch27.get("suppressed_candidates", [])

    # Trim event payload — drop fields the dashboard doesn't use
    keep_fields = {
        "table", "time_sec", "frame", "cycle_duration_sec", "air_motion_peak",
        "v4_prob", "v4_eff_thresh", "long_cycle", "via_overlap_detector",
        "via_secondary_path", "segment", "blob_max_area",
    }
    slim_events = [{k: e[k] for k in keep_fields if k in e} for e in events]

    return {
        "generated_at": datetime.now().isoformat(),
        "metadata": metadata,
        "summary": summary,
        "segments": segments,
        "events": slim_events,
        # summary.suppressed_count is the authoritative total; the
        # `suppressed_candidates` array is capped at 5000 in the JSON file
        "suppressed_count": summary.get("suppressed_count", len(suppressed)),
        "start_clock_hour": START_CLOCK_HOUR,
    }


TEMPLATE = r'''<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>CH27 Taping Station — Blanket Tracker</title>
<link href="https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
<style>
  :root {
    --bg: #0a0a0f;
    --surface: #12121a;
    --card: #1a1a26;
    --border: #2a2a3d;
    --left:  #7c3aed;   /* LEFT table = purple */
    --right: #06b6d4;   /* RIGHT table = cyan */
    --green: #10b981;
    --red:   #ef4444;
    --amber: #f59e0b;
    --accent: #7c3aed;
    --accent-grad: linear-gradient(135deg, #7c3aed, #06b6d4);
    --text: #e8e8f0;
    --muted: #6b7280;
  }

  * { margin: 0; padding: 0; box-sizing: border-box; }

  body {
    background: var(--bg);
    color: var(--text);
    font-family: 'Syne', sans-serif;
    min-height: 100vh;
    overflow-x: hidden;
  }

  body::before {
    content: '';
    position: fixed;
    inset: 0;
    background-image:
      linear-gradient(rgba(124,58,237,0.04) 1px, transparent 1px),
      linear-gradient(90deg, rgba(6,182,212,0.04) 1px, transparent 1px);
    background-size: 40px 40px;
    pointer-events: none;
    z-index: 0;
  }

  .container {
    position: relative;
    z-index: 1;
    max-width: 1400px;
    margin: 0 auto;
    padding: 2rem;
  }

  header {
    display: flex;
    align-items: flex-start;
    justify-content: space-between;
    flex-wrap: wrap;
    gap: 1rem;
    margin-bottom: 2.5rem;
    padding-bottom: 1.5rem;
    border-bottom: 1px solid var(--border);
  }

  .logo-area h1 {
    font-size: 2.1rem;
    font-weight: 800;
    letter-spacing: -0.04em;
    background: var(--accent-grad);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
  }

  .logo-area p {
    color: var(--muted);
    font-size: 0.85rem;
    font-family: 'JetBrains Mono', monospace;
    margin-top: 0.3rem;
  }

  .header-badges { display: flex; gap: 0.5rem; flex-wrap: wrap; }

  .duration-badge {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    padding: 0.4rem 0.9rem;
    border-radius: 100px;
    font-size: 0.75rem;
    font-family: 'JetBrains Mono', monospace;
    background: rgba(124,58,237,0.1);
    border: 1px solid rgba(124,58,237,0.3);
    color: var(--left);
  }

  .duration-badge.right {
    background: rgba(6,182,212,0.1);
    border: 1px solid rgba(6,182,212,0.3);
    color: var(--right);
  }

  .duration-badge.amber {
    background: rgba(245,158,11,0.1);
    border: 1px solid rgba(245,158,11,0.3);
    color: var(--amber);
  }

  .duration-badge.date {
    background: rgba(232,232,240,0.05);
    border: 1px solid rgba(232,232,240,0.2);
    color: var(--text);
    font-weight: 600;
  }

  /* KPIs */
  .kpi-grid {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 1rem;
    margin-bottom: 1.5rem;
  }

  .kpi-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: 1.2rem;
    position: relative;
    overflow: hidden;
    transition: transform 0.2s, border-color 0.2s;
  }

  .kpi-card:hover { transform: translateY(-2px); border-color: rgba(124,58,237,0.4); }

  .kpi-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: var(--accent-grad);
    opacity: 0.7;
  }

  .kpi-card.left::before  { background: linear-gradient(90deg, #7c3aed, #a78bfa); }
  .kpi-card.right::before { background: linear-gradient(90deg, #06b6d4, #22d3ee); }
  .kpi-card.amber::before { background: linear-gradient(90deg, #f59e0b, #d97706); }
  .kpi-card.green::before { background: linear-gradient(90deg, #10b981, #059669); }

  .kpi-label {
    font-size: 0.7rem;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: var(--muted);
    font-family: 'JetBrains Mono', monospace;
    margin-bottom: 0.5rem;
  }

  .kpi-value {
    font-size: 2.2rem;
    font-weight: 800;
    letter-spacing: -0.04em;
    line-height: 1;
    color: var(--text);
  }

  .kpi-value.left  { color: var(--left); }
  .kpi-value.right { color: var(--right); }
  .kpi-value.green { color: var(--green); }
  .kpi-value.amber { color: var(--amber); }

  .kpi-sub {
    font-size: 0.7rem;
    color: var(--muted);
    font-family: 'JetBrains Mono', monospace;
    margin-top: 0.4rem;
  }

  /* Panels */
  .panel {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: 1.5rem;
    margin-bottom: 1.5rem;
  }

  .panel-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 1rem;
    padding-bottom: 0.8rem;
    border-bottom: 1px solid var(--border);
  }

  .panel-title {
    font-size: 0.85rem;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    font-family: 'JetBrains Mono', monospace;
    font-weight: 600;
    color: var(--text);
    display: flex;
    align-items: center;
    gap: 0.5rem;
  }

  .panel-title .dot {
    width: 8px;
    height: 8px;
    border-radius: 50%;
    background: var(--accent);
  }

  .panel-tag {
    font-size: 0.7rem;
    font-family: 'JetBrains Mono', monospace;
    color: var(--muted);
    background: rgba(124,58,237,0.05);
    padding: 0.25rem 0.7rem;
    border-radius: 100px;
    border: 1px solid rgba(124,58,237,0.15);
  }

  canvas { width: 100%; display: block; }

  .panel-grid-2 {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 1.5rem;
    margin-bottom: 1.5rem;
  }

  /* Tables */
  table { width: 100%; border-collapse: collapse; }

  thead th {
    text-align: left;
    padding: 0.7rem 0.8rem;
    font-size: 0.7rem;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    font-family: 'JetBrains Mono', monospace;
    color: var(--muted);
    border-bottom: 1px solid var(--border);
    font-weight: 600;
  }
  thead th.num { text-align: right; }

  tbody td {
    padding: 0.6rem 0.8rem;
    font-size: 0.9rem;
    border-bottom: 1px solid rgba(42,42,61,0.4);
    font-family: 'JetBrains Mono', monospace;
  }
  tbody td.num { text-align: right; }
  tbody td.label { color: var(--text); font-weight: 500; }
  tbody tr:hover { background: rgba(124,58,237,0.04); }

  .bar-cell { padding: 0.6rem 0.8rem; min-width: 200px; }
  .bar-row {
    display: flex;
    height: 18px;
    border-radius: 4px;
    overflow: hidden;
    background: rgba(42,42,61,0.3);
  }
  .bar-l { background: linear-gradient(90deg, #7c3aed, #a78bfa); }
  .bar-r { background: linear-gradient(90deg, #06b6d4, #22d3ee); }

  .model-banner {
    display: flex;
    flex-wrap: wrap;
    gap: 0.6rem 1.2rem;
    align-items: center;
    background: rgba(124,58,237,0.05);
    border: 1px solid rgba(124,58,237,0.2);
    border-radius: 12px;
    padding: 0.9rem 1.2rem;
    margin-bottom: 1.5rem;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.78rem;
    color: var(--text);
  }
  .model-banner .pill {
    background: rgba(16,185,129,0.12);
    color: var(--green);
    padding: 0.18rem 0.55rem;
    border-radius: 100px;
    font-weight: 600;
    border: 1px solid rgba(16,185,129,0.25);
  }
  .model-banner .pill.amber { background: rgba(245,158,11,0.12); color: var(--amber); border-color: rgba(245,158,11,0.25); }
  .model-banner .pill.muted { background: rgba(107,114,128,0.12); color: var(--muted); border-color: rgba(107,114,128,0.25); }
  .model-banner .label { color: var(--muted); margin-right: 0.3rem; }

  footer {
    text-align: center;
    padding: 2rem 0 1rem;
    color: var(--muted);
    font-size: 0.75rem;
    font-family: 'JetBrains Mono', monospace;
    border-top: 1px solid var(--border);
    margin-top: 2rem;
  }

  @media (max-width: 900px) {
    .kpi-grid { grid-template-columns: repeat(2, 1fr); }
    .panel-grid-2 { grid-template-columns: 1fr; }
  }
</style>
</head>
<body>
<div class="container">
  <header>
    <div class="logo-area">
      <h1>CH27 TAPING STATION</h1>
      <p id="header-subtitle">Two parallel taping tables · Panipat factory</p>
    </div>
    <div class="header-badges">
      <div class="duration-badge date" id="badge-date">28/04/2026</div>
      <div class="duration-badge amber" id="badge-duration">--</div>
      <div class="duration-badge" id="badge-left">--</div>
      <div class="duration-badge right" id="badge-right">--</div>
    </div>
  </header>

  <div class="model-banner">
    <span><span class="label">Model</span><span class="pill" id="banner-model">v11</span></span>
    <span><span class="label">Per-clip F1</span><span class="pill" id="banner-f1">0.963</span></span>
    <span><span class="label">Features</span><span class="pill muted" id="banner-features">30</span></span>
    <span><span class="label">Training clips</span><span class="pill muted" id="banner-clips">20</span></span>
    <span id="banner-extra" class="label" style="flex:1; min-width: 200px; text-align:right;"></span>
  </div>

  <!-- KPIs -->
  <div class="kpi-grid">
    <div class="kpi-card amber">
      <div class="kpi-label">Total Cycles</div>
      <div class="kpi-value amber" id="kpi-total">0</div>
      <div class="kpi-sub" id="kpi-total-sub">L + R combined</div>
    </div>
    <div class="kpi-card left">
      <div class="kpi-label">LEFT Table</div>
      <div class="kpi-value left" id="kpi-left">0</div>
      <div class="kpi-sub" id="kpi-left-sub">cycles</div>
    </div>
    <div class="kpi-card right">
      <div class="kpi-label">RIGHT Table</div>
      <div class="kpi-value right" id="kpi-right">0</div>
      <div class="kpi-sub" id="kpi-right-sub">cycles</div>
    </div>
    <div class="kpi-card">
      <div class="kpi-label">Throughput</div>
      <div class="kpi-value" id="kpi-rate">0</div>
      <div class="kpi-sub">cycles / hour</div>
    </div>
    <div class="kpi-card">
      <div class="kpi-label">Peak (5-min)</div>
      <div class="kpi-value" id="kpi-peak">0</div>
      <div class="kpi-sub">scaled to /hr equivalent</div>
    </div>
    <div class="kpi-card">
      <div class="kpi-label">Median Cycle</div>
      <div class="kpi-value" id="kpi-median">0</div>
      <div class="kpi-sub" id="kpi-median-sub">load → toss duration</div>
    </div>
  </div>

  <!-- Cumulative L vs R -->
  <div class="panel">
    <div class="panel-header">
      <div class="panel-title">
        <span class="dot"></span> Cumulative Cycles — LEFT vs RIGHT
      </div>
      <div class="panel-tag">Running totals over the day</div>
    </div>
    <canvas id="chart-cumulative" style="height: 320px;"></canvas>
  </div>

  <!-- Hourly + cycle duration -->
  <div class="panel-grid-2">
    <div class="panel">
      <div class="panel-header">
        <div class="panel-title">
          <span class="dot"></span> Per-Segment Throughput
        </div>
        <div class="panel-tag">By NVR file (real clock time)</div>
      </div>
      <canvas id="chart-hourly" style="height: 280px;"></canvas>
    </div>

    <div class="panel">
      <div class="panel-header">
        <div class="panel-title">
          <span class="dot" style="background:var(--amber)"></span> Cycle-Duration Distribution
        </div>
        <div class="panel-tag" id="duration-tag">Histogram of cycle_duration_sec</div>
      </div>
      <canvas id="chart-duration" style="height: 280px;"></canvas>
    </div>
  </div>

  <!-- Confidence -->
  <div class="panel">
    <div class="panel-header">
      <div class="panel-title">
        <span class="dot" style="background:var(--green)"></span> Classifier Confidence Distribution
      </div>
      <div class="panel-tag" id="conf-tag">Stacked by table — XGBoost predict_proba</div>
    </div>
    <canvas id="chart-confidence" style="height: 240px;"></canvas>
  </div>

  <!-- Per-segment breakdown table -->
  <div class="panel">
    <div class="panel-header">
      <div class="panel-title">
        <span class="dot"></span> Per-Segment Breakdown
      </div>
      <div class="panel-tag" id="segment-tag">Per-NVR-file stats</div>
    </div>
    <table>
      <thead>
        <tr>
          <th>#</th>
          <th>Window</th>
          <th class="num">Duration</th>
          <th class="num">LEFT</th>
          <th class="num">RIGHT</th>
          <th class="num">Total</th>
          <th class="num">Rate/hr</th>
          <th class="num">Median Cycle</th>
          <th class="num">Avg Conf</th>
          <th class="bar-cell">Distribution</th>
        </tr>
      </thead>
      <tbody id="segment-tbody"></tbody>
    </table>
  </div>

  <footer>
    <p>Blanket Production Tracker · CH27 dashboard · Generated <span id="gen-time"></span></p>
    <p style="margin-top: 0.4rem;" id="footer-model">CH27 v11 · 30 features · 20 training clips · F1=0.963</p>
  </footer>
</div>

<script>
const D = __DATA__;

// ── Derived state ───────────────────────────────────────────────
const events  = D.events;
const segs    = D.segments;
const summary = D.summary || {};
const meta    = D.metadata || {};
const dur     = meta.duration_sec || 0;
const startHour = D.start_clock_hour || 9;
const numHours  = Math.ceil(dur / 3600);

const left  = events.filter(e => e.table === 'left');
const right = events.filter(e => e.table === 'right');
const totalCycles = events.length;
const leftN  = left.length;
const rightN = right.length;
const balance = leftN > 0 && rightN > 0 ? Math.min(leftN, rightN) / Math.max(leftN, rightN) : 0;
const suppressed = D.suppressed_count || summary.suppressed_count || 0;

// Sort events by time once
events.sort((a,b) => a.time_sec - b.time_sec);

// ── Helpers ─────────────────────────────────────────────────────
const fmtDur = (s) => s >= 3600 ? (s/3600).toFixed(1) + 'hr' : Math.round(s/60) + 'min';

function fmtClockHour(h) {
  const hr = (startHour + h) % 24;
  return hr.toString().padStart(2,'0') + ':00';
}

function setupCanvas(id, h) {
  const c = document.getElementById(id);
  if (!c) return null;
  c.style.height = h + 'px';
  const dpr = window.devicePixelRatio || 1;
  const r = c.getBoundingClientRect();
  // Guard against zero-width on first paint (flex layout still resolving)
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

function drawHourAxis(ctx, pad, cW, H, dur) {
  ctx.font = '10px JetBrains Mono, monospace';
  ctx.fillStyle = 'rgba(107,114,128,0.8)';
  ctx.textAlign = 'center';
  for (let h = 0; h <= numHours; h++) {
    const tSec = h * 3600;
    if (tSec > dur) continue;
    const x = pad.left + (tSec / dur) * cW;
    ctx.fillText(fmtClockHour(h), x, H - 6);
  }
}

function median(arr) {
  if (!arr.length) return 0;
  const s = [...arr].sort((a,b)=>a-b);
  const m = Math.floor(s.length/2);
  return s.length % 2 ? s[m] : (s[m-1] + s[m]) / 2;
}

function animateCount(el, target, duration = 1200, decimals = 0) {
  const start = performance.now();
  function tick(now) {
    const progress = Math.min((now - start) / duration, 1);
    const eased = 1 - Math.pow(1 - progress, 3);
    const current = target * eased;
    el.textContent = decimals > 0 ? current.toFixed(decimals) : Math.round(current);
    if (progress < 1) requestAnimationFrame(tick);
  }
  requestAnimationFrame(tick);
}

// ── Header & banner ─────────────────────────────────────────────
document.getElementById('badge-duration').textContent = fmtDur(dur) + ' · ' + segs.length + ' segments';
document.getElementById('badge-left').textContent = leftN + ' LEFT';
document.getElementById('badge-right').textContent = rightN + ' RIGHT';
document.getElementById('gen-time').textContent = new Date(D.generated_at).toLocaleString();

// Model banner — pull version + scores from metadata.version string if present
const ver = meta.version || 'v11';
document.getElementById('banner-model').textContent = ver.split(' ')[0];
document.getElementById('banner-extra').textContent = '· ' + ver;

// ── KPIs ────────────────────────────────────────────────────────
const rate = totalCycles / Math.max(0.001, dur / 3600);

// Median cycle (cap absurd long-cycle outliers)
const allDurs = events.map(e => e.cycle_duration_sec || 0).filter(d => d > 0 && d < 600);
const medCycle = median(allDurs);

// Peak 5-min throughput
let peak = 0;
for (let i = 0; i < events.length; i++) {
  const wEnd = events[i].time_sec + 300;
  let count = 0;
  for (let j = i; j < events.length && events[j].time_sec <= wEnd; j++) count++;
  if (count > peak) peak = count;
}
peak = Math.round(peak * 12);

animateCount(document.getElementById('kpi-total'), totalCycles);
animateCount(document.getElementById('kpi-left'), leftN);
animateCount(document.getElementById('kpi-right'), rightN);
animateCount(document.getElementById('kpi-rate'), Math.round(rate));
animateCount(document.getElementById('kpi-peak'), peak);
animateCount(document.getElementById('kpi-median'), medCycle, 1200, 1);

setTimeout(() => {
  document.getElementById('kpi-median').insertAdjacentText('afterend', 's');
}, 1300);

const leftRate  = leftN  / Math.max(0.001, dur / 3600);
const rightRate = rightN / Math.max(0.001, dur / 3600);
document.getElementById('kpi-left-sub').textContent  = 'cycles · ' + Math.round(leftRate)  + '/hr';
document.getElementById('kpi-right-sub').textContent = 'cycles · ' + Math.round(rightRate) + '/hr';
document.getElementById('kpi-total-sub').textContent =
  'L + R · long cycles: ' + events.filter(e => e.long_cycle).length;

// ════════════════════════════════════════════════════════════════
// CHART 1 — Cumulative LEFT vs RIGHT
// ════════════════════════════════════════════════════════════════
function drawCumulative() {
  const setup = setupCanvas('chart-cumulative', 320);
  if (!setup) return;
  const { ctx, W, H } = setup;
  const pad = { top: 25, right: 80, bottom: 35, left: 55 };
  const cW = W - pad.left - pad.right;
  const cH = H - pad.top - pad.bottom;

  const lPts = [{t:0,v:0}], rPts = [{t:0,v:0}], tPts = [{t:0,v:0}];
  let lc=0, rc=0;
  for (const e of events) {
    if (e.table === 'left')  lc++;
    if (e.table === 'right') rc++;
    if (e.table === 'left')  lPts.push({t:e.time_sec, v:lc});
    if (e.table === 'right') rPts.push({t:e.time_sec, v:rc});
    tPts.push({t:e.time_sec, v: lc + rc});
  }
  lPts.push({t:dur, v:lc}); rPts.push({t:dur, v:rc}); tPts.push({t:dur, v: lc+rc});
  const vMax = Math.max(lc, rc, 1);

  drawGridH(ctx, pad, cW, cH, vMax, 5);
  drawHourAxis(ctx, pad, cW, H, dur);

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

  // LEFT (purple) with light fill
  plot(lPts, 'rgb(124,58,237)', 2.4, ['rgba(124,58,237,0.22)', 'rgba(124,58,237,0.02)']);
  // RIGHT (cyan)
  plot(rPts, 'rgb(6,182,212)', 2.4, null);

  // End-of-day labels
  ctx.font = '11px JetBrains Mono, monospace';
  ctx.textAlign = 'left';
  const lblX = pad.left + cW + 6;
  ctx.fillStyle = 'rgb(124,58,237)'; ctx.fillText(lc + ' LEFT',  lblX, pad.top + (1 - lc/vMax) * cH + 4);
  ctx.fillStyle = 'rgb(6,182,212)';  ctx.fillText(rc + ' RIGHT', lblX, pad.top + (1 - rc/vMax) * cH + 4);

  // Y label
  ctx.save();
  ctx.translate(14, pad.top + cH/2);
  ctx.rotate(-Math.PI/2);
  ctx.fillStyle = 'rgba(107,114,128,0.7)';
  ctx.textAlign = 'center';
  ctx.font = '10px JetBrains Mono, monospace';
  ctx.fillText('cumulative cycles', 0, 0);
  ctx.restore();
}

// ════════════════════════════════════════════════════════════════
// CHART 2 — Per-segment stacked bars (LEFT + RIGHT)
//   Uses real segment labels so missing hours (e.g. 16:00-17:00 gap
//   in the v11 data) are visible as absent bars rather than mislabeled.
// ════════════════════════════════════════════════════════════════

// Pre-compute "buckets" from clock hours. Build a 9-slot timeline
// 09:00→18:00 with one slot per real clock hour. Multiple segments
// inside the same clock hour are merged. Missing hours render as
// gaps with empty bars.
function buildClockBuckets() {
  // Find the day's clock-time span from segments
  if (!segs.length) return [];
  // Parse each segment's label like "09:00-10:00" or "12:00-12:53"
  // and convert to start_clock_hour, end_clock_hour
  const startSec = startHour * 3600;
  let minClock = startHour, maxClock = startHour + numHours;
  // Use segment labels to find true min / max clock hour
  for (const s of segs) {
    const m = (s.label || '').match(/^(\d{2}):(\d{2})-(\d{2}):(\d{2})$/);
    if (!m) continue;
    const sH = +m[1], eH = +m[3], eM = +m[4];
    minClock = Math.min(minClock, sH);
    // round end up if it's mid-hour (e.g. 12:53)
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
  // Map each event to its clock-hour via its segment
  const segByIdx = {};
  for (const s of segs) segByIdx[s.segment_index ?? -1] = s;
  for (const e of events) {
    const seg = segByIdx[e.segment ?? -1];
    if (!seg) continue;
    const m = (seg.label || '').match(/^(\d{2}):(\d{2})-(\d{2}):(\d{2})$/);
    if (!m) continue;
    const segStartSec = (+m[1]) * 3600 + (+m[2]) * 60;
    // Real-clock seconds for this event = seg start + (event time - seg offset)
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
  // Mark which buckets are inside a real segment vs. gap
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
const clockBuckets = buildClockBuckets();

function drawHourly() {
  const setup = setupCanvas('chart-hourly', 280);
  if (!setup) return;
  const { ctx, W, H } = setup;
  const pad = { top: 25, right: 15, bottom: 32, left: 50 };
  const cW = W - pad.left - pad.right;
  const cH = H - pad.top - pad.bottom;

  const N = clockBuckets.length;
  if (N === 0) return;
  const maxBar = Math.max(...clockBuckets.map(b => b.l + b.r), 1);

  drawGridH(ctx, pad, cW, cH, maxBar, 4);

  const groupW = cW / N;
  const barW = groupW * 0.62;
  const barPad = (groupW - barW) / 2;

  for (let i = 0; i < N; i++) {
    const b = clockBuckets[i];
    const x = pad.left + i * groupW + barPad;

    // Empty / no-data slots render as faint dashed outline
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

  // Clock-hour labels
  ctx.font = '10px JetBrains Mono, monospace';
  ctx.fillStyle = 'rgba(107,114,128,0.8)';
  ctx.textAlign = 'center';
  for (let i = 0; i < N; i++) {
    const x = pad.left + i * groupW + groupW/2;
    ctx.fillStyle = clockBuckets[i].hasData
      ? 'rgba(107,114,128,0.85)'
      : 'rgba(107,114,128,0.4)';
    ctx.fillText(clockBuckets[i].label, x, H - 8);
  }

  // Legend
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

// ════════════════════════════════════════════════════════════════
// CHART 3 — Cycle-duration histogram (clipped to <= 30s)
// ════════════════════════════════════════════════════════════════
function drawDuration() {
  const setup = setupCanvas('chart-duration', 280);
  if (!setup) return;
  const { ctx, W, H } = setup;
  const pad = { top: 25, right: 15, bottom: 38, left: 50 };
  const cW = W - pad.left - pad.right;
  const cH = H - pad.top - pad.bottom;

  // Bin: 0-30s in 1s buckets, 30+ in overflow
  const numBins = 30;
  const binSec = 1;
  const lBins = new Array(numBins + 1).fill(0); // last bin = overflow >30s
  const rBins = new Array(numBins + 1).fill(0);
  for (const e of events) {
    const d = e.cycle_duration_sec || 0;
    if (d <= 0) continue;
    const idx = d >= numBins * binSec ? numBins : Math.floor(d / binSec);
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

  // X labels at 0, 5, 10, 15, 20, 25, 30+
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

  // Median marker
  const validDurs = events.map(e => e.cycle_duration_sec).filter(d => d > 0 && d < 600);
  const med = median(validDurs);
  if (med > 0 && med < numBins) {
    const x = pad.left + med * groupW + groupW/2;
    ctx.setLineDash([4,3]);
    ctx.strokeStyle = 'rgba(245,158,11,0.7)';
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    ctx.moveTo(x, pad.top);
    ctx.lineTo(x, pad.top + cH);
    ctx.stroke();
    ctx.setLineDash([]);
    ctx.fillStyle = 'rgba(245,158,11,0.9)';
    ctx.font = '10px JetBrains Mono, monospace';
    ctx.textAlign = 'left';
    ctx.fillText('median ' + med.toFixed(1) + 's', x + 4, pad.top + 12);
  }

  document.getElementById('duration-tag').textContent =
    'Cycles ≤ 30s shown · ' + events.filter(e => e.cycle_duration_sec > 30).length + ' longer cycles in overflow';
}

// ════════════════════════════════════════════════════════════════
// CHART 4 — Confidence (v4_prob) histogram, stacked by table
// ════════════════════════════════════════════════════════════════
function drawConfidence() {
  const setup = setupCanvas('chart-confidence', 240);
  if (!setup) return;
  const { ctx, W, H } = setup;
  const pad = { top: 25, right: 15, bottom: 36, left: 50 };
  const cW = W - pad.left - pad.right;
  const cH = H - pad.top - pad.bottom;

  // 20 bins from 0.5 to 1.0 (production thresholds are L=0.62 R=0.68)
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

  // Threshold markers (L=0.62, R=0.68)
  function drawThresh(p, color, label) {
    if (p < lo || p > hi) return;
    const x = pad.left + (p - lo) / span * cW;
    ctx.setLineDash([4,3]);
    ctx.strokeStyle = color;
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    ctx.moveTo(x, pad.top);
    ctx.lineTo(x, pad.top + cH);
    ctx.stroke();
    ctx.setLineDash([]);
    ctx.fillStyle = color;
    ctx.font = '10px JetBrains Mono, monospace';
    ctx.textAlign = 'left';
    ctx.fillText(label, x + 3, pad.top + 12);
  }
  drawThresh(0.62, 'rgba(124,58,237,0.95)', 'L th=0.62');
  drawThresh(0.68, 'rgba(6,182,212,0.95)', 'R th=0.68');

  // X labels
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

// ════════════════════════════════════════════════════════════════
// PER-SEGMENT TABLE (merged: was hourly + segment, deduplicated)
// ════════════════════════════════════════════════════════════════
function renderSegmentTable() {
  const tbody = document.getElementById('segment-tbody');

  // Group events by segment for median cycle + avg confidence
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
    const rateHr = seg.duration_sec > 0
      ? Math.round(t / (seg.duration_sec/3600))
      : '—';
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
      <td class="num" style="color:var(--left)">${l}</td>
      <td class="num" style="color:var(--right)">${r}</td>
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

  // Add a synthetic gap row if there's a missing clock hour
  const missingHours = clockBuckets.filter(b => !b.hasData).map(b => b.label);
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
  document.getElementById('segment-tag').textContent =
    segs.length + ' files · ' + fmtDur(dur) + ' analyzed' +
    (missingHours.length ? ' · gap: ' + missingHours.join(', ') : '');
}

// ── Render + redraw on resize ───────────────────────────────────
function renderAll() {
  drawCumulative();
  drawHourly();
  drawDuration();
  drawConfidence();
}

renderSegmentTable();

// First paint after layout settles (avoids zero-width canvases)
requestAnimationFrame(() => requestAnimationFrame(renderAll));

let resizeTimer;
window.addEventListener('resize', () => {
  clearTimeout(resizeTimer);
  resizeTimer = setTimeout(renderAll, 120);
});
</script>
</body>
</html>'''


def main():
    if not CH27_JSON.exists():
        print(f"ERROR: {CH27_JSON} not found.")
        sys.exit(1)

    print(f"Loading {CH27_JSON}...")
    data = load_and_compact()

    summary = data["summary"]
    meta = data["metadata"]
    print(f"  Version : {meta.get('version', '?')}")
    print(f"  Duration: {meta.get('duration_sec', 0)/3600:.2f}h")
    print(f"  Total   : {summary.get('total_cycles', 0)} cycles "
          f"(L={summary.get('left_cycles', 0)}, R={summary.get('right_cycles', 0)})")
    print(f"  Events  : {len(data['events'])}")
    print(f"  Segments: {len(data['segments'])}")
    print(f"  Suppressed: {data['suppressed_count']}")

    json_blob = json.dumps(data, separators=(",", ":"))
    html = TEMPLATE.replace("__DATA__", json_blob)
    OUTPUT_HTML.write_text(html)

    size_mb = len(html) / (1024 * 1024)
    print(f"\nWrote {OUTPUT_HTML} ({size_mb:.2f} MB)")


if __name__ == "__main__":
    main()
