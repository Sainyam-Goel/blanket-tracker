#!/usr/bin/env python3
"""Generate dual-camera dashboard HTML with embedded data from CH19 + CH21."""

import json
import sys
from datetime import datetime

CH19_JSON = "cutting_full_v2.json"
CH21_JSON = "blanket_count_1hr_v4.json"
OUTPUT_HTML = "blanket_tracker_dashboard.html"


def load_and_compact():
    """Load both data files and compact for embedding."""
    with open(CH19_JSON) as f:
        ch19 = json.load(f)
    with open(CH21_JSON) as f:
        ch21 = json.load(f)

    ch21_video = ch21["videos"][0]

    dashboard_data = {
        "generated_at": datetime.now().isoformat(),
        "ch19": {
            "metadata": ch19["metadata"],
            "config": ch19["config"],
            "summary": ch19["summary"],
            "events": ch19["events"],
            "breaks": ch19["breaks"],
            "frame_data": ch19["frame_data"][::4],  # ~900 entries
        },
        "ch21": {
            "video_info": ch21_video["video_info"],
            "detection_config": ch21_video["detection_config"],
            "results": ch21_video["results"],
            "source": ch21_video["source"],
            "events": ch21_video["events"],
            "frames": ch21_video["frames"][::100],  # ~899 entries
        },
    }
    return dashboard_data


TEMPLATE = r'''<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Blanket Production Tracker</title>
<link href="https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
<style>
  :root {
    --bg: #0a0a0f;
    --surface: #12121a;
    --card: #1a1a26;
    --border: #2a2a3d;
    --accent: #ff6b35;
    --accent2: #7c3aed;
    --accent3: #10b981;
    --text: #e8e8f0;
    --muted: #6b7280;
    --ch19: #f59e0b;
    --ch21: #3b82f6;
    --red: #ef4444;
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
      linear-gradient(rgba(124,58,237,0.03) 1px, transparent 1px),
      linear-gradient(90deg, rgba(124,58,237,0.03) 1px, transparent 1px);
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

  /* Header */
  header {
    display: flex;
    align-items: flex-start;
    justify-content: space-between;
    margin-bottom: 2.5rem;
    padding-bottom: 1.5rem;
    border-bottom: 1px solid var(--border);
  }

  .logo-area h1 {
    font-size: 2rem;
    font-weight: 800;
    letter-spacing: -0.04em;
    background: linear-gradient(135deg, #ff6b35, #f59e0b);
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

  .header-badges {
    display: flex;
    gap: 0.5rem;
  }

  .duration-badge {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    padding: 0.4rem 0.9rem;
    border-radius: 100px;
    font-size: 0.75rem;
    font-family: 'JetBrains Mono', monospace;
  }

  .badge-ch19 {
    background: rgba(245,158,11,0.1);
    border: 1px solid rgba(245,158,11,0.3);
    color: var(--ch19);
  }

  .badge-ch21 {
    background: rgba(59,130,246,0.1);
    border: 1px solid rgba(59,130,246,0.3);
    color: var(--ch21);
  }

  /* KPI Split Layout */
  .kpi-split {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 1.5rem;
    margin-bottom: 1.5rem;
  }

  .kpi-group {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: 1.2rem;
  }

  .kpi-group-header {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    margin-bottom: 1rem;
    padding-bottom: 0.7rem;
    border-bottom: 1px solid var(--border);
  }

  .kpi-group-dot {
    width: 10px;
    height: 10px;
    border-radius: 50%;
  }

  .kpi-group-title {
    font-size: 0.75rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    font-family: 'JetBrains Mono', monospace;
  }

  .kpi-group-subtitle {
    font-size: 0.65rem;
    color: var(--muted);
    font-family: 'JetBrains Mono', monospace;
    margin-left: auto;
  }

  .kpi-row {
    display: grid;
    grid-template-columns: repeat(2, 1fr);
    gap: 0.8rem;
  }

  .kpi-card {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 1.4rem;
    position: relative;
    overflow: hidden;
    transition: border-color 0.2s;
  }

  .kpi-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
  }

  .kpi-card.orange::before { background: linear-gradient(90deg, #ff6b35, #f59e0b); }
  .kpi-card.amber::before { background: var(--ch19); }
  .kpi-card.green::before { background: var(--accent3); }
  .kpi-card.red::before { background: var(--red); }
  .kpi-card.blue::before { background: var(--ch21); }
  .kpi-card.purple::before { background: var(--accent2); }

  .kpi-label {
    font-size: 0.7rem;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: var(--muted);
    font-family: 'JetBrains Mono', monospace;
    margin-bottom: 0.6rem;
  }

  .kpi-value {
    font-size: 2.8rem;
    font-weight: 800;
    letter-spacing: -0.04em;
    line-height: 1;
  }

  .kpi-sub {
    font-size: 0.72rem;
    color: var(--muted);
    margin-top: 0.4rem;
    font-family: 'JetBrains Mono', monospace;
  }

  .kpi-sub .camera-tag {
    font-size: 0.6rem;
    padding: 0.1rem 0.35rem;
    border-radius: 3px;
    font-weight: 500;
    margin-left: 0.3rem;
  }

  .camera-tag.ch19 {
    background: rgba(245,158,11,0.15);
    color: var(--ch19);
  }

  .camera-tag.ch21 {
    background: rgba(59,130,246,0.15);
    color: var(--ch21);
  }

  /* Panel */
  .panel {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 12px;
    overflow: hidden;
    margin-bottom: 1.5rem;
  }

  .panel-header {
    padding: 1rem 1.4rem;
    border-bottom: 1px solid var(--border);
    display: flex;
    align-items: center;
    justify-content: space-between;
  }

  .panel-title {
    font-size: 0.8rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: var(--text);
  }

  .panel-tag {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.7rem;
    padding: 0.2rem 0.6rem;
    border-radius: 4px;
    font-weight: 500;
  }

  .tag-combined {
    background: rgba(255,107,53,0.12);
    color: var(--accent);
  }

  .tag-ch19 {
    background: rgba(245,158,11,0.15);
    color: var(--ch19);
  }

  .tag-ch21 {
    background: rgba(59,130,246,0.15);
    color: var(--ch21);
  }

  .panel-body { padding: 1.4rem; }

  .chart-area { position: relative; }

  canvas { width: 100%; height: 100%; }

  /* Signal charts side by side */
  .signal-row {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 1.5rem;
    margin-bottom: 1.5rem;
  }

  /* Summary panel */
  .summary-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 1.5rem;
  }

  .summary-card {
    background: var(--surface);
    border-radius: 10px;
    padding: 1.2rem 1.4rem;
  }

  .summary-card h3 {
    font-size: 0.75rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-bottom: 0.8rem;
    display: flex;
    align-items: center;
    gap: 0.5rem;
  }

  .summary-card h3 .dot {
    width: 8px;
    height: 8px;
    border-radius: 50%;
    display: inline-block;
  }

  .summary-card .stat-row {
    display: flex;
    justify-content: space-between;
    padding: 0.35rem 0;
    border-bottom: 1px solid rgba(42,42,61,0.4);
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.75rem;
  }

  .summary-card .stat-row:last-child { border-bottom: none; }

  .stat-label { color: var(--muted); }
  .stat-value { color: var(--text); font-weight: 500; }

  /* Footer */
  footer {
    text-align: center;
    color: var(--muted);
    font-size: 0.7rem;
    font-family: 'JetBrains Mono', monospace;
    padding-top: 1rem;
    border-top: 1px solid var(--border);
  }

  /* Responsive */
  @media (max-width: 900px) {
    .kpi-split { grid-template-columns: 1fr; }
    .kpi-row { grid-template-columns: repeat(2, 1fr); }
    .signal-row { grid-template-columns: 1fr; }
    .summary-grid { grid-template-columns: 1fr; }
    .header-badges { flex-direction: column; }
  }
</style>
</head>
<body>
<div class="container">

  <header>
    <div class="logo-area">
      <h1>Blanket Production Tracker</h1>
      <p id="header-subtitle">Loading...</p>
    </div>
    <div class="header-badges">
      <div class="duration-badge badge-ch19" id="badge-ch19">CH19 --</div>
      <div class="duration-badge badge-ch21" id="badge-ch21">CH21 --</div>
    </div>
  </header>

  <!-- KPI Cards: CH19 left, CH21 right -->
  <div class="kpi-split">
    <!-- CH19 Cutting -->
    <div class="kpi-group">
      <div class="kpi-group-header">
        <span class="kpi-group-dot" style="background:var(--ch19)"></span>
        <span class="kpi-group-title" style="color:var(--ch19)">CH19 — Cutting</span>
        <span class="kpi-group-subtitle" id="ch19-duration-label">--</span>
      </div>
      <div class="kpi-row">
        <div class="kpi-card amber">
          <div class="kpi-label">Cuts Detected</div>
          <div class="kpi-value" id="kpi-cuts" style="color:var(--ch19)">0</div>
          <div class="kpi-sub">total pieces cut</div>
        </div>
        <div class="kpi-card amber">
          <div class="kpi-label">Cut Rate</div>
          <div class="kpi-value" id="kpi-cut-rate" style="color:var(--ch19)">0</div>
          <div class="kpi-sub" id="kpi-cut-rate-sub">cuts/min active</div>
        </div>
        <div class="kpi-card amber">
          <div class="kpi-label">Avg Cycle</div>
          <div class="kpi-value" id="kpi-cut-cycle" style="color:var(--ch19)">0</div>
          <div class="kpi-sub">seconds between cuts</div>
        </div>
        <div class="kpi-card amber">
          <div class="kpi-label">Breaks</div>
          <div class="kpi-value" id="kpi-breaks" style="color:var(--ch19)">0</div>
          <div class="kpi-sub" id="kpi-breaks-sub">idle periods</div>
        </div>
      </div>
    </div>

    <!-- CH21 Weighing -->
    <div class="kpi-group">
      <div class="kpi-group-header">
        <span class="kpi-group-dot" style="background:var(--ch21)"></span>
        <span class="kpi-group-title" style="color:var(--ch21)">CH21 — Weighing</span>
        <span class="kpi-group-subtitle" id="ch21-duration-label">--</span>
      </div>
      <div class="kpi-row">
        <div class="kpi-card green">
          <div class="kpi-label">Accepted</div>
          <div class="kpi-value" id="kpi-accepted" style="color:var(--accent3)">0</div>
          <div class="kpi-sub">weighed on scale</div>
        </div>
        <div class="kpi-card red">
          <div class="kpi-label">Rejected</div>
          <div class="kpi-value" id="kpi-rejected" style="color:var(--red)">0</div>
          <div class="kpi-sub">without weighing</div>
        </div>
        <div class="kpi-card blue">
          <div class="kpi-label">Finish Rate</div>
          <div class="kpi-value" id="kpi-finish-rate" style="color:var(--ch21)">0</div>
          <div class="kpi-sub" id="kpi-finish-rate-sub">blankets/hr</div>
        </div>
        <div class="kpi-card purple">
          <div class="kpi-label">Reject Rate</div>
          <div class="kpi-value" id="kpi-reject-pct" style="color:var(--accent2)">0</div>
          <div class="kpi-sub">of finished blankets</div>
        </div>
      </div>
    </div>
  </div>

  <!-- Combined Production Timeline -->
  <div class="panel">
    <div class="panel-header">
      <span class="panel-title">Production Timeline</span>
      <span class="panel-tag tag-combined" id="timeline-tag">loading...</span>
    </div>
    <div class="panel-body">
      <div class="chart-area" style="height:300px"><canvas id="chart-timeline"></canvas></div>
    </div>
  </div>

  <!-- Side-by-side Signal Charts -->
  <div class="signal-row">
    <div class="panel">
      <div class="panel-header">
        <span class="panel-title">Cutting Signal</span>
        <span class="panel-tag tag-ch19">CH19 derivative</span>
      </div>
      <div class="panel-body">
        <div class="chart-area" style="height:200px"><canvas id="chart-ch19-signal"></canvas></div>
      </div>
    </div>
    <div class="panel">
      <div class="panel-header">
        <span class="panel-title">Finishing Signal</span>
        <span class="panel-tag tag-ch21">CH21 scale + table</span>
      </div>
      <div class="panel-body">
        <div class="chart-area" style="height:200px"><canvas id="chart-ch21-signal"></canvas></div>
      </div>
    </div>
  </div>

  <!-- Production Breakdown -->
  <div class="panel">
    <div class="panel-header">
      <span class="panel-title">Production Breakdown</span>
      <span class="panel-tag tag-combined" id="breakdown-tag">5-min intervals</span>
    </div>
    <div class="panel-body">
      <div class="chart-area" style="height:220px"><canvas id="chart-breakdown"></canvas></div>
    </div>
  </div>

  <!-- Summary Panel -->
  <div class="panel">
    <div class="panel-header">
      <span class="panel-title">Session Summary</span>
      <span class="panel-tag tag-combined" id="summary-tag">--</span>
    </div>
    <div class="panel-body">
      <div class="summary-grid">
        <div class="summary-card" id="summary-ch19"></div>
        <div class="summary-card" id="summary-ch21"></div>
      </div>
    </div>
  </div>

  <footer>
    <p>Blanket Production Tracker v3.0 &nbsp;|&nbsp; Dual-camera detection &nbsp;|&nbsp; <span id="footer-generated">--</span></p>
  </footer>

</div>

<script>
// ══════════════════════════════════════════════════════════════════
// DATA (auto-generated — do not edit manually)
// ══════════════════════════════════════════════════════════════════
const D = %%DASHBOARD_DATA%%;

// ── PARSE DATA ──────────────────────────────────────────────────
const ch19 = D.ch19;
const ch21 = D.ch21;

// CH19 derived
const ch19Events = ch19.events;  // cut events
const ch19Breaks = ch19.breaks;
const ch19Frames = ch19.frame_data;
const ch19Duration = ch19.metadata.duration_sec;
const ch19Cuts = ch19.summary.total_cuts;
const ch19Rate = ch19.summary.cuts_per_minute;
const ch19AvgCycle = ch19.summary.avg_cycle_sec;
const ch19ActiveTime = ch19.summary.active_time_sec;
const ch19BreakTime = ch19.summary.break_time_sec;

// CH21 derived
const ch21Events = ch21.events;
const ch21Frames = ch21.frames;
const ch21Duration = ch21.video_info.duration_sec;
const ch21Config = ch21.detection_config;
const ch21Accepted = ch21.results.accepted;
const ch21Rejected = ch21.results.rejected;
const ch21Total = ch21.results.total_blankets;
const ch21Rate = ch21Total / (ch21Duration / 3600);

// CH21 event filtering
const scaleEvents = ch21Events.filter(e => e.type === 'scale_cycle_complete');
const acceptedEvents = ch21Events.filter(e => e.type === 'blanket_accepted');
const rejectedEvents = ch21Events.filter(e => e.type === 'blanket_rejected');
const ch21CountEvents = acceptedEvents.length > 0 ? acceptedEvents : scaleEvents;
const ch21AllFinished = [...ch21CountEvents, ...rejectedEvents].sort((a, b) => a.time_sec - b.time_sec);

// Shared
const maxDuration = Math.max(ch19Duration, ch21Duration);

// CH21 avg cycle
let ch21AvgCycle = 0;
if (ch21AllFinished.length > 1) {
  let totalGap = 0;
  for (let i = 1; i < ch21AllFinished.length; i++) {
    totalGap += ch21AllFinished[i].time_sec - ch21AllFinished[i - 1].time_sec;
  }
  ch21AvgCycle = totalGap / (ch21AllFinished.length - 1);
}

// Peak 5-min rate for CH21
let ch21Peak5 = 0;
if (ch21AllFinished.length > 0) {
  for (let i = 0; i < ch21AllFinished.length; i++) {
    const wEnd = ch21AllFinished[i].time_sec + 300;
    let count = 0;
    for (let j = i; j < ch21AllFinished.length && ch21AllFinished[j].time_sec <= wEnd; j++) count++;
    if (count > ch21Peak5) ch21Peak5 = count;
  }
  ch21Peak5 = Math.round(ch21Peak5 * 12);
}

// Reject rate
const rejectPct = ch21Total > 0 ? (ch21Rejected / ch21Total * 100) : 0;

// ── HEADER & BADGES ─────────────────────────────────────────────
const fmtDur = (s) => s >= 3600 ? Math.round(s / 3600) + 'hr' : Math.round(s / 60) + 'min';
document.getElementById('header-subtitle').textContent =
  'CH19 Cutting + CH21 Finishing \u00b7 Panipat Factory';
document.getElementById('badge-ch19').textContent = 'CH19 ' + fmtDur(ch19Duration);
document.getElementById('badge-ch21').textContent = 'CH21 ' + fmtDur(ch21Duration);

// ── ANIMATE KPIs ────────────────────────────────────────────────
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

// CH19 KPIs
animateCount(document.getElementById('kpi-cuts'), ch19Cuts);
animateCount(document.getElementById('kpi-cut-rate'), ch19Rate, 1200, 1);
animateCount(document.getElementById('kpi-cut-cycle'), ch19AvgCycle, 1200, 1);
const ch19BreakCountKPI = Math.floor(ch19Breaks.length / 2);
animateCount(document.getElementById('kpi-breaks'), ch19BreakCountKPI);
document.getElementById('kpi-cut-cycle').insertAdjacentText('afterend', 's');
document.getElementById('kpi-breaks-sub').textContent =
  (ch19BreakTime / 60).toFixed(1) + ' min idle';
document.getElementById('ch19-duration-label').textContent =
  fmtDur(ch19Duration) + ' analyzed';

// CH21 KPIs
animateCount(document.getElementById('kpi-accepted'), ch21Accepted);
animateCount(document.getElementById('kpi-rejected'), ch21Rejected);
animateCount(document.getElementById('kpi-finish-rate'), Math.round(ch21Rate));
animateCount(document.getElementById('kpi-reject-pct'), rejectPct, 1200, 1);
document.getElementById('kpi-reject-pct').insertAdjacentText('afterend', '%');
document.getElementById('ch21-duration-label').textContent =
  fmtDur(ch21Duration) + ' analyzed';

document.getElementById('timeline-tag').textContent =
  ch19Cuts + ' cuts + ' + ch21Total + ' blankets';

document.getElementById('footer-generated').textContent =
  'Generated ' + D.generated_at.replace('T', ' ').split('.')[0];

// ── CANVAS HELPERS ──────────────────────────────────────────────
function setupCanvas(canvasId, height) {
  const canvas = document.getElementById(canvasId);
  const dpr = window.devicePixelRatio || 1;
  const rect = canvas.parentElement.getBoundingClientRect();
  canvas.width = rect.width * dpr;
  canvas.height = height * dpr;
  canvas.style.width = rect.width + 'px';
  canvas.style.height = height + 'px';
  const ctx = canvas.getContext('2d');
  ctx.scale(dpr, dpr);
  return { ctx, W: rect.width, H: height };
}

function getTimeAxisConfig(maxSec) {
  const hrs = maxSec / 3600;
  if (maxSec <= 30) return { unit: 's', step: 1, divisor: 1 };
  if (hrs <= 1) return { unit: 'm', step: 5, divisor: 60 };
  if (hrs <= 4) return { unit: 'm', step: 15, divisor: 60 };
  if (hrs <= 8) return { unit: 'm', step: 30, divisor: 60 };
  return { unit: 'h', step: 1, divisor: 3600 };
}

function drawTimeAxis(ctx, pad, cW, H, maxT) {
  const cfg = getTimeAxisConfig(maxT);
  ctx.fillStyle = 'rgba(107,114,128,0.7)';
  ctx.font = '10px JetBrains Mono, monospace';
  ctx.textAlign = 'center';
  const stepSec = cfg.step * cfg.divisor;
  for (let t = 0; t <= maxT; t += stepSec) {
    const x = pad.left + (t / maxT) * cW;
    ctx.fillText(Math.round(t / cfg.divisor) + cfg.unit, x, H - 5);
  }
}

function drawGridH(ctx, pad, cW, cH, maxVal, steps) {
  ctx.strokeStyle = 'rgba(42,42,61,0.8)';
  ctx.lineWidth = 1;
  ctx.fillStyle = 'rgba(107,114,128,0.7)';
  ctx.font = '10px JetBrains Mono, monospace';
  ctx.textAlign = 'right';
  for (let i = 0; i <= steps; i++) {
    const y = pad.top + cH - (i / steps) * cH;
    ctx.beginPath();
    ctx.moveTo(pad.left, y);
    ctx.lineTo(pad.left + cW, y);
    ctx.stroke();
    ctx.fillText(Math.round(i / steps * maxVal), pad.left - 5, y + 4);
  }
}

// ══════════════════════════════════════════════════════════════════
// CHART: COMBINED PRODUCTION TIMELINE (dual Y-axis)
// ══════════════════════════════════════════════════════════════════
function drawTimeline() {
  const { ctx, W, H } = setupCanvas('chart-timeline', 300);
  const pad = { top: 25, right: 55, bottom: 30, left: 55 };
  const cW = W - pad.left - pad.right;
  const cH = H - pad.top - pad.bottom;
  const maxT = maxDuration;

  // CH19 break bands
  for (let i = 0; i < ch19Breaks.length; i += 2) {
    if (i + 1 < ch19Breaks.length) {
      const x1 = pad.left + (ch19Breaks[i].time_sec / maxT) * cW;
      const x2 = pad.left + (ch19Breaks[i + 1].time_sec / maxT) * cW;
      ctx.fillStyle = 'rgba(245,158,11,0.06)';
      ctx.fillRect(x1, pad.top, x2 - x1, cH);
    }
  }

  // CH21 idle gaps (>60s between blankets)
  for (let i = 1; i < ch21AllFinished.length; i++) {
    const gap = ch21AllFinished[i].time_sec - ch21AllFinished[i - 1].time_sec;
    if (gap > 60) {
      const x1 = pad.left + (ch21AllFinished[i - 1].time_sec / maxT) * cW;
      const x2 = pad.left + (ch21AllFinished[i].time_sec / maxT) * cW;
      ctx.fillStyle = 'rgba(59,130,246,0.05)';
      ctx.fillRect(x1, pad.top, x2 - x1, cH);
    }
  }

  // Left Y-axis grid (CH19 cuts)
  const maxCuts = ch19Cuts;
  const gridSteps = 5;
  ctx.strokeStyle = 'rgba(42,42,61,0.6)';
  ctx.lineWidth = 1;
  ctx.fillStyle = 'rgba(245,158,11,0.6)';
  ctx.font = '10px JetBrains Mono, monospace';
  ctx.textAlign = 'right';
  for (let i = 0; i <= gridSteps; i++) {
    const y = pad.top + cH - (i / gridSteps) * cH;
    ctx.beginPath();
    ctx.moveTo(pad.left, y);
    ctx.lineTo(pad.left + cW, y);
    ctx.stroke();
    ctx.fillText(Math.round(i / gridSteps * maxCuts), pad.left - 5, y + 4);
  }

  // Right Y-axis labels (CH21 total)
  const maxBlankets = ch21Total;
  ctx.fillStyle = 'rgba(16,185,129,0.6)';
  ctx.textAlign = 'left';
  for (let i = 0; i <= gridSteps; i++) {
    const y = pad.top + cH - (i / gridSteps) * cH;
    ctx.fillText(Math.round(i / gridSteps * maxBlankets), pad.left + cW + 5, y + 4);
  }

  // Time axis
  drawTimeAxis(ctx, pad, cW, H, maxT);

  // ── CH19 cuts step line (amber) ──
  const ch19Grad = ctx.createLinearGradient(0, pad.top, 0, pad.top + cH);
  ch19Grad.addColorStop(0, 'rgba(245,158,11,0.15)');
  ch19Grad.addColorStop(1, 'rgba(245,158,11,0.01)');

  // Fill
  ctx.beginPath();
  ctx.moveTo(pad.left, pad.top + cH);
  let cutCount = 0;
  for (const ev of ch19Events) {
    const x = pad.left + (ev.time_sec / maxT) * cW;
    ctx.lineTo(x, pad.top + cH - (cutCount / maxCuts) * cH);
    cutCount++;
    ctx.lineTo(x, pad.top + cH - (cutCount / maxCuts) * cH);
  }
  ctx.lineTo(pad.left + cW, pad.top + cH - (cutCount / maxCuts) * cH);
  ctx.lineTo(pad.left + cW, pad.top + cH);
  ctx.closePath();
  ctx.fillStyle = ch19Grad;
  ctx.fill();

  // Stroke
  ctx.beginPath();
  ctx.strokeStyle = 'rgba(245,158,11,0.8)';
  ctx.lineWidth = 2;
  cutCount = 0;
  ctx.moveTo(pad.left, pad.top + cH);
  for (const ev of ch19Events) {
    const x = pad.left + (ev.time_sec / maxT) * cW;
    ctx.lineTo(x, pad.top + cH - (cutCount / maxCuts) * cH);
    cutCount++;
    ctx.lineTo(x, pad.top + cH - (cutCount / maxCuts) * cH);
  }
  ctx.lineTo(pad.left + cW, pad.top + cH - (cutCount / maxCuts) * cH);
  ctx.stroke();

  // ── CH21 blankets step line (green, mapped to right Y) ──
  const ch21Grad = ctx.createLinearGradient(0, pad.top, 0, pad.top + cH);
  ch21Grad.addColorStop(0, 'rgba(16,185,129,0.15)');
  ch21Grad.addColorStop(1, 'rgba(16,185,129,0.01)');

  // Fill
  ctx.beginPath();
  ctx.moveTo(pad.left, pad.top + cH);
  let blkCount = 0;
  for (const ev of ch21AllFinished) {
    const x = pad.left + (ev.time_sec / maxT) * cW;
    ctx.lineTo(x, pad.top + cH - (blkCount / maxBlankets) * cH);
    blkCount++;
    ctx.lineTo(x, pad.top + cH - (blkCount / maxBlankets) * cH);
  }
  ctx.lineTo(pad.left + cW, pad.top + cH - (blkCount / maxBlankets) * cH);
  ctx.lineTo(pad.left + cW, pad.top + cH);
  ctx.closePath();
  ctx.fillStyle = ch21Grad;
  ctx.fill();

  // Stroke
  ctx.beginPath();
  ctx.strokeStyle = 'rgba(16,185,129,0.8)';
  ctx.lineWidth = 2;
  blkCount = 0;
  ctx.moveTo(pad.left, pad.top + cH);
  for (const ev of ch21AllFinished) {
    const x = pad.left + (ev.time_sec / maxT) * cW;
    ctx.lineTo(x, pad.top + cH - (blkCount / maxBlankets) * cH);
    blkCount++;
    ctx.lineTo(x, pad.top + cH - (blkCount / maxBlankets) * cH);
  }
  ctx.lineTo(pad.left + cW, pad.top + cH - (blkCount / maxBlankets) * cH);
  ctx.stroke();

  // CH21 rejected ticks
  if (rejectedEvents.length > 0) {
    ctx.strokeStyle = 'rgba(239,68,68,0.5)';
    ctx.lineWidth = 1;
    for (const ev of rejectedEvents) {
      const x = pad.left + (ev.time_sec / maxT) * cW;
      ctx.beginPath();
      ctx.moveTo(x, pad.top + cH - 5);
      ctx.lineTo(x, pad.top + cH + 2);
      ctx.stroke();
    }
  }

  // End labels
  const xEnd = pad.left + cW;
  ctx.font = 'bold 11px JetBrains Mono, monospace';
  ctx.textAlign = 'right';
  const yCuts = pad.top + cH - (ch19Cuts / maxCuts) * cH;
  ctx.fillStyle = 'rgba(245,158,11,0.9)';
  ctx.fillText(ch19Cuts + ' cuts', xEnd - 10, yCuts - 6);
  const yBlk = pad.top + cH - (ch21Total / maxBlankets) * cH;
  ctx.fillStyle = 'rgba(16,185,129,0.9)';
  ctx.fillText(ch21Accepted + ' accepted', xEnd - 10, yBlk + 16);
  if (ch21Rejected > 0) {
    ctx.fillStyle = 'rgba(239,68,68,0.9)';
    ctx.fillText(ch21Rejected + ' rejected', xEnd - 10, yBlk + 28);
  }

  // Legend
  ctx.font = '9px JetBrains Mono, monospace';
  ctx.textAlign = 'left';
  let lx = pad.left + 5;
  const ly = pad.top + 8;
  ctx.fillStyle = 'rgba(245,158,11,0.8)';
  ctx.fillRect(lx, ly, 14, 3);
  ctx.fillStyle = 'rgba(200,200,220,0.7)';
  ctx.fillText('CH19 Cuts', lx + 18, ly + 4);
  lx += 85;
  ctx.fillStyle = 'rgba(16,185,129,0.8)';
  ctx.fillRect(lx, ly, 14, 3);
  ctx.fillStyle = 'rgba(200,200,220,0.7)';
  ctx.fillText('CH21 Blankets', lx + 18, ly + 4);
  lx += 105;
  ctx.fillStyle = 'rgba(239,68,68,0.5)';
  ctx.fillRect(lx, ly, 14, 2);
  ctx.fillStyle = 'rgba(200,200,220,0.7)';
  ctx.fillText('Rejected', lx + 18, ly + 4);
  lx += 70;
  ctx.fillStyle = 'rgba(245,158,11,0.15)';
  ctx.fillRect(lx, ly - 2, 14, 8);
  ctx.fillStyle = 'rgba(200,200,220,0.7)';
  ctx.fillText('Break', lx + 18, ly + 4);

  // Y-axis labels
  ctx.save();
  ctx.font = '9px JetBrains Mono, monospace';
  ctx.fillStyle = 'rgba(245,158,11,0.6)';
  ctx.translate(12, pad.top + cH / 2);
  ctx.rotate(-Math.PI / 2);
  ctx.textAlign = 'center';
  ctx.fillText('CH19 Cuts', 0, 0);
  ctx.restore();

  ctx.save();
  ctx.fillStyle = 'rgba(16,185,129,0.6)';
  ctx.translate(W - 12, pad.top + cH / 2);
  ctx.rotate(Math.PI / 2);
  ctx.textAlign = 'center';
  ctx.fillText('CH21 Blankets', 0, 0);
  ctx.restore();
}

// ══════════════════════════════════════════════════════════════════
// CHART: CH19 CUTTING SIGNAL (derivative)
// ══════════════════════════════════════════════════════════════════
function drawCH19Signal() {
  const { ctx, W, H } = setupCanvas('chart-ch19-signal', 200);
  const pad = { top: 20, right: 15, bottom: 30, left: 45 };
  const cW = W - pad.left - pad.right;
  const cH = H - pad.top - pad.bottom;
  const maxT = ch19Duration;
  const maxDeriv = 120;
  const threshold = ch19.config.deriv_threshold || 20;

  // Break bands
  for (let i = 0; i < ch19Breaks.length; i += 2) {
    if (i + 1 < ch19Breaks.length) {
      const x1 = pad.left + (ch19Breaks[i].time_sec / maxT) * cW;
      const x2 = pad.left + (ch19Breaks[i + 1].time_sec / maxT) * cW;
      ctx.fillStyle = 'rgba(245,158,11,0.08)';
      ctx.fillRect(x1, pad.top, x2 - x1, cH);
    }
  }

  // Grid
  drawGridH(ctx, pad, cW, cH, maxDeriv, 4);
  drawTimeAxis(ctx, pad, cW, H, maxT);

  // Threshold line
  const threshY = pad.top + cH - (threshold / maxDeriv) * cH;
  ctx.setLineDash([6, 4]);
  ctx.strokeStyle = 'rgba(239,68,68,0.5)';
  ctx.lineWidth = 1.5;
  ctx.beginPath();
  ctx.moveTo(pad.left, threshY);
  ctx.lineTo(pad.left + cW, threshY);
  ctx.stroke();
  ctx.setLineDash([]);
  ctx.fillStyle = 'rgba(239,68,68,0.7)';
  ctx.font = '9px JetBrains Mono, monospace';
  ctx.textAlign = 'right';
  ctx.fillText('thresh=' + threshold, pad.left + cW - 2, threshY - 4);

  // Downsample for smooth plotting
  const maxPts = 300;
  const step = ch19Frames.length > maxPts ? Math.ceil(ch19Frames.length / maxPts) : 1;
  const sampled = ch19Frames.filter((_, i) => i % step === 0 || i === ch19Frames.length - 1);

  // Derivative fill
  const derivGrad = ctx.createLinearGradient(0, pad.top, 0, pad.top + cH);
  derivGrad.addColorStop(0, 'rgba(245,158,11,0.25)');
  derivGrad.addColorStop(1, 'rgba(245,158,11,0.02)');
  ctx.beginPath();
  sampled.forEach((f, i) => {
    const x = pad.left + (f.time_sec / maxT) * cW;
    const val = Math.max(0, Math.min(f.derivative, maxDeriv));
    const y = pad.top + cH - (val / maxDeriv) * cH;
    i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
  });
  ctx.lineTo(pad.left + (sampled[sampled.length - 1].time_sec / maxT) * cW, pad.top + cH);
  ctx.lineTo(pad.left + (sampled[0].time_sec / maxT) * cW, pad.top + cH);
  ctx.closePath();
  ctx.fillStyle = derivGrad;
  ctx.fill();

  // Derivative line
  ctx.beginPath();
  ctx.strokeStyle = 'rgba(245,158,11,0.8)';
  ctx.lineWidth = 1.5;
  ctx.lineJoin = 'round';
  sampled.forEach((f, i) => {
    const x = pad.left + (f.time_sec / maxT) * cW;
    const val = Math.max(0, Math.min(f.derivative, maxDeriv));
    const y = pad.top + cH - (val / maxDeriv) * cH;
    i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
  });
  ctx.stroke();

  // Cut event markers (small dots along top)
  const labelEvery = ch19Cuts > 100 ? 50 : ch19Cuts > 50 ? 25 : 10;
  ch19Events.forEach((ev, idx) => {
    const num = idx + 1;
    const x = pad.left + (ev.time_sec / maxT) * cW;
    ctx.strokeStyle = 'rgba(245,158,11,0.25)';
    ctx.lineWidth = 0.5;
    ctx.setLineDash([2, 3]);
    ctx.beginPath();
    ctx.moveTo(x, pad.top);
    ctx.lineTo(x, pad.top + cH);
    ctx.stroke();
    ctx.setLineDash([]);
    if (num % labelEvery === 0 || num === ch19Cuts) {
      ctx.fillStyle = 'rgba(245,158,11,0.9)';
      ctx.beginPath();
      ctx.arc(x, pad.top + 8, 6, 0, Math.PI * 2);
      ctx.fill();
      ctx.fillStyle = '#000';
      ctx.font = '7px JetBrains Mono, monospace';
      ctx.textAlign = 'center';
      ctx.fillText(num, x, pad.top + 11);
    }
  });

  // Legend
  ctx.font = '9px JetBrains Mono, monospace';
  ctx.textAlign = 'left';
  let lx = pad.left + 5;
  const ly = pad.top + cH + 16;
  ctx.fillStyle = 'rgba(245,158,11,0.8)';
  ctx.fillRect(lx, ly - 3, 12, 3);
  ctx.fillStyle = 'rgba(200,200,220,0.8)';
  ctx.fillText('Brightness derivative', lx + 15, ly);
}

// ══════════════════════════════════════════════════════════════════
// CHART: CH21 FINISHING SIGNAL (scale + table)
// ══════════════════════════════════════════════════════════════════
function drawCH21Signal() {
  const { ctx, W, H } = setupCanvas('chart-ch21-signal', 200);
  const pad = { top: 20, right: 15, bottom: 30, left: 45 };
  const cW = W - pad.left - pad.right;
  const cH = H - pad.top - pad.bottom;
  const maxT = ch21Frames[ch21Frames.length - 1].time_sec;
  const maxTexture = 120;
  const threshold = ch21Config.scale_on_threshold || 25;

  // Background shading for loaded state
  let inLoaded = false;
  let loadedStart = 0;
  for (const f of ch21Frames) {
    const wasLoaded = inLoaded;
    inLoaded = f.scale_state === 'loaded';
    if (inLoaded && !wasLoaded) loadedStart = f.time_sec;
    if (!inLoaded && wasLoaded) {
      const x1 = pad.left + (loadedStart / maxT) * cW;
      const x2 = pad.left + (f.time_sec / maxT) * cW;
      ctx.fillStyle = 'rgba(59,130,246,0.08)';
      ctx.fillRect(x1, pad.top, x2 - x1, cH);
    }
  }

  // Grid
  drawGridH(ctx, pad, cW, cH, maxTexture, 4);
  drawTimeAxis(ctx, pad, cW, H, maxT);

  // Threshold line
  const threshY = pad.top + cH - (threshold / maxTexture) * cH;
  ctx.setLineDash([6, 4]);
  ctx.strokeStyle = 'rgba(255,107,53,0.6)';
  ctx.lineWidth = 1.5;
  ctx.beginPath();
  ctx.moveTo(pad.left, threshY);
  ctx.lineTo(pad.left + cW, threshY);
  ctx.stroke();
  ctx.setLineDash([]);
  ctx.fillStyle = 'rgba(255,107,53,0.8)';
  ctx.font = '9px JetBrains Mono, monospace';
  ctx.textAlign = 'left';
  ctx.fillText('ON=' + threshold, pad.left + cW - 50, threshY - 5);

  // Downsample
  const maxPts = 300;
  const step = ch21Frames.length > maxPts ? Math.ceil(ch21Frames.length / maxPts) : 1;
  const sampled = ch21Frames.filter((_, i) => i % step === 0 || i === ch21Frames.length - 1);

  // Table texture fill
  const tableGrad = ctx.createLinearGradient(0, pad.top, 0, pad.top + cH);
  tableGrad.addColorStop(0, 'rgba(59,130,246,0.25)');
  tableGrad.addColorStop(1, 'rgba(59,130,246,0.02)');
  ctx.beginPath();
  sampled.forEach((f, i) => {
    const x = pad.left + (f.time_sec / maxT) * cW;
    const y = pad.top + cH - (Math.min(f.table_texture, maxTexture) / maxTexture) * cH;
    i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
  });
  ctx.lineTo(pad.left + (sampled[sampled.length - 1].time_sec / maxT) * cW, pad.top + cH);
  ctx.lineTo(pad.left + (sampled[0].time_sec / maxT) * cW, pad.top + cH);
  ctx.closePath();
  ctx.fillStyle = tableGrad;
  ctx.fill();

  // Table texture line
  ctx.beginPath();
  ctx.strokeStyle = 'rgb(59,130,246)';
  ctx.lineWidth = 1.5;
  ctx.lineJoin = 'round';
  sampled.forEach((f, i) => {
    const x = pad.left + (f.time_sec / maxT) * cW;
    const y = pad.top + cH - (Math.min(f.table_texture, maxTexture) / maxTexture) * cH;
    i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
  });
  ctx.stroke();

  // Scale diff line
  ctx.beginPath();
  ctx.strokeStyle = 'rgba(245,158,11,0.6)';
  ctx.lineWidth = 1;
  sampled.forEach((f, i) => {
    const x = pad.left + (f.time_sec / maxT) * cW;
    const y = pad.top + cH - (Math.min(f.scale_diff, maxTexture) / maxTexture) * cH;
    i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
  });
  ctx.stroke();

  // Blanket count markers
  const countEvts = scaleEvents.length > 0 ? scaleEvents : [];
  const labelEvery = ch21Total > 100 ? 50 : ch21Total > 50 ? 25 : 10;
  countEvts.forEach((ev, idx) => {
    const num = idx + 1;
    const x = pad.left + (ev.time_sec / maxT) * cW;
    ctx.strokeStyle = 'rgba(16,185,129,0.2)';
    ctx.lineWidth = 0.5;
    ctx.setLineDash([2, 3]);
    ctx.beginPath();
    ctx.moveTo(x, pad.top);
    ctx.lineTo(x, pad.top + cH);
    ctx.stroke();
    ctx.setLineDash([]);
    if (num % labelEvery === 0 || num === countEvts.length) {
      ctx.fillStyle = 'rgb(16,185,129)';
      ctx.beginPath();
      ctx.arc(x, pad.top + 8, 6, 0, Math.PI * 2);
      ctx.fill();
      ctx.fillStyle = '#fff';
      ctx.font = '7px JetBrains Mono, monospace';
      ctx.textAlign = 'center';
      ctx.fillText(num, x, pad.top + 11);
    }
  });

  // Legend
  ctx.font = '9px JetBrains Mono, monospace';
  ctx.textAlign = 'left';
  let lx = pad.left + 5;
  const ly = pad.top + cH + 16;
  ctx.fillStyle = 'rgb(59,130,246)';
  ctx.fillRect(lx, ly - 3, 12, 3);
  ctx.fillStyle = 'rgba(200,200,220,0.8)';
  ctx.fillText('Table texture', lx + 15, ly);
  lx += 95;
  ctx.fillStyle = 'rgba(245,158,11,0.6)';
  ctx.fillRect(lx, ly - 3, 12, 3);
  ctx.fillStyle = 'rgba(200,200,220,0.8)';
  ctx.fillText('Scale diff', lx + 15, ly);
}

// ══════════════════════════════════════════════════════════════════
// CHART: PRODUCTION BREAKDOWN (3-series bars)
// ══════════════════════════════════════════════════════════════════
function drawBreakdown() {
  const { ctx, W, H } = setupCanvas('chart-breakdown', 220);
  const pad = { top: 20, right: 15, bottom: 30, left: 50 };
  const cW = W - pad.left - pad.right;
  const cH = H - pad.top - pad.bottom;

  // Adaptive bucket sizing
  let bucketSec, bucketLabel;
  const durationHrs = maxDuration / 3600;
  if (durationHrs < 2) { bucketSec = 300; bucketLabel = '5-min'; }
  else if (durationHrs < 6) { bucketSec = 900; bucketLabel = '15-min'; }
  else { bucketSec = 1800; bucketLabel = '30-min'; }

  document.getElementById('breakdown-tag').textContent = bucketLabel + ' intervals';

  const numBuckets = Math.ceil(maxDuration / bucketSec);
  const bucketsCuts = new Array(numBuckets).fill(0);
  const bucketsAcc = new Array(numBuckets).fill(0);
  const bucketsRej = new Array(numBuckets).fill(0);

  for (const ev of ch19Events) {
    const idx = Math.min(Math.floor(ev.time_sec / bucketSec), numBuckets - 1);
    bucketsCuts[idx]++;
  }
  const accEvts = acceptedEvents.length > 0 ? acceptedEvents : scaleEvents;
  for (const ev of accEvts) {
    const idx = Math.min(Math.floor(ev.time_sec / bucketSec), numBuckets - 1);
    bucketsAcc[idx]++;
  }
  for (const ev of rejectedEvents) {
    const idx = Math.min(Math.floor(ev.time_sec / bucketSec), numBuckets - 1);
    bucketsRej[idx]++;
  }

  const maxBucket = Math.max(...bucketsCuts.map((c, i) => Math.max(c, bucketsAcc[i] + bucketsRej[i])), 1);

  // Grid
  drawGridH(ctx, pad, cW, cH, maxBucket, 4);

  // Each bucket has 3 bars side-by-side
  const groupW = cW / numBuckets;
  const barW = groupW * 0.25;
  const gapInner = groupW * 0.03;
  const groupPad = groupW * 0.11;

  for (let i = 0; i < numBuckets; i++) {
    const gx = pad.left + i * groupW + groupPad;

    // Bar 1: CH19 cuts (amber)
    if (bucketsCuts[i] > 0) {
      const h = (bucketsCuts[i] / maxBucket) * cH;
      ctx.fillStyle = 'rgba(245,158,11,0.75)';
      ctx.beginPath();
      ctx.roundRect(gx, pad.top + cH - h, barW, h, [3, 3, 0, 0]);
      ctx.fill();
      ctx.fillStyle = 'rgba(245,158,11,0.9)';
      ctx.font = '8px JetBrains Mono, monospace';
      ctx.textAlign = 'center';
      ctx.fillText(bucketsCuts[i], gx + barW / 2, pad.top + cH - h - 3);
    }

    // Bar 2: CH21 accepted (purple)
    const x2 = gx + barW + gapInner;
    if (bucketsAcc[i] > 0) {
      const h = (bucketsAcc[i] / maxBucket) * cH;
      const accGrad = ctx.createLinearGradient(0, pad.top + cH - h, 0, pad.top + cH);
      accGrad.addColorStop(0, 'rgba(124,58,237,0.9)');
      accGrad.addColorStop(1, 'rgba(124,58,237,0.4)');
      ctx.fillStyle = accGrad;
      const r = bucketsRej[i] > 0 ? [0,0,0,0] : [3,3,0,0];
      ctx.beginPath();
      ctx.roundRect(x2, pad.top + cH - h, barW, h, r);
      ctx.fill();
    }

    // Bar 2b: CH21 rejected (red, stacked on accepted)
    if (bucketsRej[i] > 0) {
      const accH = (bucketsAcc[i] / maxBucket) * cH;
      const rejH = (bucketsRej[i] / maxBucket) * cH;
      ctx.fillStyle = 'rgba(239,68,68,0.7)';
      ctx.beginPath();
      ctx.roundRect(x2, pad.top + cH - accH - rejH, barW, rejH, [3, 3, 0, 0]);
      ctx.fill();
    }

    // Acc+Rej total on top of stacked bar
    const ch21BucketTotal = bucketsAcc[i] + bucketsRej[i];
    if (ch21BucketTotal > 0) {
      const totalH = (ch21BucketTotal / maxBucket) * cH;
      ctx.fillStyle = 'rgba(200,200,220,0.8)';
      ctx.font = '8px JetBrains Mono, monospace';
      ctx.textAlign = 'center';
      ctx.fillText(ch21BucketTotal, x2 + barW / 2, pad.top + cH - totalH - 3);
    }
  }

  // X-axis labels
  const timeCfg = getTimeAxisConfig(maxDuration);
  ctx.fillStyle = 'rgba(107,114,128,0.7)';
  ctx.font = '10px JetBrains Mono, monospace';
  ctx.textAlign = 'center';
  const labelEvery = numBuckets > 24 ? 4 : numBuckets > 12 ? 2 : 1;
  for (let i = 0; i < numBuckets; i += labelEvery) {
    const tSec = i * bucketSec;
    const x = pad.left + (i + 0.5) * groupW;
    ctx.fillText(Math.round(tSec / timeCfg.divisor) + timeCfg.unit, x, H - 5);
  }

  // Legend
  ctx.font = '9px JetBrains Mono, monospace';
  ctx.textAlign = 'left';
  let lx = pad.left + 5;
  const ly = pad.top + 5;
  ctx.fillStyle = 'rgba(245,158,11,0.75)';
  ctx.fillRect(lx, ly, 10, 8);
  ctx.fillStyle = 'rgba(200,200,220,0.7)';
  ctx.fillText('Cuts', lx + 14, ly + 7);
  lx += 45;
  ctx.fillStyle = 'rgba(124,58,237,0.7)';
  ctx.fillRect(lx, ly, 10, 8);
  ctx.fillStyle = 'rgba(200,200,220,0.7)';
  ctx.fillText('Accepted', lx + 14, ly + 7);
  lx += 65;
  ctx.fillStyle = 'rgba(239,68,68,0.7)';
  ctx.fillRect(lx, ly, 10, 8);
  ctx.fillStyle = 'rgba(200,200,220,0.7)';
  ctx.fillText('Rejected', lx + 14, ly + 7);
}

// ══════════════════════════════════════════════════════════════════
// SUMMARY PANEL
// ══════════════════════════════════════════════════════════════════
function renderSummary() {
  // CH19 summary
  const ch19BreakCount = Math.floor(ch19Breaks.length / 2);
  const ch19ActiveMin = (ch19ActiveTime / 60).toFixed(1);
  const ch19BreakMin = (ch19BreakTime / 60).toFixed(1);

  document.getElementById('summary-ch19').innerHTML = `
    <h3><span class="dot" style="background:var(--ch19)"></span> CH19 — Cutting Table</h3>
    <div class="stat-row"><span class="stat-label">Total Cuts</span><span class="stat-value" style="color:var(--ch19)">${ch19Cuts}</span></div>
    <div class="stat-row"><span class="stat-label">Active Rate</span><span class="stat-value">${ch19Rate}/min</span></div>
    <div class="stat-row"><span class="stat-label">Avg Cycle</span><span class="stat-value">${ch19AvgCycle}s</span></div>
    <div class="stat-row"><span class="stat-label">Active Time</span><span class="stat-value">${ch19ActiveMin} min</span></div>
    <div class="stat-row"><span class="stat-label">Breaks</span><span class="stat-value">${ch19BreakCount} (${ch19BreakMin} min)</span></div>
    <div class="stat-row"><span class="stat-label">Duration</span><span class="stat-value">${(ch19Duration / 60).toFixed(1)} min</span></div>
  `;

  // CH21 summary
  const ch21DurMin = (ch21Duration / 60).toFixed(1);
  const tableEvents = ch21Events.filter(e => e.type === 'table_blanket_off');

  document.getElementById('summary-ch21').innerHTML = `
    <h3><span class="dot" style="background:var(--ch21)"></span> CH21 — Finishing Station</h3>
    <div class="stat-row"><span class="stat-label">Total Blankets</span><span class="stat-value" style="color:var(--ch21)">${ch21Total}</span></div>
    <div class="stat-row"><span class="stat-label">Accepted</span><span class="stat-value" style="color:var(--accent3)">${ch21Accepted}</span></div>
    <div class="stat-row"><span class="stat-label">Rejected</span><span class="stat-value" style="color:var(--red)">${ch21Rejected}</span></div>
    <div class="stat-row"><span class="stat-label">Reject Rate</span><span class="stat-value">${rejectPct.toFixed(1)}%</span></div>
    <div class="stat-row"><span class="stat-label">Hourly Rate</span><span class="stat-value">${Math.round(ch21Rate)}/hr</span></div>
    <div class="stat-row"><span class="stat-label">Avg Cycle</span><span class="stat-value">${ch21AvgCycle.toFixed(1)}s</span></div>
    <div class="stat-row"><span class="stat-label">Peak (5-min)</span><span class="stat-value">${ch21Peak5}/hr</span></div>
    <div class="stat-row"><span class="stat-label">Table Cycles</span><span class="stat-value">${tableEvents.length}</span></div>
    <div class="stat-row"><span class="stat-label">Duration</span><span class="stat-value">${ch21DurMin} min</span></div>
  `;

  document.getElementById('summary-tag').textContent =
    (ch19Duration / 60).toFixed(0) + ' + ' + (ch21Duration / 60).toFixed(0) + ' min analyzed';
}

// ── INIT ────────────────────────────────────────────────────────
setTimeout(() => {
  drawTimeline();
  drawCH19Signal();
  drawCH21Signal();
  drawBreakdown();
  renderSummary();
}, 100);

window.addEventListener('resize', () => {
  drawTimeline();
  drawCH19Signal();
  drawCH21Signal();
  drawBreakdown();
});
</script>
</body>
</html>'''


def main():
    data = load_and_compact()
    data_json = json.dumps(data, separators=(",", ":"))

    html = TEMPLATE.replace("%%DASHBOARD_DATA%%", data_json)

    with open(OUTPUT_HTML, "w") as f:
        f.write(html)

    size_kb = len(html) / 1024
    print(f"Generated {OUTPUT_HTML} ({size_kb:.1f} KB)")
    print(f"  CH19: {data['ch19']['summary']['total_cuts']} cuts, {len(data['ch19']['frame_data'])} frame samples")
    print(f"  CH21: {data['ch21']['results']['accepted']} accepted, {data['ch21']['results']['rejected']} rejected, {len(data['ch21']['frames'])} frame samples")


if __name__ == "__main__":
    main()
