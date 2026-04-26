import { useState, useEffect, useRef, useCallback } from "react";
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip,
  ResponsiveContainer, PieChart, Pie, Cell, Legend,
} from "recharts";

// ─── CONFIG ──────────────────────────────────────────────────────────────────
const API_BASE = "http://localhost:8000";

// ─── HELPERS ─────────────────────────────────────────────────────────────────
const fmt = (iso) => {
  if (!iso) return "—";
  return new Date(iso).toLocaleTimeString("en-GB", { hour12: false });
};

const RISK_COLOR = (score, max = 8) => {
  const pct = score / max;
  if (pct >= 0.5) return "#ff4757";
  if (pct >= 0.25) return "#ffa502";
  return "#2ed573";
};

const DECISION_COLOR = (d) => (d === "BLOCK" ? "#ff4757" : "#2ed573");

const PIPE_ICONS = {
  preprocessor:      "🔍",
  rule_engine:       "🛡",
  intent_classifier: "🧠",
  jailbreak_model:   "🔐",
  guard_llm:         "🦙",
  agent_monitor:     "📋",
};

// ─── GLOBAL STYLES ────────────────────────────────────────────────────────────
const css = `
  @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500;700&display=swap');

  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

  :root {
    --bg:       #060a0f;
    --surface:  #0b1018;
    --card:     #0f1822;
    --card2:    #131f2e;
    --border:   #1a2d42;
    --border2:  #203549;
    --cyan:     #00d4ff;
    --cyan-dim: rgba(0,212,255,0.12);
    --green:    #2ed573;
    --red:      #ff4757;
    --amber:    #ffa502;
    --purple:   #9d7bff;
    --muted:    #3d5a73;
    --text:     #b8cfe0;
    --text-hi:  #ddeeff;
    --mono:     'JetBrains Mono', monospace;
    --sans:     'Space Grotesk', sans-serif;
    --radius:   10px;
    --radius-sm: 6px;
  }

  html { scroll-behavior: smooth; }

  body {
    background: var(--bg);
    color: var(--text);
    font-family: var(--sans);
    font-size: 14px;
    line-height: 1.5;
    -webkit-font-smoothing: antialiased;
  }

  /* ── SCROLLBAR ── */
  ::-webkit-scrollbar { width: 4px; height: 4px; }
  ::-webkit-scrollbar-track { background: transparent; }
  ::-webkit-scrollbar-thumb { background: var(--border2); border-radius: 4px; }

  /* ── APP SHELL ── */
  .cw-app {
    min-height: 100vh;
    background: var(--bg);
    background-image:
      radial-gradient(ellipse 70% 40% at 50% -10%, rgba(0,212,255,0.07) 0%, transparent 65%),
      linear-gradient(rgba(0,212,255,0.025) 1px, transparent 1px),
      linear-gradient(90deg, rgba(0,212,255,0.025) 1px, transparent 1px);
    background-size: 100% 100%, 48px 48px, 48px 48px;
  }

  /* ── NAV ── */
  .cw-nav {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 0 24px;
    height: 56px;
    background: rgba(9,14,21,0.92);
    border-bottom: 1px solid var(--border);
    position: sticky;
    top: 0;
    z-index: 200;
    backdrop-filter: blur(16px);
    -webkit-backdrop-filter: blur(16px);
    gap: 12px;
  }

  .cw-logo {
    display: flex;
    align-items: center;
    gap: 10px;
    flex-shrink: 0;
  }

  .cw-logo-icon {
    width: 32px; height: 32px;
    background: linear-gradient(135deg, var(--cyan), #0088bb);
    border-radius: 8px;
    display: flex; align-items: center; justify-content: center;
    font-size: 15px;
    box-shadow: 0 0 16px rgba(0,212,255,0.25);
    flex-shrink: 0;
  }

  .cw-logo-text {
    font-family: var(--mono);
    font-size: 14px;
    font-weight: 700;
    color: var(--text-hi);
    letter-spacing: 1px;
    white-space: nowrap;
  }
  .cw-logo-text span { color: var(--cyan); }

  .cw-nav-tabs {
    display: flex;
    gap: 2px;
    overflow-x: auto;
    scrollbar-width: none;
  }
  .cw-nav-tabs::-webkit-scrollbar { display: none; }

  .cw-tab {
    padding: 6px 14px;
    border-radius: var(--radius-sm);
    cursor: pointer;
    font-family: var(--mono);
    font-size: 11px;
    letter-spacing: 1.5px;
    color: var(--muted);
    background: none;
    border: 1px solid transparent;
    transition: all .15s;
    white-space: nowrap;
    font-weight: 500;
  }
  .cw-tab:hover { color: var(--text); background: rgba(255,255,255,0.04); }
  .cw-tab.active {
    color: var(--cyan);
    background: var(--cyan-dim);
    border-color: rgba(0,212,255,0.25);
  }

  .cw-status-badge {
    font-family: var(--mono);
    font-size: 10px;
    letter-spacing: 1.5px;
    padding: 5px 12px;
    border-radius: 4px;
    background: rgba(46,213,115,0.08);
    color: var(--green);
    border: 1px solid rgba(46,213,115,0.25);
    animation: pulse-badge 2.5s infinite;
    white-space: nowrap;
    flex-shrink: 0;
  }
  @keyframes pulse-badge { 0%,100%{opacity:1} 50%{opacity:.55} }

  /* ── MAIN LAYOUT ── */
  .cw-main {
    padding: 20px 20px 40px;
    max-width: 1440px;
    margin: 0 auto;
  }

  /* ── GRIDS ── */
  .cw-grid-4 {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 14px;
    margin-bottom: 16px;
  }
  .cw-grid-main {
    display: grid;
    grid-template-columns: 1.5fr 0.9fr 1.6fr;
    gap: 14px;
    margin-bottom: 16px;
    align-items: start;
  }
  .cw-grid-2 {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 14px;
    margin-bottom: 16px;
  }
  .cw-col { display: flex; flex-direction: column; gap: 14px; }

  /* ── CARDS ── */
  .cw-card {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 16px;
    position: relative;
    overflow: hidden;
    transition: border-color .2s;
  }
  .cw-card::before {
    content: '';
    position: absolute; top: 0; left: 0; right: 0; height: 1px;
    background: linear-gradient(90deg, transparent 10%, rgba(0,212,255,0.35) 50%, transparent 90%);
  }

  .cw-card-title {
    font-family: var(--mono);
    font-size: 10px;
    letter-spacing: 2px;
    color: var(--muted);
    text-transform: uppercase;
    margin-bottom: 14px;
    display: flex;
    align-items: center;
    gap: 8px;
    font-weight: 500;
  }

  .dot {
    width: 6px; height: 6px;
    border-radius: 50%;
    background: var(--cyan);
    animation: pulse-dot 2s infinite;
    flex-shrink: 0;
  }
  @keyframes pulse-dot { 0%,100%{opacity:1;transform:scale(1)} 50%{opacity:.4;transform:scale(.7)} }

  /* ── STAT CARDS ── */
  .cw-stat-val {
    font-family: var(--mono);
    font-size: clamp(28px, 4vw, 40px);
    font-weight: 700;
    line-height: 1;
    margin-bottom: 5px;
    letter-spacing: -1px;
  }
  .cw-stat-label {
    font-size: 11px;
    color: var(--muted);
    letter-spacing: .5px;
    font-weight: 500;
  }
  .cw-stat-sub {
    font-family: var(--mono);
    font-size: 10px;
    margin-top: 6px;
    color: var(--muted);
  }

  /* ── SYSTEM STATUS ── */
  .cw-threat-level {
    font-family: var(--mono);
    font-size: clamp(18px, 3vw, 24px);
    font-weight: 700;
    letter-spacing: 2px;
    margin-bottom: 14px;
  }
  .cw-status-row {
    display: flex;
    gap: 20px;
    flex-wrap: wrap;
  }
  .cw-status-item-label {
    font-family: var(--mono);
    font-size: 9px;
    color: var(--muted);
    letter-spacing: 1.5px;
    margin-bottom: 2px;
  }
  .cw-status-item-val {
    font-family: var(--mono);
    font-size: 22px;
    font-weight: 700;
  }

  /* ── GAUGE ── */
  .cw-gauge-wrap {
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 10px;
    padding: 8px 0;
  }
  .cw-gauge-val { font-family: var(--mono); font-size: 30px; font-weight: 700; text-anchor: middle; dominant-baseline: middle; }
  .cw-gauge-sub { font-family: var(--mono); font-size: 10px; fill: var(--muted); text-anchor: middle; }
  .cw-threat-chip {
    padding: 5px 18px;
    border-radius: 4px;
    font-family: var(--mono);
    font-size: 11px;
    letter-spacing: 2.5px;
    font-weight: 600;
  }

  /* ── PIPELINE ── */
  .cw-pipeline {
    display: flex;
    align-items: center;
    gap: 0;
    overflow-x: auto;
    padding: 6px 0 10px;
    scrollbar-width: none;
  }
  .cw-pipeline::-webkit-scrollbar { display: none; }

  .cw-pipe-step {
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 6px;
    flex: 1;
    min-width: 64px;
  }
  .cw-pipe-icon {
    width: 42px; height: 42px;
    border-radius: 8px;
    background: var(--surface);
    border: 1px solid var(--border);
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 17px;
    transition: all .2s;
    position: relative;
  }
  .cw-pipe-icon.active {
    border-color: rgba(0,212,255,0.5);
    box-shadow: 0 0 14px rgba(0,212,255,0.15);
    background: rgba(0,212,255,0.05);
  }
  .cw-pipe-label {
    font-family: var(--mono);
    font-size: 8.5px;
    color: var(--muted);
    text-align: center;
    line-height: 1.3;
    letter-spacing: .5px;
  }
  .cw-pipe-arrow {
    color: var(--border2);
    font-size: 14px;
    padding: 0 2px;
    flex-shrink: 0;
    margin-bottom: 20px;
  }
  .cw-pipe-status-dot {
    width: 8px; height: 8px;
    border-radius: 50%;
    position: absolute;
    top: -3px; right: -3px;
    border: 1.5px solid var(--card);
  }

  .cw-pipe-list { margin-top: 14px; display: flex; flex-direction: column; gap: 6px; }
  .cw-pipe-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    font-family: var(--mono);
    font-size: 11px;
    padding: 7px 10px;
    border-radius: var(--radius-sm);
    background: var(--surface);
    border: 1px solid var(--border);
  }
  .cw-pipe-row-name { color: var(--text); display: flex; align-items: center; gap: 7px; }
  .cw-pipe-row-status { font-size: 10px; letter-spacing: 1px; }

  /* ── INSPECTOR ── */
  .cw-inspector-wrap { position: relative; }
  .cw-inspector-input {
    width: 100%;
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: var(--radius-sm);
    padding: 12px 48px 12px 14px;
    resize: vertical;
    color: var(--text-hi);
    font-family: var(--mono);
    font-size: 12px;
    line-height: 1.6;
    outline: none;
    min-height: 88px;
    max-height: 200px;
    transition: border-color .2s, box-shadow .2s;
  }
  .cw-inspector-input:focus {
    border-color: rgba(0,212,255,0.5);
    box-shadow: 0 0 0 3px rgba(0,212,255,0.06);
  }
  .cw-inspector-input::placeholder { color: var(--muted); }

  .cw-send-btn {
    position: absolute;
    right: 10px; bottom: 10px;
    width: 30px; height: 30px;
    border-radius: var(--radius-sm);
    background: var(--cyan);
    border: none;
    cursor: pointer;
    display: flex; align-items: center; justify-content: center;
    font-size: 13px;
    font-weight: 700;
    transition: all .15s;
    color: #000;
  }
  .cw-send-btn:hover { background: #22eeff; transform: scale(1.06); }
  .cw-send-btn:disabled { background: var(--muted); cursor: not-allowed; transform: none; color: #000; }
  .cw-send-hint {
    font-family: var(--mono);
    font-size: 9.5px;
    color: var(--muted);
    margin-top: 6px;
    letter-spacing: .5px;
  }

  /* ── RESULT ── */
  .cw-result {
    margin-top: 12px;
    padding: 14px;
    border-radius: var(--radius-sm);
    border: 1px solid var(--border);
    background: var(--surface);
    font-family: var(--mono);
    font-size: 11px;
    line-height: 1.8;
    animation: fadein .25s ease;
  }
  @keyframes fadein { from{opacity:0;transform:translateY(5px)} to{opacity:1;transform:none} }
  .cw-result.block { border-color: rgba(255,71,87,0.35); background: rgba(255,71,87,0.04); }
  .cw-result.allow { border-color: rgba(46,213,115,0.35); background: rgba(46,213,115,0.04); }

  .cw-result-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 10px;
    padding-bottom: 8px;
    border-bottom: 1px solid var(--border);
  }
  .cw-result-decision { font-size: 13px; font-weight: 700; letter-spacing: 1.5px; }
  .cw-result-risk { font-size: 11px; }

  .cw-detail-row {
    display: flex;
    justify-content: space-between;
    align-items: flex-start;
    padding: 5px 0;
    border-bottom: 1px solid rgba(26,45,66,0.5);
    gap: 12px;
  }
  .cw-detail-row:last-child { border-bottom: none; }
  .cw-detail-key { color: var(--muted); flex-shrink: 0; }
  .cw-detail-val { text-align: right; color: var(--text); word-break: break-word; }

  /* ── EVENTS ── */
  .cw-event {
    padding: 10px 13px;
    border-radius: var(--radius-sm);
    background: var(--surface);
    border: 1px solid var(--border);
    transition: border-color .2s;
  }
  .cw-event-label {
    font-family: var(--mono);
    font-size: 9px;
    letter-spacing: 2px;
    margin-bottom: 4px;
    font-weight: 600;
  }
  .cw-event-prompt {
    font-size: 12px;
    color: var(--text);
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
    font-family: var(--mono);
  }
  .cw-event-time {
    font-family: var(--mono);
    font-size: 9px;
    color: var(--muted);
    margin-top: 4px;
  }

  /* ── LOGS TABLE ── */
  .cw-log-table {
    width: 100%;
    border-collapse: collapse;
    font-family: var(--mono);
    font-size: 11.5px;
    min-width: 600px;
  }
  .cw-log-table th {
    text-align: left;
    padding: 9px 14px;
    color: var(--muted);
    border-bottom: 1px solid var(--border);
    font-weight: 500;
    letter-spacing: 1.5px;
    font-size: 9px;
    text-transform: uppercase;
    white-space: nowrap;
  }
  .cw-log-table td {
    padding: 10px 14px;
    border-bottom: 1px solid rgba(26,45,66,0.4);
    vertical-align: middle;
  }
  .cw-log-table tr:hover td { background: rgba(0,212,255,0.025); }
  .cw-log-table tr:last-child td { border-bottom: none; }

  .cw-badge {
    display: inline-block;
    padding: 3px 9px;
    border-radius: 4px;
    font-size: 10px;
    letter-spacing: 1px;
    font-weight: 600;
    white-space: nowrap;
  }
  .cw-badge.BLOCK { background: rgba(255,71,87,0.12); color: #ff4757; border: 1px solid rgba(255,71,87,0.3); }
  .cw-badge.ALLOW { background: rgba(46,213,115,0.1); color: #2ed573; border: 1px solid rgba(46,213,115,0.25); }

  .cw-type-chip {
    display: inline-block;
    padding: 2px 8px;
    border-radius: 4px;
    font-size: 10px;
    font-weight: 500;
    white-space: nowrap;
  }
  /* safe */
  .cw-type-chip.SAFE,
  .cw-type-chip.Safe       { color: #4a6a80; background: rgba(74,106,128,0.12); border: 1px solid rgba(74,106,128,0.2); }
  /* injection family */
  .cw-type-chip.PROMPT_INJECTION,
  .cw-type-chip.Injection,
  .cw-type-chip.INDIRECT_INJECTION  { color: #ffa502; background: rgba(255,165,2,0.1); border: 1px solid rgba(255,165,2,0.25); }
  /* jailbreak family */
  .cw-type-chip.JAILBREAK,
  .cw-type-chip.Jailbreak,
  .cw-type-chip.ROLEPLAY_JAILBREAK,
  .cw-type-chip.PERSONA_SWAP,
  .cw-type-chip.HYPOTHETICAL_BYPASS,
  .cw-type-chip.TRAINING_MODE_EXPLOIT { color: #ff4757; background: rgba(255,71,87,0.1); border: 1px solid rgba(255,71,87,0.25); }
  /* policy / override family */
  .cw-type-chip.POLICY_BYPASS,
  .cw-type-chip.INSTRUCTION_OVERRIDE,
  .cw-type-chip.SYSTEM_PROMPT_EXTRACTION,
  .cw-type-chip.COORDINATED_ATTACK,
  .cw-type-chip.Suspicious { color: #9d7bff; background: rgba(157,123,255,0.1); border: 1px solid rgba(157,123,255,0.25); }
  /* preprocessor types */
  .cw-type-chip.SUSPICIOUS_PATTERN,
  .cw-type-chip.BASE64_ENCODING,
  .cw-type-chip.FRAGMENTED_INSTRUCTIONS,
  .cw-type-chip.LANGUAGE_SWITCHING,
  .cw-type-chip.MULTI_TURN_ATTACK,
  .cw-type-chip.CONVERSATION_HISTORY_INJECTION,
  .cw-type-chip.WEBPAGE_POISONING { color: #00d4ff; background: rgba(0,212,255,0.08); border: 1px solid rgba(0,212,255,0.2); }

  /* ── LIVE ── */
  .cw-live {
    display: flex;
    align-items: center;
    gap: 6px;
    font-family: var(--mono);
    font-size: 10px;
    color: var(--green);
    letter-spacing: 1px;
    white-space: nowrap;
  }
  .cw-live-dot {
    width: 7px; height: 7px;
    border-radius: 50%;
    background: var(--green);
    animation: pulse-dot 1.4s infinite;
  }

  /* ── CHART TOOLTIP ── */
  .cw-tooltip {
    background: var(--card2);
    border: 1px solid var(--border2);
    border-radius: var(--radius-sm);
    padding: 8px 13px;
    font-family: var(--mono);
    font-size: 11px;
    color: var(--text);
    box-shadow: 0 8px 24px rgba(0,0,0,0.4);
  }
  .cw-tooltip-label { color: var(--muted); font-size: 9.5px; margin-bottom: 5px; letter-spacing: 1px; }

  /* ── EMPTY STATE ── */
  .cw-empty {
    text-align: center;
    padding: 48px 20px;
    color: var(--muted);
    font-family: var(--mono);
    font-size: 12px;
    letter-spacing: .5px;
    line-height: 2;
  }

  /* ── LOG FILTERS ── */
  .cw-filter-bar {
    display: flex;
    align-items: center;
    gap: 6px;
    margin-bottom: 14px;
    flex-wrap: wrap;
  }

  /* ── PAGE HEADERS ── */
  .cw-page-title {
    font-family: var(--mono);
    font-size: clamp(16px, 3vw, 22px);
    color: var(--cyan);
    letter-spacing: 2.5px;
    font-weight: 700;
    margin-bottom: 4px;
  }
  .cw-page-sub {
    font-family: var(--mono);
    font-size: 11px;
    color: var(--muted);
    margin-bottom: 20px;
    letter-spacing: .5px;
  }

  /* ── RISK BAR ── */
  .cw-risk-bar-wrap {
    display: flex;
    align-items: center;
    gap: 8px;
  }
  .cw-risk-bar {
    flex: 1;
    height: 4px;
    background: var(--border);
    border-radius: 2px;
    overflow: hidden;
  }
  .cw-risk-bar-fill { height: 100%; border-radius: 2px; transition: width .5s ease; }

  /* ── RESPONSIVE ── */
  @media (max-width: 1100px) {
    .cw-grid-4 { grid-template-columns: repeat(2, 1fr); }
    .cw-grid-main { grid-template-columns: 1fr; }
    .cw-grid-2 { grid-template-columns: 1fr; }
  }

  @media (max-width: 640px) {
    .cw-nav { padding: 0 14px; height: 52px; }
    .cw-logo-text { font-size: 12px; }
    .cw-status-badge { display: none; }
    .cw-main { padding: 14px 12px 40px; }
    .cw-grid-4 { grid-template-columns: repeat(2, 1fr); gap: 10px; }
    .cw-card { padding: 13px; }
    .cw-tab { padding: 5px 10px; font-size: 10px; }
  }

  @media (max-width: 380px) {
    .cw-grid-4 { grid-template-columns: 1fr 1fr; }
    .cw-stat-val { font-size: 26px; }
  }
`;

// ─── RISK GAUGE ───────────────────────────────────────────────────────────────
function RiskGauge({ value = 0 }) {
  const r = 56, cx = 72, cy = 72;
  const circ = Math.PI * r;
  const progress = (value / 100) * circ;
  const color = value >= 70 ? "#ff4757" : value >= 40 ? "#ffa502" : "#2ed573";
  const label = value >= 70 ? "CRITICAL" : value >= 40 ? "HIGH" : value >= 20 ? "MEDIUM" : "LOW";
  const chipBg = value >= 70 ? "rgba(255,71,87,0.1)" : value >= 40 ? "rgba(255,165,2,0.1)" : "rgba(46,213,115,0.1)";
  return (
    <div className="cw-gauge-wrap">
      <svg width="144" height="92" style={{ overflow: "visible" }}>
        <defs>
          <filter id="glow">
            <feGaussianBlur stdDeviation="3" result="blur" />
            <feMerge><feMergeNode in="blur" /><feMergeNode in="SourceGraphic" /></feMerge>
          </filter>
        </defs>
        <path d={`M ${cx-r} ${cy} A ${r} ${r} 0 0 1 ${cx+r} ${cy}`} fill="none" stroke="#1a2d42" strokeWidth="10" strokeLinecap="round" />
        <path
          d={`M ${cx-r} ${cy} A ${r} ${r} 0 0 1 ${cx+r} ${cy}`}
          fill="none" stroke={color} strokeWidth="10" strokeLinecap="round"
          strokeDasharray={`${progress} ${circ}`}
          style={{ transition: "stroke-dasharray 1.2s ease, stroke .4s", filter: `drop-shadow(0 0 6px ${color}88)` }}
        />
        <text x={cx} y={cy - 10} className="cw-gauge-val" style={{ fill: color }}>{value}</text>
        <text x={cx} y={cy + 10} className="cw-gauge-sub">/ 100</text>
      </svg>
      <div className="cw-threat-chip" style={{ background: chipBg, color, border: `1px solid ${color}55` }}>
        {label}
      </div>
    </div>
  );
}

// ─── CHART TOOLTIP ───────────────────────────────────────────────────────────
function ChartTip({ active, payload, label }) {
  if (!active || !payload?.length) return null;
  return (
    <div className="cw-tooltip">
      <div className="cw-tooltip-label">{label}</div>
      {payload.map((p) => (
        <div key={p.name} style={{ color: p.color, display: "flex", justifyContent: "space-between", gap: 16 }}>
          <span>{p.name}</span><span style={{ fontWeight: 700 }}>{p.value}</span>
        </div>
      ))}
    </div>
  );
}

// ─── PIPELINE BAR ─────────────────────────────────────────────────────────────
function PipelineBar({ components }) {
  return (
    <>
      <div className="cw-pipeline">
        {components.map((c, i) => (
          <>
            <div key={c.id} className="cw-pipe-step">
              <div className={`cw-pipe-icon ${c.status === "active" ? "active" : ""}`}>
                {PIPE_ICONS[c.id] || "⚙"}
                <div className="cw-pipe-status-dot" style={{ background: c.status === "active" ? "var(--green)" : "var(--red)" }} />
              </div>
              <div className="cw-pipe-label">{c.name}</div>
            </div>
            {i < components.length - 1 && <div key={`arr-${i}`} className="cw-pipe-arrow">›</div>}
          </>
        ))}
      </div>
      <div className="cw-pipe-list">
        {components.map((c) => (
          <div key={c.id} className="cw-pipe-row">
            <span className="cw-pipe-row-name">{PIPE_ICONS[c.id]} {c.name}</span>
            <span className="cw-pipe-row-status" style={{ color: c.status === "active" ? "var(--green)" : "var(--red)" }}>
              ● {c.status.toUpperCase()}
            </span>
          </div>
        ))}
      </div>
    </>
  );
}

// ─── PROMPT INSPECTOR ─────────────────────────────────────────────────────────
function PromptInspector({ onNewLog }) {
  const [prompt, setPrompt] = useState("");
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);

  const analyze = async () => {
    if (!prompt.trim() || loading) return;
    setLoading(true);
    setResult(null);
    try {
      const res = await fetch(`${API_BASE}/analyze`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ prompt }),
      });
      if (!res.ok) throw new Error(`Server error: ${res.status}`);
      const data = await res.json();
      setResult(data);
      onNewLog?.(data);
    } catch (e) {
      setResult({ error: e.message || "API unreachable — is your backend running on port 8000?" });
    } finally {
      setLoading(false);
    }
  };

  const handleKey = (e) => {
    if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); analyze(); }
  };

  return (
    <div>
      <div className="cw-inspector-wrap">
        <textarea
          className="cw-inspector-input"
          placeholder={"Paste or type an LLM prompt to analyze...\n↵ Enter to scan  |  Shift+Enter for newline"}
          value={prompt}
          onChange={(e) => setPrompt(e.target.value)}
          onKeyDown={handleKey}
        />
        <button className="cw-send-btn" onClick={analyze} disabled={loading || !prompt.trim()}>
          {loading ? "⏳" : "▶"}
        </button>
      </div>
      <div className="cw-send-hint">↵ Enter to analyze · Shift+Enter for new line</div>

      {result && !result.error && (
        <div className={`cw-result ${result.final_decision?.toLowerCase()}`}>
          <div className="cw-result-header">
            <span className="cw-result-decision" style={{ color: DECISION_COLOR(result.final_decision) }}>
              ● {result.final_decision}
            </span>
            <span className="cw-result-risk" style={{ color: RISK_COLOR(result.risk_score, result.max_risk) }}>
              RISK {result.risk_score}/{result.max_risk ?? 8}
            </span>
          </div>
          <div className="cw-detail-row">
            <span className="cw-detail-key">Attack Type</span>
            <span className="cw-detail-val"><span className={`cw-type-chip ${result.attack_type}`}>{result.attack_type}</span></span>
          </div>
          <div className="cw-detail-row">
            <span className="cw-detail-key">Confidence</span>
            <span className="cw-detail-val">{(result.confidence * 100).toFixed(1)}%</span>
          </div>
          {result.details?.rule_engine && (
            <div className="cw-detail-row">
              <span className="cw-detail-key">Rule Engine</span>
              <span className="cw-detail-val" style={{ color: (result.details.rule_engine.hits?.length > 0 || result.details.rule_engine.triggered) ? "var(--amber)" : "var(--green)" }}>
                {(result.details.rule_engine.hits?.length > 0 || result.details.rule_engine.triggered)
                  ? `⚠ ${result.details.rule_engine.hits?.length ?? 1} hit(s)`
                  : "✓ Clean"}
              </span>
            </div>
          )}
          {result.details?.intent_classifier && (
            <div className="cw-detail-row">
              <span className="cw-detail-key">Intent</span>
              <span className="cw-detail-val">
                {result.details.intent_classifier.intent ?? result.details.intent_classifier.label ?? "—"}
                {" "}({((result.details.intent_classifier.confidence ?? result.details.intent_classifier.score ?? 0) * 100).toFixed(0)}%)
              </span>
            </div>
          )}
          {(result.details?.jailbreak_detector || result.details?.jailbreak_model) && (
            <div className="cw-detail-row">
              <span className="cw-detail-key">Jailbreak Prob.</span>
              <span className="cw-detail-val">
                {((result.details.jailbreak_detector?.probability ?? result.details.jailbreak_model?.probability ?? 0) * 100).toFixed(0)}%
              </span>
            </div>
          )}
          {result.details?.guard_llm && (
            <div className="cw-detail-row">
              <span className="cw-detail-key">Guard LLM</span>
              <span className="cw-detail-val" style={{ color: result.details.guard_llm.result?.is_blocked ? "var(--red)" : "var(--green)" }}>
                {result.details.guard_llm.decision ?? (result.details.guard_llm.result?.is_blocked ? "BLOCK" : "SAFE")}
              </span>
            </div>
          )}
          {result.details?.preprocessor?.detected_attacks?.length > 0 && (
            <div className="cw-detail-row">
              <span className="cw-detail-key">Preprocessor</span>
              <span className="cw-detail-val" style={{ color: "var(--amber)", fontSize: 10 }}>
                ⚠ {result.details.preprocessor.detected_attacks.join(", ")}
              </span>
            </div>
          )}
        </div>
      )}

      {result?.error && (
        <div className="cw-result" style={{ borderColor: "rgba(255,71,87,.3)", color: "#ff4757", marginTop: 12 }}>
          ⚠ {result.error}
        </div>
      )}
    </div>
  );
}

// ─── LOGS PAGE ────────────────────────────────────────────────────────────────
function LogsPage({ logs }) {
  const [filter, setFilter] = useState("ALL");
  const filtered = filter === "ALL" ? logs : logs.filter((l) => l.final_decision === filter);

  return (
    <div>
      <div className="cw-filter-bar">
        {["ALL", "BLOCK", "ALLOW"].map((f) => (
          <button key={f} className={`cw-tab ${filter === f ? "active" : ""}`} onClick={() => setFilter(f)}>
            {f} {f !== "ALL" && <span style={{ opacity: .6 }}>({logs.filter(l => f === "ALL" || l.final_decision === f).length})</span>}
          </button>
        ))}
        <div style={{ marginLeft: "auto" }} className="cw-live"><div className="cw-live-dot" />LIVE</div>
      </div>
      <div className="cw-card" style={{ padding: 0 }}>
        {filtered.length === 0 ? (
          <div className="cw-empty">
            No logs yet.<br />Submit a prompt via the Inspector on the Dashboard.
          </div>
        ) : (
          <div style={{ overflowX: "auto" }}>
            <table className="cw-log-table">
              <thead>
                <tr>
                  <th>Time</th>
                  <th>Status</th>
                  <th>Prompt Snippet</th>
                  <th>Type</th>
                  <th>Confidence</th>
                  <th>Risk</th>
                </tr>
              </thead>
              <tbody>
                {filtered.map((l) => (
                  <tr key={l.id}>
                    <td style={{ color: "var(--muted)", whiteSpace: "nowrap" }}>{fmt(l.timestamp)}</td>
                    <td><span className={`cw-badge ${l.final_decision}`}>{l.final_decision}</span></td>
                    <td style={{ maxWidth: 320, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap", color: "var(--text-hi)" }}>
                      {l.prompt_snippet}
                    </td>
                    <td><span className={`cw-type-chip ${l.attack_type}`}>{l.attack_type}</span></td>
                    <td style={{ color: "var(--cyan)" }}>{(l.confidence * 100).toFixed(1)}%</td>
                    <td>
                      <div className="cw-risk-bar-wrap">
                        <div className="cw-risk-bar">
                          <div className="cw-risk-bar-fill" style={{ width: `${(l.risk_score / (l.max_risk ?? 8)) * 100}%`, background: RISK_COLOR(l.risk_score, l.max_risk ?? 8) }} />
                        </div>
                        <span style={{ color: RISK_COLOR(l.risk_score, l.max_risk ?? 8), fontFamily: "var(--mono)", fontSize: 10, minWidth: 16 }}>{l.risk_score}</span>
                      </div>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </div>
  );
}

// ─── ANALYTICS PAGE ───────────────────────────────────────────────────────────
function AnalyticsPage({ stats }) {
  const PIE_COLORS = ["#ffa502", "#ff4757", "#9d7bff", "#00d4ff", "#2ed573", "#ffd32a", "#ff6b81", "#7bed9f"];
  const dist = stats?.attack_distribution || {};

  // Build pie from all attack categories returned by the API
  const attackEntries = Object.entries(dist).filter(([k]) => k !== "SAFE" && k !== "Safe");
  const safeCount = Math.max(0, (stats?.total_monitored || 0) - attackEntries.reduce((s, [, v]) => s + v, 0));

  const pieData = [
    ...attackEntries.map(([name, value]) => ({ name, value })),
    ...(safeCount > 0 ? [{ name: "SAFE", value: safeCount }] : []),
  ].filter((d) => d.value > 0);

  return (
    <div>
      <div className="cw-page-title">ANALYTICS</div>
      <div className="cw-page-sub">Attack distribution & trend analysis</div>

      <div className="cw-grid-2">
        <div className="cw-card">
          <div className="cw-card-title"><div className="dot" />TRAFFIC & THREATS — 7 DAYS</div>
          <ResponsiveContainer width="100%" height={220}>
            <LineChart data={stats?.trend_7d || []} margin={{ top: 5, right: 10, left: -22, bottom: 5 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#1a2d42" />
              <XAxis dataKey="date" tick={{ fontFamily: "var(--mono)", fontSize: 9, fill: "#3d5a73" }} tickFormatter={(v) => v.slice(5)} />
              <YAxis tick={{ fontFamily: "var(--mono)", fontSize: 9, fill: "#3d5a73" }} allowDecimals={false} />
              <Tooltip content={<ChartTip />} />
              <Line type="monotone" dataKey="safe" stroke="#2ed573" strokeWidth={2} dot={{ r: 3, fill: "#2ed573" }} name="Safe" />
              <Line type="monotone" dataKey="blocked" stroke="#ff4757" strokeWidth={2} dot={{ r: 3, fill: "#ff4757" }} name="Blocked" />
            </LineChart>
          </ResponsiveContainer>
        </div>

        <div className="cw-card">
          <div className="cw-card-title"><div className="dot" />ATTACK DISTRIBUTION</div>
          {pieData.length === 0 ? (
            <div className="cw-empty">No attack data yet</div>
          ) : (
            <ResponsiveContainer width="100%" height={220}>
              <PieChart>
                <Pie data={pieData} cx="50%" cy="50%" innerRadius={58} outerRadius={88} dataKey="value" paddingAngle={3}>
                  {pieData.map((_, i) => <Cell key={i} fill={PIE_COLORS[i % PIE_COLORS.length]} />)}
                </Pie>
                <Tooltip content={<ChartTip />} />
                <Legend iconType="circle" iconSize={8}
                  formatter={(v) => <span style={{ fontFamily: "var(--mono)", fontSize: 11, color: "var(--text)" }}>{v}</span>} />
              </PieChart>
            </ResponsiveContainer>
          )}
        </div>
      </div>

      <div className="cw-card">
        <div className="cw-card-title"><div className="dot" />SUMMARY STATISTICS</div>
        <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(150px, 1fr))", gap: 12 }}>
          {[
            { label: "Total Monitored",      val: stats?.total_monitored || 0,      color: "var(--cyan)" },
            { label: "Blocked Today",         val: stats?.blocked_today || 0,        color: "var(--red)" },
            { label: "Safe Analyzed",         val: stats?.safe_analyzed || 0,        color: "var(--green)" },
            { label: "Injections Blocked",    val: stats?.injections_blocked || 0,   color: "var(--amber)" },
            { label: "Jailbreaks Prevented",  val: stats?.jailbreaks_prevented || 0, color: "var(--purple)" },
          ].map((s) => (
            <div key={s.label} style={{ padding: "14px 16px", background: "var(--surface)", borderRadius: "var(--radius-sm)", border: "1px solid var(--border)" }}>
              <div style={{ fontFamily: "var(--mono)", fontSize: 30, color: s.color, fontWeight: 700, lineHeight: 1, marginBottom: 6 }}>{s.val}</div>
              <div style={{ fontSize: 11, color: "var(--muted)", fontWeight: 500 }}>{s.label}</div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}

// ─── MAIN APP ─────────────────────────────────────────────────────────────────
export default function CyberWatchDashboard() {
  const [tab, setTab] = useState("dashboard");
  const [stats, setStats] = useState(null);
  const [logs, setLogs] = useState([]);
  const [pipeline, setPipeline] = useState([
    { id: "rule_engine",       name: "Rule Engine",     status: "active" },
    { id: "intent_classifier", name: "Intent Model",    status: "active" },
    { id: "jailbreak_model",   name: "Jailbreak Model", status: "active" },
    { id: "guard_llm",         name: "Guard LLM",       status: "active" },
    { id: "agent_monitor",     name: "Agent Monitor",   status: "active" },
  ]);

  const pollRef = useRef(null);

  const fetchAll = useCallback(async () => {
    try {
      const [sRes, lRes, pRes] = await Promise.all([
        fetch(`${API_BASE}/stats`),
        fetch(`${API_BASE}/logs?limit=100`),
        fetch(`${API_BASE}/pipeline/status`),
      ]);
      if (sRes.ok) setStats(await sRes.json());
      if (lRes.ok) { const d = await lRes.json(); setLogs(d.logs || []); }
      if (pRes.ok) { const d = await pRes.json(); setPipeline(d.components || []); }
    } catch { /* backend offline — silently wait */ }
  }, []);

  useEffect(() => {
    fetchAll();
    pollRef.current = setInterval(fetchAll, 5000);
    return () => clearInterval(pollRef.current);
  }, [fetchAll]);

  const handleNewLog = () => setTimeout(fetchAll, 300);

  const statCards = [
    { label: "Total Monitored", val: stats?.total_monitored ?? "—", sub: "all time",  color: "var(--cyan)" },
    { label: "Blocked Today",   val: stats?.blocked_today   ?? "—", sub: "today",     color: "var(--red)" },
    { label: "Safe Analyzed",   val: stats?.safe_analyzed   ?? "—", sub: "today",     color: "var(--green)" },
    { label: "System Risk",     val: stats ? `${stats.system_risk_score}` : "—", sub: "/ 100",
      color: stats?.system_risk_score >= 40 ? "var(--amber)" : "var(--green)" },
  ];

  return (
    <>
      <style>{css}</style>
      <div className="cw-app">

        {/* ── NAV ── */}
        <nav className="cw-nav">
          <div className="cw-logo">
            <div className="cw-logo-icon">🛡</div>
            <div className="cw-logo-text">Cyber<span>Watch</span> AI</div>
          </div>
          <div className="cw-nav-tabs">
            {[["dashboard","DASHBOARD"],["logs","LOGS"],["analytics","ANALYTICS"]].map(([id, label]) => (
              <button key={id} className={`cw-tab ${tab === id ? "active" : ""}`} onClick={() => setTab(id)}>
                {label}
              </button>
            ))}
          </div>
          <div className="cw-status-badge">● SYSTEM ACTIVE</div>
        </nav>

        <main className="cw-main">

          {/* ── DASHBOARD ── */}
          {tab === "dashboard" && (
            <>
              {/* Stat row */}
              <div className="cw-grid-4">
                {statCards.map((s) => (
                  <div key={s.label} className="cw-card">
                    <div className="cw-card-title">{s.label}</div>
                    <div className="cw-stat-val" style={{ color: s.color }}>{s.val}</div>
                    <div className="cw-stat-sub">{s.sub}</div>
                  </div>
                ))}
              </div>

              {/* Main 3-col grid */}
              <div className="cw-grid-main">

                {/* Left column */}
                <div className="cw-col">
                  {/* System Status */}
                  <div className="cw-card">
                    <div className="cw-card-title"><div className="dot" />SYSTEM STATUS</div>
                    <div style={{ fontFamily: "var(--mono)", fontSize: 10, color: "var(--muted)", letterSpacing: 1.5, marginBottom: 6 }}>THREAT LEVEL</div>
                    <div className="cw-threat-level" style={{
                      color: stats?.system_risk_score >= 70 ? "var(--red)" : stats?.system_risk_score >= 40 ? "var(--amber)" : "var(--green)"
                    }}>
                      {stats?.threat_level || "—"}
                    </div>
                    <div className="cw-status-row">
                      <div>
                        <div className="cw-status-item-label">BLOCKED TODAY</div>
                        <div className="cw-status-item-val" style={{ color: "var(--red)" }}>{stats?.blocked_today ?? "—"}</div>
                      </div>
                      <div>
                        <div className="cw-status-item-label">SAFE ANALYZED</div>
                        <div className="cw-status-item-val" style={{ color: "var(--green)" }}>{stats?.safe_analyzed ?? "—"}</div>
                      </div>
                    </div>
                  </div>

                  {/* Events */}
                  {stats?.latest_blocked && (
                    <div className="cw-event" style={{ borderColor: "rgba(255,71,87,0.3)" }}>
                      <div className="cw-event-label" style={{ color: "var(--red)" }}>⚠ LATEST THREAT BLOCKED</div>
                      <div className="cw-event-prompt">{stats.latest_blocked.prompt_snippet}</div>
                      <div className="cw-event-time">{fmt(stats.latest_blocked.timestamp)}</div>
                    </div>
                  )}
                  {stats?.latest_safe && (
                    <div className="cw-event" style={{ borderColor: "rgba(46,213,115,0.2)" }}>
                      <div className="cw-event-label" style={{ color: "var(--green)" }}>✓ LATEST SAFE PROMPT</div>
                      <div className="cw-event-prompt">{stats.latest_safe.prompt_snippet}</div>
                      <div className="cw-event-time">{fmt(stats.latest_safe.timestamp)}</div>
                    </div>
                  )}

                  {/* Prompt Inspector */}
                  <div className="cw-card">
                    <div className="cw-card-title">▶_ PROMPT INSPECTOR</div>
                    <PromptInspector onNewLog={handleNewLog} />
                  </div>
                </div>

                {/* Middle — Gauge */}
                <div className="cw-card" style={{ display: "flex", flexDirection: "column", alignItems: "center", justifyContent: "center", minHeight: 200 }}>
                  <div className="cw-card-title" style={{ alignSelf: "flex-start" }}>SYSTEM RISK LEVEL</div>
                  <RiskGauge value={stats?.system_risk_score || 0} />
                </div>

                {/* Right — Pipeline */}
                <div className="cw-card">
                  <div className="cw-card-title"><div className="dot" />SECURITY PIPELINE</div>
                  <PipelineBar components={pipeline} />
                </div>
              </div>

              {/* Trend chart */}
              <div className="cw-card" style={{ marginBottom: 16 }}>
                <div className="cw-card-title"><div className="dot" />TRAFFIC & THREATS TREND — 7 DAYS</div>
                <ResponsiveContainer width="100%" height={180}>
                  <LineChart data={stats?.trend_7d || []} margin={{ top: 5, right: 10, left: -22, bottom: 5 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#1a2d42" />
                    <XAxis dataKey="date" tick={{ fontFamily: "var(--mono)", fontSize: 9, fill: "#3d5a73" }} tickFormatter={(v) => v.slice(5)} />
                    <YAxis tick={{ fontFamily: "var(--mono)", fontSize: 9, fill: "#3d5a73" }} allowDecimals={false} />
                    <Tooltip content={<ChartTip />} />
                    <Line type="monotone" dataKey="safe" stroke="#2ed573" strokeWidth={2} dot={{ r: 3 }} name="Safe" />
                    <Line type="monotone" dataKey="blocked" stroke="#ff4757" strokeWidth={2} dot={{ r: 3 }} name="Blocked" />
                  </LineChart>
                </ResponsiveContainer>
              </div>

              {/* Live logs preview */}
              <div className="cw-card">
                <div className="cw-card-title" style={{ justifyContent: "space-between" }}>
                  <span style={{ display: "flex", alignItems: "center", gap: 8 }}>
                    <div className="dot" />LIVE INTERCEPT LOGS
                  </span>
                  <div className="cw-live"><div className="cw-live-dot" />LIVE</div>
                </div>
                <LogsPage logs={logs.slice(0, 8)} />
              </div>
            </>
          )}

          {/* ── LOGS ── */}
          {tab === "logs" && (
            <>
              <div className="cw-page-title">INTERCEPT LOGS</div>
              <div className="cw-page-sub">{logs.length} total entries</div>
              <LogsPage logs={logs} />
            </>
          )}

          {/* ── ANALYTICS ── */}
          {tab === "analytics" && <AnalyticsPage stats={stats} />}

        </main>
      </div>
    </>
  );
}