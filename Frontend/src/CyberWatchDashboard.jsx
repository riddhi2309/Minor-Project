import { useState, useEffect, useRef, useCallback } from "react";
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip,
  ResponsiveContainer, PieChart, Pie, Cell, Legend,
} from "recharts";

// ─── CONFIG ──────────────────────────────────────────────────────────────────
const API_BASE = "http://localhost:8000"; // ← change if your backend runs elsewhere

// ─── HELPERS ──────────────────────────────────────────────────────────────────
const fmt = (iso) => {
  if (!iso) return "—";
  const d = new Date(iso);
  return d.toLocaleTimeString("en-GB", { hour12: false });
};

const RISK_COLOR = (score) => {
  if (score >= 4) return "#ef4444";
  if (score >= 2) return "#f59e0b";
  return "#22c55e";
};

const DECISION_COLOR = (d) => (d === "BLOCK" ? "#ef4444" : "#22c55e");

const PIPE_ICONS = {
  rule_engine:       "🛡",
  intent_classifier: "🧠",
  jailbreak_model:   "🔐",
  guard_llm:         "🦙",
  agent_monitor:     "📋",
};

// ─── STYLES ───────────────────────────────────────────────────────────────────
const css = `
  @import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=Rajdhani:wght@400;500;600;700&display=swap');

  :root {
    --bg:      #080c10;
    --surface: #0d1117;
    --card:    #111820;
    --border:  #1e2d3d;
    --cyan:    #00e5ff;
    --green:   #22c55e;
    --red:     #ef4444;
    --amber:   #f59e0b;
    --muted:   #4a6070;
    --text:    #c9d8e8;
    --mono:    'Share Tech Mono', monospace;
    --sans:    'Rajdhani', sans-serif;
  }
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { background: var(--bg); color: var(--text); font-family: var(--sans); }

  .cw-app {
    min-height: 100vh;
    background: var(--bg);
    background-image:
      radial-gradient(ellipse 80% 50% at 50% -20%, rgba(0,229,255,0.06) 0%, transparent 60%),
      linear-gradient(rgba(0,229,255,0.03) 1px, transparent 1px),
      linear-gradient(90deg, rgba(0,229,255,0.03) 1px, transparent 1px);
    background-size: 100% 100%, 40px 40px, 40px 40px;
  }

  /* ── NAV ── */
  .cw-nav {
    display: flex; align-items: center; justify-content: space-between;
    padding: 0 28px; height: 52px;
    background: rgba(13,17,23,0.95);
    border-bottom: 1px solid var(--border);
    position: sticky; top: 0; z-index: 100;
    backdrop-filter: blur(12px);
  }
  .cw-logo { display: flex; align-items: center; gap: 10px; }
  .cw-logo-icon {
    width: 30px; height: 30px;
    background: linear-gradient(135deg, var(--cyan), #0099bb);
    border-radius: 6px;
    display: flex; align-items: center; justify-content: center;
    font-size: 14px;
  }
  .cw-logo-text { font-family: var(--mono); font-size: 15px; color: #fff; letter-spacing: 1px; }
  .cw-logo-text span { color: var(--cyan); }
  .cw-nav-tabs { display: flex; gap: 2px; }
  .cw-tab {
    padding: 6px 16px; border-radius: 4px; cursor: pointer;
    font-family: var(--mono); font-size: 11px; letter-spacing: 1px;
    color: var(--muted); background: none; border: none;
    transition: all .15s;
  }
  .cw-tab:hover { color: var(--text); background: rgba(255,255,255,0.05); }
  .cw-tab.active { color: var(--cyan); background: rgba(0,229,255,0.08); border: 1px solid rgba(0,229,255,0.2); }
  .cw-status-badge {
    font-family: var(--mono); font-size: 10px; letter-spacing: 2px;
    padding: 4px 12px; border-radius: 3px;
    background: rgba(34,197,94,0.1); color: var(--green);
    border: 1px solid rgba(34,197,94,0.3);
    animation: pulse-badge 2s infinite;
  }
  @keyframes pulse-badge { 0%,100%{opacity:1} 50%{opacity:.6} }

  /* ── LAYOUT ── */
  .cw-main { padding: 24px 28px; max-width: 1400px; margin: 0 auto; }
  .cw-grid-4 { display: grid; grid-template-columns: repeat(4,1fr); gap: 16px; margin-bottom: 20px; }
  .cw-grid-3 { display: grid; grid-template-columns: 1.4fr 1fr 1.6fr; gap: 16px; margin-bottom: 20px; }
  .cw-grid-2 { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; margin-bottom: 20px; }

  /* ── CARDS ── */
  .cw-card {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 18px;
    position: relative;
    overflow: hidden;
  }
  .cw-card::before {
    content:''; position:absolute; top:0; left:0; right:0; height:1px;
    background: linear-gradient(90deg, transparent, var(--cyan), transparent);
    opacity: .3;
  }
  .cw-card-title {
    font-family: var(--mono); font-size: 10px; letter-spacing: 2px;
    color: var(--muted); text-transform: uppercase; margin-bottom: 14px;
    display: flex; align-items: center; gap: 8px;
  }
  .cw-card-title .dot {
    width: 6px; height: 6px; border-radius: 50%; background: var(--cyan);
    animation: pulse-dot 2s infinite;
  }
  @keyframes pulse-dot { 0%,100%{opacity:1;transform:scale(1)} 50%{opacity:.5;transform:scale(.8)} }

  /* ── STAT CARDS ── */
  .cw-stat-val {
    font-family: var(--mono); font-size: 36px; font-weight: 700; line-height: 1;
    margin-bottom: 4px;
  }
  .cw-stat-label { font-size: 12px; color: var(--muted); letter-spacing: .5px; }
  .cw-stat-sub { font-family: var(--mono); font-size: 10px; margin-top: 6px; }

  /* ── RISK GAUGE ── */
  .cw-gauge-wrap { display: flex; flex-direction: column; align-items: center; }
  .cw-gauge-svg { overflow: visible; }
  .cw-gauge-val {
    font-family: var(--mono); font-size: 28px; font-weight: 700;
    fill: var(--cyan); text-anchor: middle; dominant-baseline: middle;
  }
  .cw-gauge-label { font-family: var(--mono); font-size: 9px; fill: var(--muted); text-anchor: middle; }
  .cw-threat-chip {
    margin-top: 10px; padding: 4px 16px; border-radius: 3px;
    font-family: var(--mono); font-size: 11px; letter-spacing: 2px;
  }

  /* ── PIPELINE ── */
  .cw-pipeline {
    display: flex; align-items: center; gap: 0;
    overflow-x: auto; padding: 8px 0;
  }
  .cw-pipe-step {
    display: flex; flex-direction: column; align-items: center; gap: 6px;
    flex: 1; min-width: 70px;
  }
  .cw-pipe-icon {
    width: 44px; height: 44px; border-radius: 8px;
    background: var(--surface); border: 1px solid var(--border);
    display: flex; align-items: center; justify-content: center;
    font-size: 18px; transition: all .2s; position: relative;
  }
  .cw-pipe-icon.active { border-color: var(--cyan); box-shadow: 0 0 12px rgba(0,229,255,0.2); }
  .cw-pipe-label { font-family: var(--mono); font-size: 9px; color: var(--muted); text-align: center; }
  .cw-pipe-arrow {
    color: var(--border); font-size: 16px; padding: 0 2px;
    flex-shrink: 0;
  }
  .cw-pipe-status {
    width: 8px; height: 8px; border-radius: 50%;
    position: absolute; top: -3px; right: -3px;
  }

  /* ── PROMPT INSPECTOR ── */
  .cw-inspector-input {
    width: 100%; background: var(--surface); border: 1px solid var(--border);
    border-radius: 6px; padding: 12px 44px 12px 12px; resize: none;
    color: var(--text); font-family: var(--mono); font-size: 12px;
    line-height: 1.5; outline: none; min-height: 80px;
    transition: border-color .2s;
  }
  .cw-inspector-input:focus { border-color: var(--cyan); }
  .cw-inspector-input::placeholder { color: var(--muted); }
  .cw-send-btn {
    position: absolute; right: 10px; bottom: 10px;
    width: 30px; height: 30px; border-radius: 6px;
    background: var(--cyan); border: none; cursor: pointer;
    display: flex; align-items: center; justify-content: center;
    font-size: 14px; transition: all .15s;
  }
  .cw-send-btn:hover { background: #00c4dd; transform: scale(1.05); }
  .cw-send-btn:disabled { background: var(--muted); cursor: not-allowed; transform: none; }

  /* ── RESULT BADGE ── */
  .cw-result {
    margin-top: 12px; padding: 12px; border-radius: 6px;
    border: 1px solid var(--border); background: var(--surface);
    font-family: var(--mono); font-size: 11px; line-height: 1.8;
    animation: fade-in .3s ease;
  }
  @keyframes fade-in { from{opacity:0;transform:translateY(4px)} to{opacity:1;transform:none} }
  .cw-result.block { border-color: rgba(239,68,68,0.4); background: rgba(239,68,68,0.05); }
  .cw-result.allow { border-color: rgba(34,197,94,0.4); background: rgba(34,197,94,0.05); }

  /* ── LOGS TABLE ── */
  .cw-log-table { width: 100%; border-collapse: collapse; font-family: var(--mono); font-size: 11px; }
  .cw-log-table th {
    text-align: left; padding: 8px 12px; color: var(--muted);
    border-bottom: 1px solid var(--border); font-weight: 400; letter-spacing: 1px;
    font-size: 9px; text-transform: uppercase;
  }
  .cw-log-table td { padding: 9px 12px; border-bottom: 1px solid rgba(30,45,61,0.5); vertical-align: middle; }
  .cw-log-table tr:hover td { background: rgba(0,229,255,0.03); }
  .cw-badge {
    padding: 2px 8px; border-radius: 3px;
    font-family: var(--mono); font-size: 10px; letter-spacing: 1px;
  }
  .cw-badge.BLOCK { background: rgba(239,68,68,0.15); color: #ef4444; border: 1px solid rgba(239,68,68,0.3); }
  .cw-badge.ALLOW { background: rgba(34,197,94,0.1); color: #22c55e; border: 1px solid rgba(34,197,94,0.2); }
  .cw-type-chip {
    padding: 1px 6px; border-radius: 3px; font-family: var(--mono); font-size: 10px;
  }
  .cw-type-chip.Safe       { color: #4a6070; background: rgba(74,96,112,0.1); }
  .cw-type-chip.Injection  { color: #f59e0b; background: rgba(245,158,11,0.1); }
  .cw-type-chip.Jailbreak  { color: #ef4444; background: rgba(239,68,68,0.1); }
  .cw-type-chip.Suspicious { color: #a78bfa; background: rgba(167,139,250,0.1); }

  /* ── LIVE DOT ── */
  .cw-live { display: flex; align-items: center; gap: 6px; font-family: var(--mono); font-size: 10px; color: var(--green); }
  .cw-live-dot { width: 7px; height: 7px; border-radius: 50%; background: var(--green); animation: pulse-dot 1.5s infinite; }

  /* ── EVENTS ── */
  .cw-event {
    display: flex; flex-direction: column; gap: 3px;
    padding: 10px 12px; border-radius: 6px; background: var(--surface);
    border: 1px solid var(--border); margin-bottom: 8px;
  }
  .cw-event-label { font-family: var(--mono); font-size: 9px; letter-spacing: 2px; }
  .cw-event-prompt { font-size: 12px; color: var(--text); margin-top: 2px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
  .cw-event-time { font-family: var(--mono); font-size: 9px; color: var(--muted); margin-top: 2px; }

  /* ── CHART TOOLTIP ── */
  .cw-tooltip {
    background: var(--card); border: 1px solid var(--border);
    border-radius: 6px; padding: 8px 12px;
    font-family: var(--mono); font-size: 11px; color: var(--text);
  }

  /* ── DETAIL PANEL ── */
  .cw-detail-row { display: flex; justify-content: space-between; padding: 4px 0; font-family: var(--mono); font-size: 11px; border-bottom: 1px solid rgba(30,45,61,0.4); }
  .cw-detail-key { color: var(--muted); }

  /* ── EMPTY STATE ── */
  .cw-empty { text-align: center; padding: 40px; color: var(--muted); font-family: var(--mono); font-size: 12px; }

  /* ── SCROLLBAR ── */
  ::-webkit-scrollbar { width: 5px; height: 5px; }
  ::-webkit-scrollbar-track { background: var(--bg); }
  ::-webkit-scrollbar-thumb { background: var(--border); border-radius: 3px; }

  /* ── RESPONSIVE ── */
  @media (max-width: 1100px) {
    .cw-grid-4 { grid-template-columns: repeat(2,1fr); }
    .cw-grid-3 { grid-template-columns: 1fr; }
    .cw-grid-2 { grid-template-columns: 1fr; }
  }
`;

// ─── GAUGE COMPONENT ──────────────────────────────────────────────────────────
function RiskGauge({ value = 0 }) {
  const r = 54;
  const cx = 70, cy = 70;
  const circumference = Math.PI * r; // half circle arc
  const progress = (value / 100) * circumference;
  const color = value >= 70 ? "#ef4444" : value >= 40 ? "#f59e0b" : "#22c55e";
  const threatLabel = value >= 70 ? "CRITICAL" : value >= 40 ? "HIGH" : value >= 20 ? "MEDIUM" : "LOW";
  const threatBg = value >= 70 ? "rgba(239,68,68,0.1)" : value >= 40 ? "rgba(245,158,11,0.1)" : value >= 20 ? "rgba(245,158,11,0.08)" : "rgba(34,197,94,0.1)";

  return (
    <div className="cw-gauge-wrap">
      <svg width="140" height="90" className="cw-gauge-svg">
        <path
          d={`M ${cx - r} ${cy} A ${r} ${r} 0 0 1 ${cx + r} ${cy}`}
          fill="none" stroke="#1e2d3d" strokeWidth="10" strokeLinecap="round"
        />
        <path
          d={`M ${cx - r} ${cy} A ${r} ${r} 0 0 1 ${cx + r} ${cy}`}
          fill="none" stroke={color} strokeWidth="10" strokeLinecap="round"
          strokeDasharray={`${progress} ${circumference}`}
          style={{ transition: "stroke-dasharray 1s ease, stroke .5s" }}
        />
        <text x={cx} y={cy - 8} className="cw-gauge-val" style={{ fill: color }}>{value}</text>
        <text x={cx} y={cy + 10} className="cw-gauge-label">/ 100</text>
      </svg>
      <div className="cw-threat-chip" style={{ background: threatBg, color, border: `1px solid ${color}40`, fontFamily: "var(--mono)", fontSize: 11, letterSpacing: 2 }}>
        {threatLabel}
      </div>
    </div>
  );
}

// ─── CUSTOM TOOLTIP ──────────────────────────────────────────────────────────
function ChartTip({ active, payload, label }) {
  if (!active || !payload?.length) return null;
  return (
    <div className="cw-tooltip">
      <div style={{ marginBottom: 4, color: "var(--muted)", fontSize: 10 }}>{label}</div>
      {payload.map((p) => (
        <div key={p.name} style={{ color: p.color }}>{p.name}: {p.value}</div>
      ))}
    </div>
  );
}

// ─── PIPELINE STATUS BAR ──────────────────────────────────────────────────────
function PipelineBar({ components }) {
  return (
    <div className="cw-pipeline">
      {components.map((c, i) => (
        <>
          <div key={c.id} className="cw-pipe-step">
            <div className={`cw-pipe-icon ${c.status === "active" ? "active" : ""}`}>
              {PIPE_ICONS[c.id] || "⚙"}
              <div className="cw-pipe-status" style={{ background: c.status === "active" ? "var(--green)" : "var(--red)" }} />
            </div>
            <div className="cw-pipe-label">{c.name}</div>
          </div>
          {i < components.length - 1 && <div key={`arr-${i}`} className="cw-pipe-arrow">▶</div>}
        </>
      ))}
    </div>
  );
}

// ─── PROMPT INSPECTOR ─────────────────────────────────────────────────────────
function PromptInspector({ onNewLog }) {
  const [prompt, setPrompt] = useState("");
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);

  const analyze = async () => {
    if (!prompt.trim() || loading) return;
    setLoading(true); setResult(null);
    try {
      const res = await fetch(`${API_BASE}/analyze`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ prompt }),
      });
      const data = await res.json();
      setResult(data);
      onNewLog?.(data);
    } catch (e) {
      setResult({ error: "API unreachable — is your backend running?" });
    } finally {
      setLoading(false);
    }
  };

  const handleKey = (e) => { if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); analyze(); } };

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 0 }}>
      <div style={{ position: "relative" }}>
        <textarea
          className="cw-inspector-input"
          placeholder={"Enter LLM prompt to simulate injection...\n(Shift+Enter for newline, Enter to analyze)"}
          value={prompt}
          onChange={(e) => setPrompt(e.target.value)}
          onKeyDown={handleKey}
        />
        <button className="cw-send-btn" onClick={analyze} disabled={loading || !prompt.trim()}>
          {loading ? "⏳" : "➤"}
        </button>
      </div>

      {result && !result.error && (
        <div className={`cw-result ${result.final_decision?.toLowerCase()}`}>
          <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 8 }}>
            <span style={{ color: DECISION_COLOR(result.final_decision), fontWeight: 700, letterSpacing: 1 }}>
              ● {result.final_decision}
            </span>
            <span style={{ color: RISK_COLOR(result.risk_score) }}>RISK {result.risk_score}/5</span>
          </div>
          <div className="cw-detail-row"><span className="cw-detail-key">Attack Type</span><span>{result.attack_type}</span></div>
          <div className="cw-detail-row"><span className="cw-detail-key">Confidence</span><span>{(result.confidence * 100).toFixed(1)}%</span></div>
          {result.details?.rule_engine && (
            <div className="cw-detail-row">
              <span className="cw-detail-key">Rule Engine</span>
              <span>{result.details.rule_engine.triggered ? `⚠ ${result.details.rule_engine.pattern}` : "✓ Clean"}</span>
            </div>
          )}
          {result.details?.intent_classifier && (
            <div className="cw-detail-row">
              <span className="cw-detail-key">Intent</span>
              <span>{result.details.intent_classifier.label} ({(result.details.intent_classifier.score * 100).toFixed(0)}%)</span>
            </div>
          )}
          {result.details?.jailbreak_model && (
            <div className="cw-detail-row">
              <span className="cw-detail-key">Jailbreak P</span>
              <span>{(result.details.jailbreak_model.probability * 100).toFixed(0)}%</span>
            </div>
          )}
          {result.details?.guard_llm && (
            <div className="cw-detail-row">
              <span className="cw-detail-key">Guard LLM</span>
              <span style={{ maxWidth: 180, textAlign: "right", fontSize: 10, color: "var(--muted)" }}>
                {result.details.guard_llm.reason}
              </span>
            </div>
          )}
        </div>
      )}

      {result?.error && (
        <div className="cw-result" style={{ borderColor: "rgba(239,68,68,.3)", color: "#ef4444" }}>
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
      <div style={{ display: "flex", gap: 8, marginBottom: 16 }}>
        {["ALL", "BLOCK", "ALLOW"].map((f) => (
          <button key={f} className={`cw-tab ${filter === f ? "active" : ""}`} onClick={() => setFilter(f)}>{f}</button>
        ))}
        <div style={{ marginLeft: "auto" }} className="cw-live"><div className="cw-live-dot" /> LIVE</div>
      </div>
      <div className="cw-card" style={{ padding: 0 }}>
        {filtered.length === 0 ? (
          <div className="cw-empty">No logs yet. Submit a prompt via the Inspector.</div>
        ) : (
          <div style={{ overflowX: "auto" }}>
            <table className="cw-log-table">
              <thead>
                <tr>
                  <th>Time</th><th>Status</th><th>Prompt Snippet</th>
                  <th>Type</th><th>Confidence</th><th>Risk</th>
                </tr>
              </thead>
              <tbody>
                {filtered.map((l) => (
                  <tr key={l.id}>
                    <td style={{ color: "var(--muted)", whiteSpace: "nowrap" }}>{fmt(l.timestamp)}</td>
                    <td><span className={`cw-badge ${l.final_decision}`}>{l.final_decision}</span></td>
                    <td style={{ maxWidth: 340, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
                      {l.prompt_snippet}
                    </td>
                    <td><span className={`cw-type-chip ${l.attack_type}`}>{l.attack_type}</span></td>
                    <td style={{ color: "var(--cyan)" }}>{(l.confidence * 100).toFixed(1)}%</td>
                    <td>
                      <span style={{ color: RISK_COLOR(l.risk_score), fontFamily: "var(--mono)" }}>
                        {"█".repeat(l.risk_score)}{"░".repeat(5 - l.risk_score)}
                      </span>
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
  const PIE_COLORS = ["#f59e0b", "#ef4444", "#a78bfa", "#22c55e"];
  const dist = stats?.attack_distribution || {};
  const pieData = [
    { name: "Injection",  value: dist.injection  || 0 },
    { name: "Jailbreak",  value: dist.jailbreak  || 0 },
    { name: "Suspicious", value: dist.suspicious || 0 },
    { name: "Safe",       value: (stats?.total_monitored || 0) - (dist.injection || 0) - (dist.jailbreak || 0) - (dist.suspicious || 0) },
  ].filter((d) => d.value > 0);

  return (
    <div>
      <div className="cw-grid-2">
        <div className="cw-card">
          <div className="cw-card-title"><div className="dot" />TRAFFIC & THREATS TREND (7 DAYS)</div>
          <ResponsiveContainer width="100%" height={220}>
            <LineChart data={stats?.trend_7d || []} margin={{ top: 5, right: 10, left: -20, bottom: 5 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#1e2d3d" />
              <XAxis dataKey="date" tick={{ fontFamily: "var(--mono)", fontSize: 9, fill: "#4a6070" }} tickFormatter={(v) => v.slice(5)} />
              <YAxis tick={{ fontFamily: "var(--mono)", fontSize: 9, fill: "#4a6070" }} allowDecimals={false} />
              <Tooltip content={<ChartTip />} />
              <Line type="monotone" dataKey="safe" stroke="#22c55e" strokeWidth={2} dot={{ r: 3 }} name="Safe" />
              <Line type="monotone" dataKey="blocked" stroke="#ef4444" strokeWidth={2} dot={{ r: 3 }} name="Blocked" />
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
                <Pie data={pieData} cx="50%" cy="50%" innerRadius={55} outerRadius={85}
                  dataKey="value" paddingAngle={3}>
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
        <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(160px, 1fr))", gap: 16 }}>
          {[
            { label: "Total Monitored",     val: stats?.total_monitored || 0,    color: "var(--cyan)" },
            { label: "Blocked Today",        val: stats?.blocked_today || 0,      color: "#ef4444" },
            { label: "Safe Analyzed",        val: stats?.safe_analyzed || 0,      color: "var(--green)" },
            { label: "Injections Blocked",   val: stats?.injections_blocked || 0, color: "#f59e0b" },
            { label: "Jailbreaks Prevented", val: stats?.jailbreaks_prevented || 0, color: "#a78bfa" },
          ].map((s) => (
            <div key={s.label} style={{ padding: 12, background: "var(--surface)", borderRadius: 6, border: "1px solid var(--border)" }}>
              <div style={{ fontFamily: "var(--mono)", fontSize: 28, color: s.color, fontWeight: 700 }}>{s.val}</div>
              <div style={{ fontSize: 11, color: "var(--muted)", marginTop: 4 }}>{s.label}</div>
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
    { id: "rule_engine", name: "Rule Engine", status: "active" },
    { id: "intent_classifier", name: "Intent Model", status: "active" },
    { id: "jailbreak_model", name: "Jailbreak Model", status: "active" },
    { id: "guard_llm", name: "Guard LLM", status: "active" },
    { id: "agent_monitor", name: "Agent Monitor", status: "active" },
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
    } catch { /* backend not running yet — silently ignore */ }
  }, []);

  useEffect(() => {
    fetchAll();
    pollRef.current = setInterval(fetchAll, 5000);
    return () => clearInterval(pollRef.current);
  }, [fetchAll]);

  const handleNewLog = () => { setTimeout(fetchAll, 300); };

  // stat cards
  const statCards = [
    { label: "Total Monitored", val: stats?.total_monitored ?? "—", sub: "all time", color: "var(--cyan)" },
    { label: "Blocked Today",   val: stats?.blocked_today ?? "—",   sub: "today",    color: "#ef4444" },
    { label: "Safe Analyzed",   val: stats?.safe_analyzed ?? "—",   sub: "today",    color: "var(--green)" },
    { label: "System Risk",     val: stats ? `${stats.system_risk_score}` : "—", sub: "/ 100", color: stats?.system_risk_score >= 40 ? "#f59e0b" : "var(--green)" },
  ];

  return (
    <>
      <style>{css}</style>
      <div className="cw-app">
        {/* NAV */}
        <nav className="cw-nav">
          <div className="cw-logo">
            <div className="cw-logo-icon">🛡</div>
            <div className="cw-logo-text">Cyber<span>Watch</span> AI</div>
          </div>
          <div className="cw-nav-tabs">
            {[["dashboard", "DASHBOARD"], ["logs", "LOGS"], ["analytics", "ANALYTICS"]].map(([id, label]) => (
              <button key={id} className={`cw-tab ${tab === id ? "active" : ""}`} onClick={() => setTab(id)}>{label}</button>
            ))}
          </div>
          <div className="cw-status-badge">● SYSTEM ACTIVE</div>
        </nav>

        <main className="cw-main">

          {/* ── DASHBOARD TAB ── */}
          {tab === "dashboard" && (
            <>
              {/* Stat row */}
              <div className="cw-grid-4">
                {statCards.map((s) => (
                  <div key={s.label} className="cw-card">
                    <div className="cw-card-title">{s.label}</div>
                    <div className="cw-stat-val" style={{ color: s.color }}>{s.val}</div>
                    <div className="cw-stat-sub" style={{ color: "var(--muted)" }}>{s.sub}</div>
                  </div>
                ))}
              </div>

              {/* Middle row */}
              <div className="cw-grid-3">
                {/* Left: status + events */}
                <div style={{ display: "flex", flexDirection: "column", gap: 16 }}>
                  <div className="cw-card">
                    <div className="cw-card-title"><div className="dot" />SYSTEM STATUS</div>
                    <div style={{ fontFamily: "var(--mono)", fontSize: 11, color: "var(--muted)", marginBottom: 8 }}>THREAT LEVEL</div>
                    <div style={{ fontFamily: "var(--mono)", fontSize: 20, fontWeight: 700, color: stats?.system_risk_score >= 70 ? "#ef4444" : stats?.system_risk_score >= 40 ? "#f59e0b" : "#22c55e", marginBottom: 12 }}>
                      {stats?.threat_level || "—"}
                    </div>
                    <div style={{ display: "flex", gap: 16 }}>
                      <div><div style={{ fontFamily: "var(--mono)", fontSize: 10, color: "var(--muted)" }}>Blocked Today</div><div style={{ fontFamily: "var(--mono)", fontSize: 22, color: "#ef4444" }}>{stats?.blocked_today ?? "—"}</div></div>
                      <div><div style={{ fontFamily: "var(--mono)", fontSize: 10, color: "var(--muted)" }}>Safe Analyzed</div><div style={{ fontFamily: "var(--mono)", fontSize: 22, color: "var(--green)" }}>{stats?.safe_analyzed ?? "—"}</div></div>
                    </div>
                  </div>

                  {stats?.latest_blocked && (
                    <div className="cw-event" style={{ borderColor: "rgba(239,68,68,0.3)" }}>
                      <div className="cw-event-label" style={{ color: "#ef4444" }}>⚠ LATEST THREAT BLOCKED</div>
                      <div className="cw-event-prompt">{stats.latest_blocked.prompt_snippet}</div>
                      <div className="cw-event-time">{fmt(stats.latest_blocked.timestamp)}</div>
                    </div>
                  )}
                  {stats?.latest_safe && (
                    <div className="cw-event" style={{ borderColor: "rgba(34,197,94,0.2)" }}>
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

                {/* Middle: gauge */}
                <div className="cw-card" style={{ display: "flex", flexDirection: "column", alignItems: "center", justifyContent: "center" }}>
                  <div className="cw-card-title" style={{ alignSelf: "flex-start" }}>SYSTEM RISK LEVEL</div>
                  <RiskGauge value={stats?.system_risk_score || 0} />
                </div>

                {/* Right: pipeline */}
                <div className="cw-card">
                  <div className="cw-card-title"><div className="dot" />SECURITY PIPELINE</div>
                  <PipelineBar components={pipeline} />
                  <div style={{ marginTop: 16, display: "flex", flexDirection: "column", gap: 8 }}>
                    {pipeline.map((c) => (
                      <div key={c.id} style={{ display: "flex", justifyContent: "space-between", fontFamily: "var(--mono)", fontSize: 11, padding: "4px 0", borderBottom: "1px solid var(--border)" }}>
                        <span>{PIPE_ICONS[c.id]} {c.name}</span>
                        <span style={{ color: c.status === "active" ? "var(--green)" : "#ef4444" }}>● {c.status}</span>
                      </div>
                    ))}
                  </div>
                </div>
              </div>

              {/* Trend chart */}
              <div className="cw-card">
                <div className="cw-card-title"><div className="dot" />TRAFFIC & THREATS TREND (7 DAYS)</div>
                <ResponsiveContainer width="100%" height={180}>
                  <LineChart data={stats?.trend_7d || []} margin={{ top: 5, right: 10, left: -20, bottom: 5 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#1e2d3d" />
                    <XAxis dataKey="date" tick={{ fontFamily: "var(--mono)", fontSize: 9, fill: "#4a6070" }} tickFormatter={(v) => v.slice(5)} />
                    <YAxis tick={{ fontFamily: "var(--mono)", fontSize: 9, fill: "#4a6070" }} allowDecimals={false} />
                    <Tooltip content={<ChartTip />} />
                    <Line type="monotone" dataKey="safe" stroke="#22c55e" strokeWidth={2} dot={{ r: 3 }} name="Safe" />
                    <Line type="monotone" dataKey="blocked" stroke="#ef4444" strokeWidth={2} dot={{ r: 3 }} name="Blocked" />
                  </LineChart>
                </ResponsiveContainer>
              </div>

              {/* Recent logs preview */}
              <div className="cw-card" style={{ marginTop: 0 }}>
                <div className="cw-card-title" style={{ justifyContent: "space-between" }}>
                  <span><span className="dot" style={{ display: "inline-block", width: 6, height: 6, borderRadius: "50%", background: "var(--cyan)", marginRight: 8 }} />LIVE INTERCEPT LOGS</span>
                  <div className="cw-live"><div className="cw-live-dot" /> LIVE</div>
                </div>
                <LogsPage logs={logs.slice(0, 8)} />
              </div>
            </>
          )}

          {/* ── LOGS TAB ── */}
          {tab === "logs" && (
            <div>
              <div style={{ marginBottom: 20 }}>
                <div style={{ fontFamily: "var(--mono)", fontSize: 18, color: "var(--cyan)", letterSpacing: 2, marginBottom: 4 }}>INTERCEPT LOGS</div>
                <div style={{ fontFamily: "var(--mono)", fontSize: 11, color: "var(--muted)" }}>{logs.length} total entries</div>
              </div>
              <LogsPage logs={logs} />
            </div>
          )}

          {/* ── ANALYTICS TAB ── */}
          {tab === "analytics" && (
            <div>
              <div style={{ marginBottom: 20 }}>
                <div style={{ fontFamily: "var(--mono)", fontSize: 18, color: "var(--cyan)", letterSpacing: 2, marginBottom: 4 }}>ANALYTICS</div>
                <div style={{ fontFamily: "var(--mono)", fontSize: 11, color: "var(--muted)" }}>Attack distribution & trend analysis</div>
              </div>
              <AnalyticsPage stats={stats} />
            </div>
          )}

        </main>
      </div>
    </>
  );
}
