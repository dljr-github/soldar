#!/usr/bin/env python3
"""SOLDAR — tactical Solana meme-coin radar terminal."""

from __future__ import annotations

import json
import sqlite3
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import plotly.graph_objects as go
import streamlit as st
from streamlit_autorefresh import st_autorefresh

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

# ─── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="SOLDAR",
    layout="wide",
    page_icon="\U0001f4e1",
    initial_sidebar_state="collapsed",
)

# ─── Auto-refresh every 90 seconds ───────────────────────────────────────────
st_autorefresh(interval=90_000, key="data_refresh")

# ─── Fonts + CSS ──────────────────────────────────────────────────────────────
import streamlit.components.v1 as _stc
_stc.html("""
<link href="https://fonts.googleapis.com/css2?family=Rajdhani:wght@400;600;700&family=Share+Tech+Mono&display=swap" rel="stylesheet">
<script>
(function() {
  var style = window.parent.document.getElementById('soldar-css');
  if (!style) {
    style = window.parent.document.createElement('style');
    style.id = 'soldar-css';
    window.parent.document.head.appendChild(style);
  }
  style.textContent = `/* ═══ Custom Properties ═══ */
:root {
  --void: #040608;
  --base: #080d14;
  --surface: #0c1220;
  --panel: #10182a;
  --raised: #152038;
  --border: #1c2d44;
  --border-hi: #2a4060;
  --amber: #ffa726;
  --amber-dim: rgba(255,167,38,0.08);
  --amber-glow: rgba(255,167,38,0.25);
  --green: #00e676;
  --green-dim: rgba(0,230,118,0.06);
  --red: #ff1744;
  --red-dim: rgba(255,23,68,0.06);
  --blue: #40c4ff;
  --blue-dim: rgba(64,196,255,0.06);
  --orange: #ff9100;
  --text-hi: #e1e8f0;
  --text-mid: #7b8fa4;
  --text-lo: #3d5068;
  --font-display: 'Rajdhani', sans-serif;
  --font-mono: 'Share Tech Mono', monospace;
}

/* ═══ Base ═══ */
html, body, .stApp {
  background: var(--void) !important;
  color: var(--text-hi);
  font-family: var(--font-mono);
}
[data-testid="stHeader"] { background: var(--void) !important; border-bottom: 1px solid var(--border); }
[data-testid="stSidebar"] { display: none !important; }
.main .block-container { padding: 0.5rem 1.25rem 1rem 1.25rem; max-width: 100%; }
.stMainBlockContainer { padding-top: 0.5rem !important; }

/* ═══ Scanlines ═══ */
.scanlines {
  position: fixed; top: 0; left: 0; right: 0; bottom: 0;
  background: repeating-linear-gradient(0deg, transparent, transparent 2px, rgba(0,0,0,0.015) 2px, rgba(0,0,0,0.015) 4px);
  pointer-events: none; z-index: 10000;
}

/* ═══ Scrollbar ═══ */
::-webkit-scrollbar { width: 5px; height: 5px; }
::-webkit-scrollbar-track { background: var(--void); }
::-webkit-scrollbar-thumb { background: var(--border); border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: var(--border-hi); }

/* ═══ Header ═══ */
.soldar-header {
  display: flex; align-items: center; gap: 16px;
  padding: 10px 0 12px 0; border-bottom: 1px solid var(--border);
  margin-bottom: 14px; flex-wrap: wrap;
}
.soldar-title {
  font-family: var(--font-display); font-size: 22px; font-weight: 700;
  color: var(--amber); letter-spacing: 4px;
  text-shadow: 0 0 20px var(--amber-glow), 0 0 40px rgba(255,167,38,0.1);
}

/* ═══ Radar Animation ═══ */
.radar-wrap {
  width: 28px; height: 28px; position: relative;
  border-radius: 50%; border: 1px solid rgba(64,196,255,0.3);
  overflow: hidden; flex-shrink: 0;
}
.radar-sweep {
  position: absolute; inset: 0; border-radius: 50%;
  background: conic-gradient(from 0deg, transparent 0deg, rgba(64,196,255,0.5) 60deg, transparent 60deg);
  animation: radarSpin 2.5s linear infinite;
}
.radar-sweep.stale {
  animation: none;
  background: conic-gradient(from 0deg, transparent 0deg, rgba(255,23,68,0.3) 60deg, transparent 60deg);
}
.radar-dot {
  position: absolute; top: 50%; left: 50%; width: 4px; height: 4px;
  background: var(--blue); border-radius: 50%; transform: translate(-50%, -50%);
  box-shadow: 0 0 6px var(--blue), 0 0 12px rgba(64,196,255,0.3);
}
.radar-dot.stale { background: var(--red); box-shadow: 0 0 6px var(--red); }
@keyframes radarSpin { to { transform: rotate(360deg); } }

/* ═══ Live Indicator ═══ */
.live-tag {
  display: flex; align-items: center; gap: 5px;
  font-family: var(--font-display); font-size: 12px; font-weight: 600;
  letter-spacing: 1px; text-transform: uppercase;
}
.live-tag.ok { color: var(--green); }
.live-tag.stale { color: var(--red); }
.live-pulse {
  width: 7px; height: 7px; border-radius: 50%; display: inline-block;
  animation: pulse 2s ease-in-out infinite;
}
.live-pulse.ok { background: var(--green); box-shadow: 0 0 8px var(--green); }
.live-pulse.stale { background: var(--red); box-shadow: 0 0 8px var(--red); animation: none; }
@keyframes pulse { 0%,100%{opacity:1;} 50%{opacity:0.3;} }

/* ═══ Header Stats ═══ */
.hstat { display: flex; flex-direction: column; align-items: center; gap: 0; }
.hstat-val { font-family: var(--font-mono); font-size: 14px; font-weight: bold; color: var(--text-hi); }
.hstat-lbl { font-family: var(--font-display); font-size: 9px; color: var(--text-lo); text-transform: uppercase; letter-spacing: 1px; }
.hstat-sep { color: var(--border); font-size: 14px; margin: 0 2px; }

/* ═══ HUD Panel ═══ */
.hud-panel {
  background: var(--panel); border: 1px solid var(--border);
  border-left: 2px solid var(--amber); padding: 14px; margin-bottom: 10px;
  box-shadow: inset 0 0 30px rgba(0,0,0,0.3);
}
.hud-panel.red { border-left-color: var(--red); }
.hud-panel.green { border-left-color: var(--green); }
.hud-panel.blue { border-left-color: var(--blue); }
.panel-title {
  font-family: var(--font-display); font-size: 10px; font-weight: 600;
  color: var(--text-lo); text-transform: uppercase; letter-spacing: 2px; margin-bottom: 10px;
}

/* ═══ Mini Stat ═══ */
.mini-stat {
  display: flex; justify-content: space-between; align-items: center;
  padding: 6px 0; border-bottom: 1px solid rgba(28,45,68,0.5);
}
.mini-stat:last-child { border-bottom: none; }
.mini-stat-lbl { font-size: 11px; color: var(--text-mid); }
.mini-stat-val { font-family: var(--font-mono); font-size: 13px; font-weight: bold; color: var(--text-hi); }

/* ═══ Health Row ═══ */
.health-row {
  display: flex; justify-content: space-between; align-items: center;
  padding: 4px 0; border-bottom: 1px solid rgba(28,45,68,0.3); font-size: 11px;
}
.health-row:last-child { border-bottom: none; }
.health-lbl { color: var(--text-mid); }
.health-val { font-family: var(--font-mono); color: var(--text-hi); }

/* ═══ Status Dots ═══ */
.dot-green { display:inline-block; width:6px; height:6px; background:var(--green); border-radius:50%; margin-right:4px; box-shadow:0 0 4px var(--green); }
.dot-yellow { display:inline-block; width:6px; height:6px; background:var(--orange); border-radius:50%; margin-right:4px; box-shadow:0 0 4px var(--orange); }
.dot-red { display:inline-block; width:6px; height:6px; background:var(--red); border-radius:50%; margin-right:4px; box-shadow:0 0 4px var(--red); }

/* ═══ Tabs ═══ */
.stTabs [data-baseweb="tab-list"] { background: transparent; border-bottom: 1px solid var(--border); gap: 0; }
.stTabs [data-baseweb="tab"] {
  background: transparent !important; color: var(--text-lo) !important;
  border: none !important; padding: 8px 16px !important;
  font-size: 11px !important; font-family: var(--font-display) !important;
  font-weight: 600 !important; letter-spacing: 1.5px !important;
  text-transform: uppercase !important; transition: color 0.2s !important;
}
.stTabs [data-baseweb="tab"]:hover { color: var(--text-mid) !important; }
.stTabs [aria-selected="true"] { color: var(--amber) !important; }
.stTabs [data-baseweb="tab-highlight"] { background: var(--amber) !important; height: 2px !important; }
.stTabs [data-baseweb="tab-border"] { display: none; }

/* ═══ Candidate Cards ═══ */
.ccard {
  background: var(--panel); border: 1px solid var(--border); border-left: 3px solid var(--border);
  padding: 10px 14px; margin-bottom: 8px; position: relative; overflow: hidden;
  transition: border-color 0.2s, box-shadow 0.2s;
}
.ccard:hover { border-color: var(--border-hi); box-shadow: inset 0 0 40px rgba(0,0,0,0.2), 0 0 1px rgba(255,167,38,0.1); }
.ccard::after {
  content: ''; position: absolute; left: 0; right: 0; height: 1px;
  background: linear-gradient(90deg, transparent, var(--blue), transparent);
  top: -1px; opacity: 0; transition: opacity 0.3s;
}
.ccard:hover::after { animation: scanLine 0.5s ease-out; opacity: 1; }
@keyframes scanLine { from{top:-1px;opacity:0.8;} to{top:100%;opacity:0;} }
.ccard-hot { border-left-color: var(--red); }
.ccard-warm { border-left-color: var(--orange); }
.ccard-watch { border-left-color: #ffd600; }
.ccard-low { border-left-color: var(--border); }

.crow1 { display: flex; align-items: center; gap: 10px; margin-bottom: 5px; }
.crow2, .crow3 { font-size: 12px; color: var(--text-mid); margin-bottom: 3px; }
.crow4 { display: flex; gap: 8px; align-items: center; flex-wrap: wrap; margin-top: 5px; }

.csym { font-family: var(--font-display); font-size: 16px; font-weight: 700; color: var(--text-hi); letter-spacing: 1px; }
.cname { color: var(--text-lo); font-size: 11px; }
.cscore { font-family: var(--font-mono); font-size: 14px; font-weight: bold; margin-left: auto; }
.cscore-hot { color: var(--red); text-shadow: 0 0 8px rgba(255,23,68,0.3); }
.cscore-warm { color: var(--orange); }
.cscore-watch { color: #ffd600; }
.cscore-low { color: var(--text-lo); }

/* ═══ Badges & Pills ═══ */
.badge { display:inline-block; padding:2px 8px; border-radius:2px; font-size:10px; font-weight:700; font-family:var(--font-display); letter-spacing:1px; }
.badge-hot { background:rgba(255,23,68,0.15); color:var(--red); border:1px solid rgba(255,23,68,0.3); }
.badge-warm { background:rgba(255,145,0,0.12); color:var(--orange); border:1px solid rgba(255,145,0,0.3); }
.badge-watch { background:rgba(255,214,0,0.1); color:#ffd600; border:1px solid rgba(255,214,0,0.25); }
.badge-low { background:rgba(61,80,104,0.15); color:var(--text-lo); border:1px solid rgba(61,80,104,0.3); }

.pill { display:inline-block; padding:1px 6px; border-radius:2px; font-size:10px; font-weight:bold; font-family:var(--font-mono); }
.pill-pass { background:rgba(0,230,118,0.1); color:var(--green); border:1px solid rgba(0,230,118,0.3); }
.pill-warn { background:rgba(255,145,0,0.1); color:var(--orange); border:1px solid rgba(255,145,0,0.3); }
.pill-fail { background:rgba(255,23,68,0.1); color:var(--red); border:1px solid rgba(255,23,68,0.3); }
.pill-na { background:rgba(61,80,104,0.1); color:var(--text-lo); border:1px solid rgba(61,80,104,0.25); }

/* ═══ Exit Banners ═══ */
.exit-banner {
  background: rgba(255,23,68,0.04); border: 1px solid rgba(255,23,68,0.2);
  border-left: 3px solid var(--red); padding: 8px 14px; margin-bottom: 8px;
}
.exit-title { font-family:var(--font-display); font-weight:700; color:var(--red); font-size:13px; letter-spacing:0.5px; }
.exit-detail { color:var(--text-mid); font-size:11px; margin-top:3px; font-family:var(--font-mono); }

/* ═══ Stat Metric Cards ═══ */
.smetric {
  background: var(--panel); border: 1px solid var(--border); border-top: 2px solid var(--amber);
  padding: 16px; text-align: center; box-shadow: inset 0 0 20px rgba(0,0,0,0.2);
}
.smetric-val { font-family:var(--font-mono); font-size:28px; font-weight:bold; color:var(--text-hi); }
.smetric-lbl { font-family:var(--font-display); font-size:10px; color:var(--text-lo); text-transform:uppercase; letter-spacing:1px; margin-top:4px; }

/* ═══ Grid Tables ═══ */
.grid-header {
  display: grid; gap: 8px; padding: 8px 12px; background: var(--base);
  font-family: var(--font-display); font-size: 10px; color: var(--text-lo);
  text-transform: uppercase; letter-spacing: 0.5px;
}
.grid-row {
  display: grid; gap: 8px; padding: 7px 12px;
  border-bottom: 1px solid rgba(28,45,68,0.3); font-size: 12px; align-items: center;
}

/* ═══ Rejected Table ═══ */
.rej-row { display:grid; grid-template-columns:80px 100px 1fr 100px; gap:10px; padding:7px 12px; border-bottom:1px solid rgba(28,45,68,0.3); font-size:12px; align-items:center; }
.rej-row.header { color:var(--text-lo); font-family:var(--font-display); font-size:10px; text-transform:uppercase; letter-spacing:1px; background:var(--base); }
.rej-row.data { background:rgba(255,23,68,0.02); }
.rej-sym { font-family:var(--font-display); font-weight:700; color:var(--text-hi); font-size:13px; }
.rej-addr { font-family:var(--font-mono); color:var(--text-lo); font-size:10px; }
.rej-reason { color:var(--text-mid); }
.rej-time { color:var(--text-lo); font-size:10px; font-family:var(--font-mono); }

/* ═══ Positions ═══ */
.pos-summary { display:flex; gap:10px; flex-wrap:wrap; margin-bottom:14px; }
.pos-summary-card { background:var(--panel); border:1px solid var(--border); border-top:2px solid var(--blue); padding:12px 18px; flex:1; min-width:130px; text-align:center; }
.pos-summary-val { font-family:var(--font-mono); font-size:22px; font-weight:bold; color:var(--text-hi); }
.pos-summary-lbl { font-family:var(--font-display); font-size:10px; color:var(--text-lo); text-transform:uppercase; letter-spacing:1px; }

/* ═══ No Data / Section Labels ═══ */
.nodata { color:var(--text-lo); font-size:13px; text-align:center; padding:30px; font-family:var(--font-mono); }
.sec-label {
  font-family:var(--font-display); font-size:11px; font-weight:600; color:var(--text-lo);
  text-transform:uppercase; letter-spacing:2px; margin:14px 0 8px 0;
  padding-bottom:4px; border-bottom:1px solid rgba(28,45,68,0.3);
}

/* ═══ Waiting Screen ═══ */
.waiting-wrap { text-align:center; padding:100px 20px; }
.waiting-title {
  font-family:var(--font-display); font-size:42px; font-weight:700; color:var(--amber);
  letter-spacing:8px; text-shadow:0 0 30px var(--amber-glow),0 0 60px rgba(255,167,38,0.1);
  margin-bottom:8px; animation:titlePulse 3s ease-in-out infinite;
}
@keyframes titlePulse { 0%,100%{opacity:1;} 50%{opacity:0.7;} }
.waiting-sub { color:var(--text-lo); font-size:13px; font-family:var(--font-mono); }
.waiting-radar {
  width:60px; height:60px; margin:30px auto 20px auto; position:relative;
  border-radius:50%; border:1px solid rgba(64,196,255,0.3); overflow:hidden;
}
.waiting-radar .wr-sweep {
  position:absolute; inset:0; border-radius:50%;
  background:conic-gradient(from 0deg,transparent 0deg,rgba(64,196,255,0.5) 60deg,transparent 60deg);
  animation:radarSpin 2.5s linear infinite;
}
.waiting-radar .wr-dot {
  position:absolute; top:50%; left:50%; width:5px; height:5px;
  background:var(--blue); border-radius:50%; transform:translate(-50%,-50%);
  box-shadow:0 0 8px var(--blue),0 0 16px rgba(64,196,255,0.3);
}

/* ═══ Utility ═══ */
.mono { font-family: var(--font-mono); }
.up { color: var(--green); font-family: var(--font-mono); }
.dn { color: var(--red); font-family: var(--font-mono); }
.flat { color: var(--text-lo); font-family: var(--font-mono); }
.btn-copy { background:rgba(64,196,255,0.08); color:var(--blue); border:1px solid rgba(64,196,255,0.25); border-radius:2px; padding:1px 7px; font-size:10px; cursor:pointer; font-family:var(--font-mono); transition:background 0.2s; }
.btn-copy:hover { background:rgba(64,196,255,0.18); }
.dex-link { color:var(--blue); text-decoration:none; font-size:10px; transition:color 0.2s; }
.dex-link:hover { text-decoration:underline; color:var(--amber); }
.ca-addr { font-family:var(--font-mono); font-size:10px; color:var(--text-lo); margin-left:auto; }

/* ═══ Streamlit Overrides ═══ */
.stSlider > label, .stRadio > label { color:var(--text-lo) !important; font-size:11px !important; font-family:var(--font-display) !important; text-transform:uppercase !important; letter-spacing:0.5px !important; }
.stRadio div[role="radiogroup"] label { color:var(--text-mid) !important; font-size:12px !important; }
[data-testid="stMetric"] { background:var(--panel) !important; border:1px solid var(--border) !important; }
[data-testid="stMetricLabel"] p { color:var(--text-lo) !important; font-size:11px !important; }
[data-testid="stMetricValue"] { font-family:var(--font-mono) !important; }
.stDataFrame { border:1px solid var(--border); overflow:hidden; }
.stButton button {
  background:rgba(255,167,38,0.08) !important; color:var(--amber) !important;
  border:1px solid rgba(255,167,38,0.3) !important; border-radius:2px !important;
  font-family:var(--font-display) !important; font-weight:600 !important;
  letter-spacing:1px !important; text-transform:uppercase !important;
  font-size:11px !important; transition:all 0.2s !important;
}
.stButton button:hover { background:rgba(255,167,38,0.18) !important; border-color:var(--amber) !important; box-shadow:0 0 12px rgba(255,167,38,0.15) !important; }
hr { border-color: var(--border); margin: 10px 0; }
.soldar-footer { text-align:center; padding:20px 0 10px 0; color:var(--text-lo); font-size:10px; font-family:var(--font-mono); border-top:1px solid rgba(28,45,68,0.3); margin-top:20px; }`;
  // Also inject Google Fonts into parent
  if (!window.parent.document.getElementById('soldar-fonts')) {
    var link = window.parent.document.createElement('link');
    link.id = 'soldar-fonts';
    link.rel = 'stylesheet';
    link.href = 'https://fonts.googleapis.com/css2?family=Rajdhani:wght@400;600;700&family=Share+Tech+Mono&display=swap';
    window.parent.document.head.appendChild(link);
  }
})();
</script>
""", height=0)

# ─── Constants ────────────────────────────────────────────────────────────────
PROJECT_ROOT  = Path(__file__).parent
STATE_FILE    = PROJECT_ROOT / "data" / "state.json"
TRADES_DB     = PROJECT_ROOT / "data" / "trades.db"
OUTCOMES_FILE = PROJECT_ROOT / "data" / "outcomes.json"
MODELS_DIR    = PROJECT_ROOT / "ml" / "models"

MODEL_REGISTRY: dict[str, dict[str, str]] = {
    "XGBoost Baseline": {
        "metrics": "xgb_metrics.json",
        "confusion_matrix": "xgb_baseline_confusion_matrix.png",
        "feature_importance": "xgb_baseline_feature_importance.png",
        "roc_curve": "xgb_baseline_roc_curve.png",
    },
    "XGBoost Tuned": {
        "metrics": "xgb_tuned_metrics.json",
        "confusion_matrix": "xgb_tuned_confusion_matrix.png",
        "feature_importance": "xgb_tuned_feature_importance.png",
        "roc_curve": "xgb_tuned_roc_curve.png",
    },
    "LightGBM": {
        "metrics": "lgbm_metrics.json",
        "confusion_matrix": "lgbm_confusion_matrix.png",
        "feature_importance": "lgbm_feature_importance.png",
        "roc_curve": "lgbm_roc_curve.png",
    },
    "LSTM (Sapienza)": {
        "metrics": "lstm_sapienza_metrics.json",
        "confusion_matrix": "lstm_sapienza_confusion_matrix.png",
        "training_curves": "lstm_sapienza_curves.png",
        "roc_curve": "lstm_sapienza_roc_curve.png",
    },
    "LSTM (MemeTransactions)": {
        "metrics": "lstm_metrics.json",
        "confusion_matrix": "lstm_confusion_matrix.png",
        "training_curves": "lstm_training_curves.png",
        "roc_curve": "lstm_roc_curve.png",
    },
}


# ─── Data loading ─────────────────────────────────────────────────────────────
@st.cache_data(ttl=30)
def load_state() -> dict | None:
    if not STATE_FILE.exists():
        return None
    try:
        with open(STATE_FILE) as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return None


@st.cache_data(ttl=30)
def load_trades_summary() -> dict:
    defaults = {"open_count": 0, "daily_pnl": 0.0, "win_rate": None, "total_pnl": 0.0}
    if not TRADES_DB.exists():
        return defaults
    try:
        conn = sqlite3.connect(str(TRADES_DB))
        conn.row_factory = sqlite3.Row
        open_count = conn.execute(
            "SELECT COUNT(*) FROM positions WHERE status='open'"
        ).fetchone()[0]
        today_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        daily_pnl = conn.execute(
            "SELECT COALESCE(SUM(pnl_usd),0) FROM trades WHERE action!='buy' AND timestamp>=?",
            (today_str,),
        ).fetchone()[0]
        all_sells = conn.execute(
            "SELECT pnl_usd FROM trades WHERE action!='buy'"
        ).fetchall()
        total_pnl = sum(float(r["pnl_usd"] or 0) for r in all_sells)
        win_rate = None
        if all_sells:
            wins = sum(1 for r in all_sells if float(r["pnl_usd"] or 0) > 0)
            win_rate = wins / len(all_sells) * 100
        conn.close()
        return {"open_count": open_count, "daily_pnl": daily_pnl, "win_rate": win_rate, "total_pnl": total_pnl}
    except Exception:
        return defaults


@st.cache_data(ttl=60)
def load_outcomes() -> dict | None:
    if not OUTCOMES_FILE.exists():
        return None
    try:
        with open(OUTCOMES_FILE) as f:
            return json.load(f)
    except Exception:
        return None


@st.cache_data(ttl=300)
def load_model_metrics(name: str) -> dict | None:
    info = MODEL_REGISTRY.get(name)
    if not info:
        return None
    path = MODELS_DIR / info["metrics"]
    if not path.exists():
        return None
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return None


# ─── Format helpers ───────────────────────────────────────────────────────────
def fmt_usd(v) -> str:
    if v is None:
        return "$0"
    v = float(v)
    if abs(v) >= 1_000_000:
        return f"${v/1_000_000:.2f}M"
    if abs(v) >= 1_000:
        return f"${v/1_000:.1f}K"
    return f"${v:.0f}"


def fmt_pct(v) -> str:
    if v is None:
        return "0.0%"
    return f"{float(v):+.1f}%"


def price_span(v) -> str:
    v = v or 0
    if v > 0:
        return f'<span class="up">+{v:.1f}%</span>'
    if v < 0:
        return f'<span class="dn">{v:.1f}%</span>'
    return f'<span class="flat">0.0%</span>'


def legit_pill(ok, label: str) -> str:
    if ok is None:
        return f'<span class="pill pill-na">? {label}</span>'
    if ok:
        return f'<span class="pill pill-pass">\u2713 {label}</span>'
    return f'<span class="pill pill-fail">\u2717 {label}</span>'


def verdict_pill(v) -> str:
    if v == "PASS":
        return '<span class="pill pill-pass">\u2713 PASS</span>'
    if v == "WARN":
        return '<span class="pill pill-warn">\u26a0 WARN</span>'
    if v == "FAIL":
        return '<span class="pill pill-fail">\u2717 FAIL</span>'
    return '<span class="pill pill-na">\u2014 N/A</span>'


def level_attrs(score: int) -> tuple[str, str, str, str]:
    if score >= 80:
        return "HOT", "badge-hot", "ccard-hot", "cscore-hot"
    if score >= 65:
        return "WARM", "badge-warm", "ccard-warm", "cscore-warm"
    if score >= 50:
        return "WATCH", "badge-watch", "ccard-watch", "cscore-watch"
    return "LOW", "badge-low", "ccard-low", "cscore-low"


def dark_plotly_layout(fig: go.Figure, **kwargs) -> go.Figure:
    fig.update_layout(
        paper_bgcolor="#040608",
        plot_bgcolor="#0c1220",
        font=dict(color="#7b8fa4", family="Share Tech Mono, monospace", size=11),
        margin=dict(l=40, r=20, t=40, b=30),
        xaxis=dict(gridcolor="#1c2d44", zerolinecolor="#1c2d44", tickfont=dict(size=10)),
        yaxis=dict(gridcolor="#1c2d44", zerolinecolor="#1c2d44", tickfont=dict(size=10)),
        legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor="#1c2d44"),
        **kwargs,
    )
    return fig


# ─── Card builders ────────────────────────────────────────────────────────────
def build_candidate_card(c: dict) -> str:
    score = c.get("score", 0)
    symbol = c.get("symbol", "?")
    name = c.get("name", "")
    address = c.get("address", "")
    age = c.get("age_minutes")
    liq = c.get("liquidity_usd") or 0
    mcap = c.get("mcap_usd") or 0
    vol_liq = c.get("vol_liq_ratio") or 0
    p5m = c.get("price_change_5m") or 0
    p1h = c.get("price_change_1h") or 0
    buys = c.get("buys_5m") or 0
    sells = c.get("sells_5m") or 0
    verdict = c.get("legit_verdict")
    mint_ok = c.get("legit_mint_revoked")
    frz_ok = c.get("legit_freeze_revoked")
    lp_lock = c.get("legit_lp_locked_pct")
    top1 = c.get("legit_top1_pct")
    rc = c.get("legit_rc_score")
    ml_score = c.get("ml_score")
    dex_url = c.get("url") or f"https://dexscreener.com/solana/{address}"

    lvl, badge_cls, card_cls, score_cls = level_attrs(score)
    age_str = f"{age:.0f}m" if age is not None else "?"
    addr_short = address[:8] + "\u2026" + address[-4:] if len(address) > 12 else address

    if lp_lock is not None:
        lp_cls = "pill-pass" if lp_lock >= 90 else ("pill-warn" if lp_lock >= 50 else "pill-fail")
        lp_pill = f'<span class="pill {lp_cls}">LP {lp_lock:.0f}%</span>'
    else:
        lp_pill = '<span class="pill pill-na">LP ?</span>'

    if top1 is not None:
        t_cls = "pill-pass" if top1 <= 10 else ("pill-warn" if top1 <= 20 else "pill-fail")
        top1_pill = f'<span class="pill {t_cls}">Top {top1:.1f}%</span>'
    else:
        top1_pill = ""

    rc_str = f'<span class="mono" style="font-size:10px;color:var(--text-lo);">RC:{rc}</span>' if rc is not None else ""

    if ml_score is not None:
        ml_pct = ml_score * 100
        ml_cls = "pill-pass" if ml_pct >= 60 else ("pill-warn" if ml_pct >= 30 else "pill-fail")
        ml_pill = f'<span class="pill {ml_cls}">ML {ml_pct:.0f}%</span>'
    else:
        ml_pill = ""

    js_copy = (
        f"navigator.clipboard.writeText('{address}')"
        ".then(function(){{var b=this;b.textContent='\u2713 Copied';"
        "setTimeout(function(){{b.textContent='\U0001f4cb Copy CA'}},2000)}}.bind(this))()"
    )
    copy_btn = f'<button class="btn-copy" onclick="{js_copy}">\U0001f4cb Copy CA</button>'

    return f"""
<div class="ccard {card_cls}">
  <div class="crow1">
    <span class="badge {badge_cls}">{lvl}</span>
    <span class="csym">{symbol}</span>
    <span class="cname">{name}</span>
    <span class="{score_cls} cscore">{score}/100</span>
  </div>
  <div class="crow2">
    Age <span class="mono">{age_str}</span> \u2502
    Liq <span class="mono">{fmt_usd(liq)}</span> \u2502
    MCap <span class="mono">{fmt_usd(mcap)}</span> \u2502
    V/L <span class="mono">{vol_liq:.1f}x</span>
  </div>
  <div class="crow3">
    5m {price_span(p5m)} \u2502
    1h {price_span(p1h)} \u2502
    B:<span style="color:var(--green);font-family:var(--font-mono)">{buys}</span>
    S:<span style="color:var(--red);font-family:var(--font-mono)">{sells}</span>
  </div>
  <div class="crow4">
    {verdict_pill(verdict)} {legit_pill(mint_ok, "Mint")} {legit_pill(frz_ok, "Frz")}
    {lp_pill} {top1_pill} {rc_str} {ml_pill}
    {copy_btn}
    <a href="{dex_url}" target="_blank" class="dex-link">DexScreener \u2197</a>
    <span class="ca-addr">{addr_short}</span>
  </div>
</div>"""


def build_exit_banner(ea: dict) -> str:
    urgency = ea.get("urgency", "LOW")
    symbol = ea.get("symbol", "?")
    reason = ea.get("exit_reason", "")
    p5m = ea.get("current_price_change_5m") or 0
    p1h = ea.get("current_price_change_1h") or 0
    urg_icon = {"HIGH": "\U0001f534", "MEDIUM": "\U0001f7e0", "LOW": "\U0001f7e1"}.get(urgency, "\u26aa")
    return f"""
<div class="exit-banner">
  <div class="exit-title">\U0001f6a8 EXIT: {symbol} \u2014 {reason} \u2502 {urg_icon} {urgency}</div>
  <div class="exit-detail">5m: {fmt_pct(p5m)} \u2502 1h: {fmt_pct(p1h)}</div>
</div>"""


# ─── Load state ───────────────────────────────────────────────────────────────
state = load_state()

if state is None:
    st.markdown("""
<div class="waiting-wrap">
  <div class="waiting-radar"><div class="wr-sweep"></div><div class="wr-dot"></div></div>
  <div class="waiting-title">SOLDAR</div>
  <div class="waiting-sub">Awaiting screener data\u2026</div>
  <div class="waiting-sub" style="margin-top:10px;font-size:11px;">
    <code>data/state.json</code> not found \u2014 screener creates it on next cycle.<br>
    Auto-refreshes every 90s.
  </div>
</div>""", unsafe_allow_html=True)
    st.stop()

# ─── Extract state ────────────────────────────────────────────────────────────
candidates: list[dict] = state.get("candidates", [])
hard_rejected: list[dict] = state.get("hard_rejected", [])
cycle_history: list[dict] = state.get("cycle_history", [])
exit_alerts: list[dict] = state.get("exit_alerts", [])

last_updated_str = state.get("last_updated", "")
try:
    updated_dt = datetime.fromisoformat(last_updated_str)
    seconds_ago = int((datetime.now(timezone.utc) - updated_dt).total_seconds())
except (ValueError, TypeError):
    seconds_ago = 0

cycles_run = max(state.get("cycle_count", 0), len(cycle_history))
scanned_count = state.get("candidates_scanned", 0)
passed_count = state.get("passed_filters", len(candidates))

candidates.sort(key=lambda c: c.get("score", 0), reverse=True)
top_score = candidates[0]["score"] if candidates else 0

hot_count = sum(1 for c in candidates if c.get("score", 0) >= 80)
warm_count = sum(1 for c in candidates if 65 <= c.get("score", 0) < 80)

age_text = f"{seconds_ago}s ago" if seconds_ago < 120 else f"{seconds_ago // 60}m ago"
api_ok = seconds_ago < 300
stale_cls = "" if api_ok else " stale"
live_cls = "ok" if api_ok else "stale"
live_text = "LIVE" if api_ok else "STALE"

# ─── Header bar ───────────────────────────────────────────────────────────────
st.markdown(f"""
<div class="soldar-header">
  <span class="soldar-title">\U0001f4e1 SOLDAR</span>
  <div class="radar-wrap"><div class="radar-sweep{stale_cls}"></div><div class="radar-dot{stale_cls}"></div></div>
  <div class="live-tag {live_cls}"><span class="live-pulse {live_cls}"></span>{live_text}</div>
  <span class="hstat-sep">\u2502</span>
  <div class="hstat"><span class="hstat-val">{age_text}</span><span class="hstat-lbl">Last Scan</span></div>
  <span class="hstat-sep">\u2502</span>
  <div class="hstat"><span class="hstat-val mono">{scanned_count}</span><span class="hstat-lbl">Scanned</span></div>
  <span class="hstat-sep">\u2502</span>
  <div class="hstat"><span class="hstat-val mono">{passed_count}</span><span class="hstat-lbl">Passed</span></div>
  <span class="hstat-sep">\u2502</span>
  <div class="hstat"><span class="hstat-val mono" style="color:var(--red)">{hot_count}</span><span class="hstat-lbl">Hot</span></div>
  <span class="hstat-sep">\u2502</span>
  <div class="hstat"><span class="hstat-val mono" style="color:var(--orange)">{warm_count}</span><span class="hstat-lbl">Warm</span></div>
  <span class="hstat-sep">\u2502</span>
  <div class="hstat"><span class="hstat-val mono">{top_score}</span><span class="hstat-lbl">Top Score</span></div>
  <span class="hstat-sep">\u2502</span>
  <div class="hstat"><span class="hstat-val mono">{cycles_run}</span><span class="hstat-lbl">Cycles</span></div>
</div>
""", unsafe_allow_html=True)

# ─── Main layout ──────────────────────────────────────────────────────────────
left_col, main_col = st.columns([1.4, 3.2], gap="medium")

# ═══ LEFT COLUMN ═══
with left_col:
    # Filters
    st.markdown('<div class="panel-title" style="font-family:var(--font-display);font-size:10px;font-weight:600;color:var(--text-lo);text-transform:uppercase;letter-spacing:2px;margin-bottom:8px;">Filters</div>', unsafe_allow_html=True)
    min_score = st.slider("Min score", 0, 100, 0, key="min_score")
    max_age = st.slider("Max age (min)", 0, 360, 360, key="max_age")
    legit_filter = st.radio("Legitimacy", ["All", "PASS only", "PASS + WARN"], index=0, key="legit_filter")
    st.markdown('<hr style="border-color:var(--border);margin:8px 0 12px 0;">', unsafe_allow_html=True)

    # Quick stats
    trades = load_trades_summary()
    daily_pnl = trades["daily_pnl"]
    pnl_cls = "up" if daily_pnl > 0 else ("dn" if daily_pnl < 0 else "")
    wr_str = f"{trades['win_rate']:.0f}%" if trades["win_rate"] is not None else "\u2014"

    st.markdown(f"""
<div class="hud-panel green">
  <div class="panel-title">Quick Stats</div>
  <div class="mini-stat"><span class="mini-stat-lbl">HOT Alerts</span><span class="mini-stat-val" style="color:var(--red)">{hot_count}</span></div>
  <div class="mini-stat"><span class="mini-stat-lbl">Open Trades</span><span class="mini-stat-val">{trades['open_count']}</span></div>
  <div class="mini-stat"><span class="mini-stat-lbl">Daily PnL</span><span class="mini-stat-val {pnl_cls}">${daily_pnl:+.2f}</span></div>
  <div class="mini-stat"><span class="mini-stat-lbl">Win Rate</span><span class="mini-stat-val">{wr_str}</span></div>
</div>""", unsafe_allow_html=True)

    # Health
    api_indicator = f'<span class="dot-green"></span>LIVE' if api_ok else '<span class="dot-red"></span>STALE'
    exit_count = len(exit_alerts)
    st.markdown(f"""
<div class="hud-panel blue">
  <div class="panel-title">Screener Health</div>
  <div class="health-row"><span class="health-lbl">API Status</span><span class="health-val">{api_indicator}</span></div>
  <div class="health-row"><span class="health-lbl">Last scan</span><span class="health-val mono">{age_text}</span></div>
  <div class="health-row"><span class="health-lbl">Candidates</span><span class="health-val mono">{len(candidates)}</span></div>
  <div class="health-row"><span class="health-lbl">Rejected</span><span class="health-val mono">{len(hard_rejected)}</span></div>
  <div class="health-row"><span class="health-lbl">Exit alerts</span><span class="health-val mono" style="color:{'var(--red)' if exit_count else 'var(--text-lo)'}">{exit_count}</span></div>
</div>""", unsafe_allow_html=True)

    # Backtest button
    st.markdown('<div class="panel-title" style="font-family:var(--font-display);font-size:10px;font-weight:600;color:var(--text-lo);text-transform:uppercase;letter-spacing:2px;margin-bottom:8px;">Actions</div>', unsafe_allow_html=True)
    if st.button("Run Backtest", key="run_backtest_btn"):
        with st.spinner("Running backtest..."):
            try:
                proc = subprocess.run(
                    ["python", "ml/backtest.py", "--save-report"],
                    cwd=str(PROJECT_ROOT), capture_output=True, text=True, timeout=120,
                )
                if proc.returncode == 0:
                    st.success("Backtest complete! Switch to BACKTEST tab.")
                else:
                    st.error(f"Backtest failed: {proc.stderr[-200:]}")
            except Exception as exc:
                st.error(f"Error: {exc}")

# ─── Apply filters ────────────────────────────────────────────────────────────
filtered: list[dict] = []
for c in candidates:
    if c.get("score", 0) < min_score:
        continue
    age = c.get("age_minutes")
    if age is not None and age > max_age:
        continue
    v = c.get("legit_verdict")
    if legit_filter == "PASS only" and v != "PASS":
        continue
    if legit_filter == "PASS + WARN" and v not in ("PASS", "WARN"):
        continue
    filtered.append(c)

# ═══ RIGHT COLUMN ═══
with main_col:
    rej_label = f"REJECTED ({len(hard_rejected)})" if hard_rejected else "REJECTED"
    ea_label = f"RADAR ({len(filtered)})"
    tab_radar, tab_intel, tab_rejected, tab_positions, tab_backtest, tab_models, tab_rl = st.tabs(
        [ea_label, "INTEL", rej_label, "POSITIONS", "BACKTEST", "MODELS", "RL AGENT"]
    )

    # ── Tab: RADAR ────────────────────────────────────────────────────────
    with tab_radar:
        if exit_alerts:
            for ea in exit_alerts:
                st.markdown(build_exit_banner(ea), unsafe_allow_html=True)
            st.markdown("<hr>", unsafe_allow_html=True)

        if not filtered:
            st.markdown('<div class="nodata">No candidates match current filters.</div>', unsafe_allow_html=True)
        else:
            st.markdown(
                f'<div class="sec-label">Showing {len(filtered)} contact(s)</div>',
                unsafe_allow_html=True,
            )
            for c in filtered:
                st.markdown(build_candidate_card(c), unsafe_allow_html=True)

    # ── Tab: INTEL ────────────────────────────────────────────────────────
    with tab_intel:
        scores = [c.get("score", 0) for c in candidates]
        avg_score = sum(scores) / len(scores) if scores else 0
        wr_display = f"{trades['win_rate']:.1f}%" if trades["win_rate"] is not None else "\u2014"

        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.markdown(f'<div class="smetric"><div class="smetric-val">{cycles_run}</div><div class="smetric-lbl">Cycles Run</div></div>', unsafe_allow_html=True)
        with m2:
            st.markdown(f'<div class="smetric"><div class="smetric-val">{scanned_count}</div><div class="smetric-lbl">Coins Scanned</div></div>', unsafe_allow_html=True)
        with m3:
            st.markdown(f'<div class="smetric"><div class="smetric-val" style="color:var(--red)">{hot_count}</div><div class="smetric-lbl">HOT Alerts Now</div></div>', unsafe_allow_html=True)
        with m4:
            st.markdown(f'<div class="smetric"><div class="smetric-val">{wr_display}</div><div class="smetric-lbl">Paper Win Rate</div></div>', unsafe_allow_html=True)

        st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)

        chart_left, chart_right = st.columns(2)

        with chart_left:
            if cycle_history:
                recent = cycle_history[-50:]
                tscores = [h.get("top_score", 0) for h in recent]
                passed_hist = [h.get("passed", 0) for h in recent]

                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=list(range(len(tscores))), y=tscores,
                    mode="lines+markers",
                    line=dict(color="#ffa726", width=2),
                    marker=dict(size=4, color="#ffa726"),
                    fill="tozeroy", fillcolor="rgba(255,167,38,0.06)",
                    name="Top Score",
                ))
                fig.add_trace(go.Scatter(
                    x=list(range(len(passed_hist))), y=passed_hist,
                    mode="lines",
                    line=dict(color="#40c4ff", width=1, dash="dot"),
                    name="Passed", yaxis="y2",
                ))
                fig.add_hline(y=80, line_dash="dot", line_color="rgba(255,23,68,0.4)",
                              annotation_text="HOT", annotation_font_color="#ff1744")
                fig.add_hline(y=65, line_dash="dot", line_color="rgba(255,145,0,0.4)",
                              annotation_text="WARM", annotation_font_color="#ff9100")
                dark_plotly_layout(fig, title="Top Score / Cycle (last 50)", yaxis_range=[0, 100])
                fig.update_layout(
                    title_font=dict(color="#7b8fa4", size=12), showlegend=True,
                    legend=dict(x=0.01, y=0.99, font=dict(size=9)),
                    yaxis2=dict(overlaying="y", side="right", showgrid=False,
                                tickfont=dict(size=10, color="#40c4ff"), title=""),
                )
                st.plotly_chart(fig, width="stretch")
            else:
                st.markdown('<div class="nodata">No cycle history yet.</div>', unsafe_allow_html=True)

        with chart_right:
            if scores:
                bins = list(range(0, 110, 10))
                labels = [f"{b}\u2013{b+9}" for b in bins[:-1]]
                counts = [0] * (len(bins) - 1)
                for s in scores:
                    idx = min(int(s) // 10, len(counts) - 1)
                    counts[idx] += 1

                colors = []
                for b in bins[:-1]:
                    if b >= 80:
                        colors.append("rgba(255,23,68,0.8)")
                    elif b >= 60:
                        colors.append("rgba(255,145,0,0.8)")
                    elif b >= 50:
                        colors.append("rgba(255,214,0,0.8)")
                    else:
                        colors.append("rgba(28,45,68,0.8)")

                fig2 = go.Figure(go.Bar(
                    x=labels, y=counts, marker_color=colors,
                    marker_line=dict(color="#040608", width=1),
                ))
                dark_plotly_layout(fig2, title="Score Distribution")
                fig2.update_layout(title_font=dict(color="#7b8fa4", size=12), showlegend=False)
                st.plotly_chart(fig2, width="stretch")
            else:
                st.markdown('<div class="nodata">No candidates to chart.</div>', unsafe_allow_html=True)

        # Outcomes summary
        outcomes = load_outcomes()
        if outcomes:
            peak_gains = [float(o.get("peak_gain_pct") or 0) for o in outcomes.values()]
            total_tracked = len(outcomes)
            avg_peak = sum(peak_gains) / len(peak_gains) if peak_gains else 0
            profitable = sum(1 for g in peak_gains if g > 10)
            hit_rate = profitable / total_tracked * 100 if total_tracked else 0

            st.markdown(f"""
<div class="hud-panel green">
  <div class="panel-title">Outcome Tracking</div>
  <div class="mini-stat"><span class="mini-stat-lbl">Tokens Tracked</span><span class="mini-stat-val">{total_tracked}</span></div>
  <div class="mini-stat"><span class="mini-stat-lbl">Avg Peak Gain</span><span class="mini-stat-val {'up' if avg_peak > 0 else 'dn'}">{avg_peak:+.1f}%</span></div>
  <div class="mini-stat"><span class="mini-stat-lbl">Hit Rate (&gt;10%)</span><span class="mini-stat-val">{hit_rate:.0f}%</span></div>
</div>""", unsafe_allow_html=True)

        # Top candidates table
        if candidates:
            st.markdown('<div class="sec-label">Top Candidates This Cycle</div>', unsafe_allow_html=True)
            table_data = []
            for c in candidates[:20]:
                lvl_str, _, _, _ = level_attrs(c.get("score", 0))
                table_data.append({
                    "Symbol": c.get("symbol", "?"),
                    "Score": c.get("score", 0),
                    "Level": lvl_str,
                    "Age(m)": f"{c.get('age_minutes', 0):.0f}",
                    "Liq": fmt_usd(c.get("liquidity_usd")),
                    "MCap": fmt_usd(c.get("mcap_usd")),
                    "5m%": fmt_pct(c.get("price_change_5m")),
                    "1h%": fmt_pct(c.get("price_change_1h")),
                    "Vol/Liq": f"{c.get('vol_liq_ratio', 0):.1f}x",
                    "Legit": c.get("legit_verdict") or "\u2014",
                    "ML": f"{c['ml_score']:.0%}" if c.get("ml_score") is not None else "\u2014",
                })
            st.dataframe(table_data, width="stretch", hide_index=True)

    # ── Tab: REJECTED ─────────────────────────────────────────────────────
    with tab_rejected:
        if not hard_rejected:
            st.markdown('<div class="nodata">No hard rejections this cycle.</div>', unsafe_allow_html=True)
        else:
            rows_html = '<div class="rej-row header"><span>Symbol</span><span>Address</span><span>Rejection Reason</span><span>Time</span></div>'
            for r in hard_rejected:
                sym = r.get("symbol", "?")
                addr = r.get("address", "?")
                addr_s = addr[:8] + "\u2026" + addr[-4:] if len(addr) > 12 else addr
                reasons = "; ".join(r.get("reasons", []))
                ts = r.get("timestamp", "")
                ts_s = ts[11:16] if len(ts) >= 16 else ts
                rows_html += f"""
<div class="rej-row data">
  <span class="rej-sym">{sym}</span>
  <span class="rej-addr">{addr_s}</span>
  <span class="rej-reason">{reasons}</span>
  <span class="rej-time">{ts_s}</span>
</div>"""
            st.markdown(
                f'<div style="background:var(--panel);border:1px solid var(--border);overflow:hidden">{rows_html}</div>',
                unsafe_allow_html=True,
            )

    # ── Tab: POSITIONS ────────────────────────────────────────────────────
    with tab_positions:
        if not TRADES_DB.exists():
            st.markdown('<div class="nodata">No trades database \u2014 paper trading hasn\'t executed any trades yet.</div>', unsafe_allow_html=True)
        else:
            try:
                conn = sqlite3.connect(str(TRADES_DB))
                conn.row_factory = sqlite3.Row

                all_sells = conn.execute("SELECT pnl_usd, pnl_pct FROM trades WHERE action!='buy'").fetchall()
                open_rows = conn.execute("SELECT * FROM positions WHERE status='open' ORDER BY entry_time DESC").fetchall()

                total_pnl = sum(float(r["pnl_usd"] or 0) for r in all_sells) if all_sells else 0
                total_trades = len(all_sells)
                wins = sum(1 for r in all_sells if float(r["pnl_usd"] or 0) > 0)
                win_rate_pct = wins / total_trades * 100 if total_trades else 0
                best_trade = max((float(r["pnl_pct"] or 0) for r in all_sells), default=0)

                pnl_color = "var(--green)" if total_pnl > 0 else ("var(--red)" if total_pnl < 0 else "var(--text-lo)")

                st.markdown(f"""
<div class="pos-summary">
  <div class="pos-summary-card"><div class="pos-summary-val" style="color:{pnl_color}">${total_pnl:+.2f}</div><div class="pos-summary-lbl">Total PnL</div></div>
  <div class="pos-summary-card"><div class="pos-summary-val">{win_rate_pct:.1f}%</div><div class="pos-summary-lbl">Win Rate</div></div>
  <div class="pos-summary-card"><div class="pos-summary-val">{len(open_rows)}</div><div class="pos-summary-lbl">Open Positions</div></div>
  <div class="pos-summary-card"><div class="pos-summary-val" style="color:var(--green)">{best_trade:+.1f}%</div><div class="pos-summary-lbl">Best Trade</div></div>
</div>""", unsafe_allow_html=True)

                st.markdown('<div class="sec-label">Open Positions</div>', unsafe_allow_html=True)
                if not open_rows:
                    st.markdown('<div class="nodata">No open positions.</div>', unsafe_allow_html=True)
                else:
                    open_data = []
                    for r in open_rows:
                        ep = float(r["entry_price"] or 0)
                        cp = float(r["current_price"] or 0)
                        pnl_pct = ((cp - ep) / ep * 100) if ep else 0
                        try:
                            entry_dt = datetime.fromisoformat(r["entry_time"])
                            held = datetime.now(timezone.utc) - entry_dt
                            held_s = f"{held.seconds // 3600}h {(held.seconds % 3600) // 60}m"
                            if held.days:
                                held_s = f"{held.days}d {held_s}"
                        except Exception:
                            held_s = "?"
                        open_data.append({
                            "Symbol": r["symbol"],
                            "Entry": f"${ep:.8f}",
                            "Current": f"${cp:.8f}",
                            "PnL%": f"{pnl_pct:+.1f}%",
                            "Size": f"${r['entry_amount_usd']:.2f}",
                            "Held": held_s,
                            "Score": r["screener_score"],
                        })
                    st.dataframe(open_data, width="stretch", hide_index=True)

                st.markdown('<div class="sec-label">Recent Closed Trades (last 20)</div>', unsafe_allow_html=True)
                closed_rows = conn.execute(
                    """SELECT t.*, p.symbol, p.entry_price, p.entry_amount_usd
                       FROM trades t JOIN positions p ON t.position_id = p.id
                       WHERE t.action != 'buy' ORDER BY t.timestamp DESC LIMIT 20"""
                ).fetchall()

                if not closed_rows:
                    st.markdown('<div class="nodata">No closed trades yet.</div>', unsafe_allow_html=True)
                else:
                    closed_data = []
                    for r in closed_rows:
                        pnl_pct_val = float(r["pnl_pct"] or 0)
                        closed_data.append({
                            "Pos#": r["position_id"],
                            "Symbol": r["symbol"],
                            "Action": r["action"],
                            "Entry": f"${r['entry_price']:.8f}",
                            "Exit": f"${r['price']:.8f}",
                            "PnL $": f"${r['pnl_usd']:+.2f}",
                            "PnL%": f"{pnl_pct_val:+.1f}%",
                            "Reason": r["exit_reason"] or "\u2014",
                            "Time": (r["timestamp"] or "")[:16],
                        })
                    st.dataframe(closed_data, width="stretch", hide_index=True)

                conn.close()
            except Exception as e:
                st.markdown(f'<div class="nodata">Error loading trades: {e}</div>', unsafe_allow_html=True)

    # ── Tab: BACKTEST ─────────────────────────────────────────────────────
    with tab_backtest:
        BT_RESULTS = MODELS_DIR / "backtest_results.json"
        BT_CURVES = MODELS_DIR / "backtest_equity_curves.json"

        if not BT_RESULTS.exists():
            st.markdown(
                '<div class="nodata">No backtest data yet \u2014 run '
                '<code>python ml/backtest.py</code> or click "Run Backtest" in the sidebar.</div>'
            , unsafe_allow_html=True)
        else:
            try:
                with open(BT_RESULTS) as f:
                    bt_results = json.load(f)

                st.markdown('<div class="sec-label">Strategy Comparison</div>', unsafe_allow_html=True)

                if bt_results:
                    best_vals = {
                        "precision": max(r["precision"] for r in bt_results),
                        "win_rate": max(r["win_rate"] for r in bt_results),
                        "total_pnl_usd": max(r["total_pnl_usd"] for r in bt_results),
                        "profit_factor": max(r["profit_factor"] for r in bt_results),
                        "max_drawdown_pct": max(r["max_drawdown_pct"] for r in bt_results),
                        "sharpe_ratio": max(r["sharpe_ratio"] for r in bt_results),
                    }

                    def _cell(val, best, fmt):
                        text = fmt.format(val)
                        is_best = abs(val - best) < 1e-9
                        if is_best and len(bt_results) > 1:
                            return f'<span style="color:var(--green);font-weight:bold">{text}</span>'
                        return text

                    rows_html = ""
                    for r in bt_results:
                        rows_html += f"""
<div class="grid-row" style="grid-template-columns:120px repeat(7,1fr);">
  <span class="mono" style="font-weight:bold;color:var(--text-hi)">{r['strategy']}</span>
  <span class="mono">{r['n_trades']}</span>
  <span class="mono">{_cell(r['precision'], best_vals['precision'], '{:.3f}')}</span>
  <span class="mono">{_cell(r['win_rate'], best_vals['win_rate'], '{:.1%}')}</span>
  <span class="mono">{_cell(r['total_pnl_usd'], best_vals['total_pnl_usd'], '${:+,.0f}')}</span>
  <span class="mono">{_cell(r['profit_factor'], best_vals['profit_factor'], '{:.1f}')}</span>
  <span class="mono">{_cell(r['max_drawdown_pct'], best_vals['max_drawdown_pct'], '{:+.1f}%')}</span>
  <span class="mono">{_cell(r['sharpe_ratio'], best_vals['sharpe_ratio'], '{:.1f}')}</span>
</div>"""

                    header_html = """
<div class="grid-header" style="grid-template-columns:120px repeat(7,1fr);">
  <span>Strategy</span><span>Trades</span><span>Precision</span><span>Win Rate</span>
  <span>Total PnL</span><span>Profit Fac</span><span>Max DD</span><span>Sharpe</span>
</div>"""

                    st.markdown(
                        f'<div style="background:var(--panel);border:1px solid var(--border);overflow:hidden">'
                        f'{header_html}{rows_html}</div>',
                        unsafe_allow_html=True,
                    )

                    # Equity curves
                    if BT_CURVES.exists():
                        st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)
                        st.markdown('<div class="sec-label">Equity Curves</div>', unsafe_allow_html=True)
                        try:
                            with open(BT_CURVES) as f:
                                curves = json.load(f)

                            strategy_colors = {
                                "rule_based": "#7b8fa4",
                                "xgb_baseline": "#ffa726",
                                "xgb_tuned": "#00e676",
                                "lgbm_baseline": "#40c4ff",
                                "combined_vote": "#ff1744",
                            }
                            best_strat = max(bt_results, key=lambda r: r["total_pnl_usd"])["strategy"]

                            fig_eq = go.Figure()
                            for name, curve in curves.items():
                                width = 3 if name == best_strat else 1.5
                                fig_eq.add_trace(go.Scatter(
                                    y=curve, mode="lines", name=name,
                                    line=dict(color=strategy_colors.get(name, "#ffa726"), width=width),
                                ))
                            fig_eq.add_hline(
                                y=1000, line_dash="dot", line_color="rgba(123,143,164,0.3)",
                                annotation_text="Starting $1,000", annotation_font_color="#7b8fa4",
                            )
                            dark_plotly_layout(fig_eq, title="Cumulative PnL by Strategy (Test Set)",
                                               xaxis_title="Trade #", yaxis_title="Portfolio Value ($)")
                            fig_eq.update_layout(title_font=dict(color="#7b8fa4", size=12), height=420)
                            st.plotly_chart(fig_eq, width="stretch")
                        except Exception:
                            st.markdown('<div class="nodata">Error loading equity curves.</div>', unsafe_allow_html=True)

                    # Key insights
                    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
                    best = max(bt_results, key=lambda r: r["total_pnl_usd"])
                    st.markdown(f"""
<div class="hud-panel green">
  <div class="panel-title">Key Insights</div>
  <div class="mini-stat"><span class="mini-stat-lbl">Best Strategy</span><span class="mini-stat-val up">{best['strategy']} (+{best['total_pnl_pct']:.1f}%)</span></div>
  <div class="mini-stat"><span class="mini-stat-lbl">Rugs Avoided</span><span class="mini-stat-val">{best['rugs_avoided']:,} / {best['rugs_total']:,} non-pumps</span></div>
  <div class="mini-stat"><span class="mini-stat-lbl">Trade Selectivity</span><span class="mini-stat-val">{best['trade_rate_pct']:.1f}% of tokens flagged</span></div>
  <div class="mini-stat"><span class="mini-stat-lbl">Final Capital</span><span class="mini-stat-val up">${best['final_capital']:,.2f}</span></div>
</div>
<div style="margin-top:8px;padding:10px 12px;background:rgba(255,145,0,0.06);
            border:1px solid rgba(255,145,0,0.2);border-left:2px solid var(--orange);
            color:var(--orange);font-size:12px;font-family:var(--font-mono);">
  \u26a0 Past performance on test set does not guarantee live results. Always paper trade first.
</div>""", unsafe_allow_html=True)

            except Exception as e:
                st.markdown(f'<div class="nodata">Error loading backtest data: {e}</div>', unsafe_allow_html=True)

    # ── Tab: MODELS (NEW) ────────────────────────────────────────────────
    with tab_models:
        st.markdown('<div class="sec-label">Model Comparison</div>', unsafe_allow_html=True)

        # Build comparison table from all model metrics
        comparison_rows = []
        for model_name in MODEL_REGISTRY:
            m = load_model_metrics(model_name)
            if m and "classification" in m:
                cl = m["classification"]
                comparison_rows.append({
                    "name": model_name,
                    "precision": cl.get("precision", 0),
                    "recall": cl.get("recall", 0),
                    "f1": cl.get("f1", 0),
                    "roc_auc": cl.get("roc_auc", 0),
                    "n_samples": cl.get("n_samples", 0),
                    "time": m.get("training_time_seconds", 0),
                })

        if comparison_rows:
            best_p = max(r["precision"] for r in comparison_rows)
            best_r = max(r["recall"] for r in comparison_rows)
            best_f1 = max(r["f1"] for r in comparison_rows)
            best_auc = max(r["roc_auc"] for r in comparison_rows)

            def _mcell(val, best, fmt_str):
                text = fmt_str.format(val)
                if abs(val - best) < 1e-9 and len(comparison_rows) > 1:
                    return f'<span style="color:var(--green);font-weight:bold">{text}</span>'
                return text

            hdr = """<div class="grid-header" style="grid-template-columns:160px repeat(6,1fr);">
  <span>Model</span><span>Precision</span><span>Recall</span><span>F1</span><span>ROC-AUC</span><span>Samples</span><span>Time</span>
</div>"""
            body = ""
            for r in comparison_rows:
                body += f"""<div class="grid-row" style="grid-template-columns:160px repeat(6,1fr);">
  <span class="mono" style="font-weight:bold;color:var(--text-hi)">{r['name']}</span>
  <span class="mono">{_mcell(r['precision'], best_p, '{:.3f}')}</span>
  <span class="mono">{_mcell(r['recall'], best_r, '{:.3f}')}</span>
  <span class="mono">{_mcell(r['f1'], best_f1, '{:.3f}')}</span>
  <span class="mono">{_mcell(r['roc_auc'], best_auc, '{:.3f}')}</span>
  <span class="mono">{r['n_samples']:,}</span>
  <span class="mono">{r['time']:.1f}s</span>
</div>"""
            st.markdown(
                f'<div style="background:var(--panel);border:1px solid var(--border);overflow:hidden">{hdr}{body}</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown('<div class="nodata">No model metrics found.</div>', unsafe_allow_html=True)

        # Dataset stats
        ds_path = PROJECT_ROOT / "ml" / "data" / "dataset_stats.json"
        if ds_path.exists():
            try:
                with open(ds_path) as f:
                    ds = json.load(f)
                total = ds.get("total_samples", 0)
                splits = ds.get("splits", {})
                train_info = splits.get("train", {})
                st.markdown(f"""
<div class="hud-panel blue" style="margin-top:12px;">
  <div class="panel-title">Dataset</div>
  <div class="mini-stat"><span class="mini-stat-lbl">Total Samples</span><span class="mini-stat-val">{total:,}</span></div>
  <div class="mini-stat"><span class="mini-stat-lbl">Train / Val / Test</span><span class="mini-stat-val">{train_info.get('size',0):,} / {splits.get('val',{}).get('size',0):,} / {splits.get('test',{}).get('size',0):,}</span></div>
  <div class="mini-stat"><span class="mini-stat-lbl">Label Balance</span><span class="mini-stat-val">Pump: {train_info.get('label_1',0):,} \u2502 Rug: {train_info.get('label_0',0):,}</span></div>
  <div class="mini-stat"><span class="mini-stat-lbl">Features</span><span class="mini-stat-val">{len(ds.get('feature_columns',[]))} cols</span></div>
</div>""", unsafe_allow_html=True)
            except Exception:
                pass

        # Model detail view
        st.markdown('<div class="sec-label">Model Inspector</div>', unsafe_allow_html=True)
        model_names = list(MODEL_REGISTRY.keys())
        selected_model = st.selectbox("Select model", model_names, key="model_selector", label_visibility="collapsed")

        if selected_model:
            info = MODEL_REGISTRY[selected_model]
            metrics = load_model_metrics(selected_model)

            if metrics and "trading" in metrics:
                tr = metrics["trading"]
                st.markdown(f"""
<div class="hud-panel" style="margin-top:8px;">
  <div class="panel-title">{selected_model} \u2014 Trading Metrics</div>
  <div class="mini-stat"><span class="mini-stat-lbl">Predicted Trades</span><span class="mini-stat-val">{tr.get('predicted_trades',0):,}</span></div>
  <div class="mini-stat"><span class="mini-stat-lbl">True Positives</span><span class="mini-stat-val up">{tr.get('true_positives',0):,}</span></div>
  <div class="mini-stat"><span class="mini-stat-lbl">EV per Trade</span><span class="mini-stat-val">{tr.get('ev_per_trade',0):.3f}</span></div>
  <div class="mini-stat"><span class="mini-stat-lbl">Rugs Avoided</span><span class="mini-stat-val">{tr.get('coins_saved_from_rugs',0):,}</span></div>
</div>""", unsafe_allow_html=True)

            # Display images
            img_cols = st.columns(2)
            img_map = [
                ("confusion_matrix", "Confusion Matrix"),
                ("roc_curve", "ROC Curve"),
                ("feature_importance", "Feature Importance"),
                ("training_curves", "Training Curves"),
            ]
            col_idx = 0
            for key, label in img_map:
                fname = info.get(key)
                if fname:
                    img_path = MODELS_DIR / fname
                    if img_path.exists():
                        with img_cols[col_idx % 2]:
                            st.markdown(f'<div class="sec-label">{label}</div>', unsafe_allow_html=True)
                            st.image(str(img_path))
                        col_idx += 1

    # ── Tab: RL AGENT (FIXED) ────────────────────────────────────────────
    with tab_rl:
        RL_MODEL_PATH = MODELS_DIR / "best_model.zip"
        RL_EVAL_NPZ = MODELS_DIR / "rl_logs" / "evaluations.npz"
        RL_CHECKPOINTS = MODELS_DIR / "checkpoints"
        RL_TRAIN_SCRIPT = PROJECT_ROOT / "ml" / "rl_exit_agent.py"

        st.markdown('<div class="sec-label">Model Status</div>', unsafe_allow_html=True)
        if RL_MODEL_PATH.exists():
            stat = RL_MODEL_PATH.stat()
            size_kb = stat.st_size / 1024
            mod_time = datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc)
            age = datetime.now(timezone.utc) - mod_time
            age_str = f"{age.days}d {age.seconds // 3600}h ago" if age.days else f"{age.seconds // 3600}h {(age.seconds % 3600) // 60}m ago"

            # Check for checkpoints
            ckpts = sorted(RL_CHECKPOINTS.glob("*.zip")) if RL_CHECKPOINTS.exists() else []
            ckpt_info = f"{len(ckpts)} checkpoint(s)" if ckpts else "none"

            st.markdown(f"""
<div class="hud-panel green">
  <div class="health-row"><span class="health-lbl">Status</span><span class="health-val"><span class="dot-green"></span>Trained</span></div>
  <div class="health-row"><span class="health-lbl">Model file</span><span class="health-val mono">{RL_MODEL_PATH.name} ({size_kb:.0f} KB)</span></div>
  <div class="health-row"><span class="health-lbl">Last modified</span><span class="health-val mono">{mod_time.strftime("%Y-%m-%d %H:%M UTC")}</span></div>
  <div class="health-row"><span class="health-lbl">Age</span><span class="health-val mono">{age_str}</span></div>
  <div class="health-row"><span class="health-lbl">Checkpoints</span><span class="health-val mono">{ckpt_info}</span></div>
</div>""", unsafe_allow_html=True)
        else:
            st.markdown("""
<div class="hud-panel">
  <div class="health-row"><span class="health-lbl">Status</span><span class="health-val"><span class="dot-yellow"></span>Not trained</span></div>
  <div class="health-row"><span class="health-lbl">Fallback</span><span class="health-val mono">rule_based exits active</span></div>
</div>""", unsafe_allow_html=True)

        # Training trigger
        st.markdown('<div class="sec-label">Training</div>', unsafe_allow_html=True)
        col_btn, col_info = st.columns([1, 2])
        with col_btn:
            if st.button("Start RL Training", key="rl_train_btn"):
                if RL_TRAIN_SCRIPT.exists():
                    try:
                        log_path = PROJECT_ROOT / "ml" / "rl_train.log"
                        log_fh = open(str(log_path), "w")
                        try:
                            proc = subprocess.Popen(
                                ["python", str(RL_TRAIN_SCRIPT), "--train"],
                                cwd=str(PROJECT_ROOT), stdout=log_fh, stderr=subprocess.STDOUT,
                            )
                        finally:
                            log_fh.close()  # Popen has duplicated the fd at the OS level
                        st.success(f"Training started (PID {proc.pid}). Check ml/rl_train.log.")
                    except Exception as exc:
                        st.error(f"Failed to start training: {exc}")
                else:
                    st.warning("ml/rl_exit_agent.py not found.")
        with col_info:
            st.markdown("""
<div style="font-size:11px;color:var(--text-lo);padding-top:6px;">
  <strong>Model:</strong> <code>ml/models/best_model.zip</code><br>
  <strong>Manual:</strong> <code>python ml/rl_exit_agent.py --train</code><br>
  <strong>Full pipeline:</strong> <code>scripts/train_all.sh</code>
</div>""", unsafe_allow_html=True)

        # Evaluation reward curve from evaluations.npz
        if HAS_NUMPY and RL_EVAL_NPZ.exists():
            st.markdown('<div class="sec-label">Training Reward Curve</div>', unsafe_allow_html=True)
            try:
                data = np.load(str(RL_EVAL_NPZ))
                timesteps = data["timesteps"]
                results = data["results"]
                mean_rewards = results.mean(axis=1)

                fig_rl = go.Figure()
                fig_rl.add_trace(go.Scatter(
                    x=timesteps.tolist(), y=mean_rewards.tolist(),
                    mode="lines+markers",
                    line=dict(color="#ffa726", width=2),
                    marker=dict(size=5, color="#ffa726"),
                    fill="tozeroy", fillcolor="rgba(255,167,38,0.06)",
                    name="Mean Reward",
                ))
                # Add std deviation band
                std_rewards = results.std(axis=1)
                upper = (mean_rewards + std_rewards).tolist()
                lower = (mean_rewards - std_rewards).tolist()
                fig_rl.add_trace(go.Scatter(
                    x=timesteps.tolist() + timesteps[::-1].tolist(),
                    y=upper + lower[::-1],
                    fill="toself", fillcolor="rgba(255,167,38,0.06)",
                    line=dict(width=0), showlegend=False, name="Std Dev",
                ))
                dark_plotly_layout(fig_rl, title="RL Eval Reward Over Training",
                                   xaxis_title="Timestep", yaxis_title="Mean Reward")
                fig_rl.update_layout(title_font=dict(color="#7b8fa4", size=12), showlegend=False, height=380)
                st.plotly_chart(fig_rl, width="stretch")

                # Show stats
                st.markdown(f"""
<div class="hud-panel">
  <div class="panel-title">RL Training Stats</div>
  <div class="mini-stat"><span class="mini-stat-lbl">Eval Points</span><span class="mini-stat-val">{len(timesteps)}</span></div>
  <div class="mini-stat"><span class="mini-stat-lbl">Total Timesteps</span><span class="mini-stat-val">{int(timesteps[-1]):,}</span></div>
  <div class="mini-stat"><span class="mini-stat-lbl">Peak Mean Reward</span><span class="mini-stat-val up">{mean_rewards.max():.2f}</span></div>
  <div class="mini-stat"><span class="mini-stat-lbl">Final Mean Reward</span><span class="mini-stat-val">{mean_rewards[-1]:.2f}</span></div>
  <div class="mini-stat"><span class="mini-stat-lbl">Episodes per Eval</span><span class="mini-stat-val">{results.shape[1]}</span></div>
</div>""", unsafe_allow_html=True)

            except Exception as e:
                st.markdown(f'<div class="nodata">Error loading evaluation data: {e}</div>', unsafe_allow_html=True)
        elif not RL_EVAL_NPZ.exists():
            st.markdown('<div class="nodata">No training logs yet. Train the model to see reward curves.</div>', unsafe_allow_html=True)

# ─── Footer ───────────────────────────────────────────────────────────────────
st.markdown("""
<div class="soldar-footer">
  SOLDAR v2.0 \u2502 Solana Meme-Coin Radar System \u2502 Data refreshes every 90s
</div>
""", unsafe_allow_html=True)
