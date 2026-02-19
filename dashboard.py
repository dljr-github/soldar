#!/usr/bin/env python3
"""Soldar — professional dark trading terminal dashboard."""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import plotly.graph_objects as go
import streamlit as st
from streamlit_autorefresh import st_autorefresh

# ─── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="SOLDAR",
    layout="wide",
    page_icon="🛰",
    initial_sidebar_state="collapsed",
)

# ─── Auto-refresh every 90 seconds ───────────────────────────────────────────
st_autorefresh(interval=90_000, key="data_refresh")

# ─── CSS ─────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* === Base === */
html, body, .stApp { background-color: #0d1117 !important; color: #e6edf3; }
[data-testid="stHeader"] { background: #0d1117; border-bottom: 1px solid #21262d; }
[data-testid="stSidebar"] { display: none !important; }
.main .block-container { padding: 0.5rem 1.25rem 1rem 1.25rem; max-width: 100%; }
.stMainBlockContainer { padding-top: 0.5rem !important; }

/* === Scrollbar === */
::-webkit-scrollbar { width: 5px; height: 5px; }
::-webkit-scrollbar-track { background: #0d1117; }
::-webkit-scrollbar-thumb { background: #30363d; border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: #484f58; }

/* === Header bar === */
.soldar-header {
    display: flex; align-items: center; gap: 18px;
    padding: 8px 0 10px 0; border-bottom: 1px solid #21262d;
    margin-bottom: 14px; flex-wrap: wrap;
}
.soldar-title {
    font-family: 'Courier New', monospace; font-size: 17px;
    font-weight: bold; color: #58a6ff; letter-spacing: 3px;
}
.live-indicator { display: flex; align-items: center; gap: 5px; font-size: 12px; color: #3fb950; }
.live-dot {
    width: 7px; height: 7px; background: #3fb950; border-radius: 50%;
    display: inline-block; animation: livepulse 2s ease-in-out infinite;
}
@keyframes livepulse {
    0%, 100% { opacity: 1; box-shadow: 0 0 5px #3fb950; }
    50% { opacity: 0.35; box-shadow: none; }
}
.hstat { display: flex; flex-direction: column; align-items: center; gap: 0; }
.hstat-val { font-family: 'Courier New', monospace; font-size: 14px; font-weight: bold; color: #e6edf3; }
.hstat-lbl { font-size: 10px; color: #7d8590; text-transform: uppercase; letter-spacing: 0.5px; }
.hstat-sep { color: #30363d; font-size: 16px; margin: 0 2px; }

/* === Filter panel === */
.fpanel {
    background: #161b22; border: 1px solid #21262d;
    border-radius: 6px; padding: 12px; margin-bottom: 10px;
}
.fpanel-title {
    font-size: 10px; color: #7d8590; text-transform: uppercase;
    letter-spacing: 1px; margin-bottom: 10px;
    font-family: 'Courier New', monospace;
}
.stSlider > label, .stRadio > label { color: #7d8590 !important; font-size: 11px !important; }
.stRadio div[role="radiogroup"] label { color: #b0b7c0 !important; font-size: 12px !important; }

/* === Mini stat card (sidebar) === */
.mini-stat {
    background: #161b22; border: 1px solid #21262d;
    border-radius: 5px; padding: 8px 10px; margin-bottom: 6px;
    display: flex; justify-content: space-between; align-items: center;
}
.mini-stat-lbl { font-size: 11px; color: #7d8590; }
.mini-stat-val { font-family: 'Courier New', monospace; font-size: 13px; font-weight: bold; color: #e6edf3; }
.mini-stat-val.up { color: #3fb950; }
.mini-stat-val.down { color: #f85149; }

/* === Health indicator === */
.health-row {
    display: flex; justify-content: space-between; align-items: center;
    padding: 3px 0; border-bottom: 1px solid #21262d; font-size: 11px;
}
.health-row:last-child { border-bottom: none; }
.health-lbl { color: #7d8590; }
.health-val { font-family: 'Courier New', monospace; color: #e6edf3; }
.dot-green { display: inline-block; width: 6px; height: 6px; background: #3fb950; border-radius: 50%; margin-right: 4px; }
.dot-yellow { display: inline-block; width: 6px; height: 6px; background: #ffd700; border-radius: 50%; margin-right: 4px; }
.dot-red { display: inline-block; width: 6px; height: 6px; background: #f85149; border-radius: 50%; margin-right: 4px; }

/* === Tabs === */
.stTabs [data-baseweb="tab-list"] { background: transparent; border-bottom: 1px solid #21262d; gap: 0; }
.stTabs [data-baseweb="tab"] {
    background: transparent !important; color: #7d8590 !important;
    border: none !important; padding: 7px 14px !important;
    font-size: 12px !important; font-family: 'Courier New', monospace !important;
    letter-spacing: 0.5px;
}
.stTabs [aria-selected="true"] { color: #e6edf3 !important; }
.stTabs [data-baseweb="tab-highlight"] { background: #58a6ff !important; height: 2px !important; }
.stTabs [data-baseweb="tab-border"] { display: none; }

/* === Exit signal banners === */
.exit-banner {
    background: rgba(248,81,73,0.07); border: 1px solid rgba(248,81,73,0.35);
    border-left: 3px solid #f85149; border-radius: 5px;
    padding: 8px 12px; margin-bottom: 7px;
}
.exit-title { color: #f85149; font-weight: bold; font-size: 13px; }
.exit-detail { color: #7d8590; font-size: 11px; margin-top: 2px; font-family: 'Courier New', monospace; }

/* === Candidate cards === */
.ccard {
    background: #161b22; border: 1px solid #21262d;
    border-radius: 6px; padding: 10px 13px;
    margin-bottom: 7px; line-height: 1.6;
}
.ccard-hot   { border-left: 3px solid #ff4444; }
.ccard-warm  { border-left: 3px solid #ff8c00; }
.ccard-watch { border-left: 3px solid #ffd700; }
.ccard-low   { border-left: 3px solid #30363d; }

.crow1 { display: flex; align-items: center; gap: 9px; margin-bottom: 4px; }
.crow2, .crow3 { font-size: 12px; color: #7d8590; margin-bottom: 3px; }
.crow4 { display: flex; gap: 8px; align-items: center; flex-wrap: wrap; margin-top: 4px; }

.csym { font-family: 'Courier New', monospace; font-size: 15px; font-weight: bold; color: #e6edf3; }
.cname { color: #7d8590; font-size: 11px; }
.cscore { font-family: 'Courier New', monospace; font-size: 13px; font-weight: bold; margin-left: auto; }
.cscore-hot   { color: #ff4444; }
.cscore-warm  { color: #ff8c00; }
.cscore-watch { color: #ffd700; }
.cscore-low   { color: #7d8590; }

.badge { display: inline-block; padding: 1px 7px; border-radius: 10px; font-size: 10px; font-weight: bold; font-family: 'Courier New', monospace; }
.badge-hot   { background: rgba(255,68,68,0.12);  color: #ff4444; border: 1px solid rgba(255,68,68,0.35); }
.badge-warm  { background: rgba(255,140,0,0.12);  color: #ff8c00; border: 1px solid rgba(255,140,0,0.35); }
.badge-watch { background: rgba(255,215,0,0.12);  color: #ffd700; border: 1px solid rgba(255,215,0,0.35); }
.badge-low   { background: rgba(125,133,144,0.1); color: #7d8590; border: 1px solid rgba(125,133,144,0.25); }

.mono { font-family: 'Courier New', monospace; }
.up   { color: #3fb950; font-family: 'Courier New', monospace; }
.dn   { color: #f85149; font-family: 'Courier New', monospace; }
.flat { color: #7d8590; font-family: 'Courier New', monospace; }

.pill { display: inline-block; padding: 1px 5px; border-radius: 3px; font-size: 10px; font-weight: bold; }
.pill-pass { background: rgba(63,185,80,0.12);  color: #3fb950; border: 1px solid rgba(63,185,80,0.35); }
.pill-warn { background: rgba(255,165,0,0.12);  color: #ffa500; border: 1px solid rgba(255,165,0,0.35); }
.pill-fail { background: rgba(248,81,73,0.12);  color: #f85149; border: 1px solid rgba(248,81,73,0.35); }
.pill-na   { background: rgba(125,133,144,0.1); color: #7d8590; border: 1px solid rgba(125,133,144,0.25); }

.ca-addr { font-family: 'Courier New', monospace; font-size: 10px; color: #484f58; margin-left: auto; }

.btn-copy {
    background: rgba(88,166,255,0.08); color: #58a6ff;
    border: 1px solid rgba(88,166,255,0.25); border-radius: 3px;
    padding: 1px 7px; font-size: 10px; cursor: pointer;
    font-family: 'Courier New', monospace;
}
.btn-copy:hover { background: rgba(88,166,255,0.18); }
.dex-link { color: #58a6ff; text-decoration: none; font-size: 10px; }
.dex-link:hover { text-decoration: underline; }

/* === Stats metric cards === */
.smetric {
    background: #161b22; border: 1px solid #21262d; border-radius: 6px;
    padding: 14px; text-align: center;
}
.smetric-val { font-family: 'Courier New', monospace; font-size: 26px; font-weight: bold; color: #e6edf3; }
.smetric-lbl { font-size: 10px; color: #7d8590; text-transform: uppercase; letter-spacing: 0.5px; margin-top: 3px; }

/* === Rejected table === */
.rej-row {
    display: grid; grid-template-columns: 80px 90px 1fr 130px;
    gap: 10px; padding: 6px 10px; border-bottom: 1px solid #1c2128;
    font-size: 12px; align-items: center;
}
.rej-row.header { color: #7d8590; font-size: 10px; text-transform: uppercase; letter-spacing: 0.5px; background: #0d1117; border-radius: 4px 4px 0 0; }
.rej-row.hard { background: rgba(248,81,73,0.05); }
.rej-sym { font-family: 'Courier New', monospace; font-weight: bold; color: #e6edf3; }
.rej-addr { font-family: 'Courier New', monospace; color: #484f58; font-size: 10px; }
.rej-reason { color: #7d8590; }
.rej-time { color: #484f58; font-size: 10px; font-family: 'Courier New', monospace; }

/* === Positions table === */
.pos-summary {
    display: flex; gap: 10px; flex-wrap: wrap; margin-bottom: 12px;
}
.pos-summary-card {
    background: #161b22; border: 1px solid #21262d; border-radius: 6px;
    padding: 10px 16px; flex: 1; min-width: 120px; text-align: center;
}
.pos-summary-val { font-family: 'Courier New', monospace; font-size: 20px; font-weight: bold; color: #e6edf3; }
.pos-summary-lbl { font-size: 10px; color: #7d8590; text-transform: uppercase; letter-spacing: 0.5px; }

/* === No data === */
.nodata { color: #7d8590; font-size: 13px; text-align: center; padding: 20px; font-family: 'Courier New', monospace; }

/* === Waiting screen === */
.waiting-wrap { text-align: center; padding: 100px 20px; }
.waiting-title { font-family: 'Courier New', monospace; font-size: 36px; font-weight: bold; color: #58a6ff; letter-spacing: 4px; margin-bottom: 10px; }
.waiting-sub { color: #7d8590; font-size: 13px; }
.waiting-blink { animation: livepulse 1.5s ease-in-out infinite; }

/* === Metric component override === */
[data-testid="stMetric"] { background: #161b22 !important; border: 1px solid #21262d !important; border-radius: 6px !important; }
[data-testid="stMetricLabel"] p { color: #7d8590 !important; font-size: 11px !important; }
[data-testid="stMetricValue"] { font-family: 'Courier New', monospace !important; }

/* === DataFrame === */
.stDataFrame { border: 1px solid #21262d; border-radius: 6px; overflow: hidden; }
[data-testid="stDataFrameResizable"] th { background: #0d1117 !important; color: #7d8590 !important; font-size: 11px !important; }

/* === Divider === */
hr { border-color: #21262d; margin: 10px 0; }

/* === Section label === */
.sec-label { font-size: 10px; color: #7d8590; text-transform: uppercase; letter-spacing: 1px; margin: 12px 0 6px 0; font-family: 'Courier New', monospace; }

/* === Count badge === */
.count-badge { background: #21262d; color: #7d8590; border-radius: 10px; padding: 1px 6px; font-size: 10px; font-family: 'Courier New', monospace; }
</style>
""", unsafe_allow_html=True)

# ─── Constants ────────────────────────────────────────────────────────────────
STATE_FILE = Path(__file__).parent / "data" / "state.json"
TRADES_DB  = Path(__file__).parent / "data" / "trades.db"


# ─── Data loading ─────────────────────────────────────────────────────────────
def load_state() -> dict | None:
    if not STATE_FILE.exists():
        return None
    try:
        with open(STATE_FILE) as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return None


def load_trades_summary() -> dict:
    """Return open_count, daily_pnl, win_rate from trades DB."""
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
        total_pnl = sum(r["pnl_usd"] for r in all_sells)
        win_rate = None
        if all_sells:
            wins = sum(1 for r in all_sells if r["pnl_usd"] > 0)
            win_rate = wins / len(all_sells) * 100
        conn.close()
        return {"open_count": open_count, "daily_pnl": daily_pnl, "win_rate": win_rate, "total_pnl": total_pnl}
    except Exception:
        return defaults


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
        return f'<span class="pill pill-pass">✓ {label}</span>'
    return f'<span class="pill pill-fail">✗ {label}</span>'


def verdict_pill(v) -> str:
    if v == "PASS":
        return '<span class="pill pill-pass">✓ PASS</span>'
    if v == "WARN":
        return '<span class="pill pill-warn">⚠ WARN</span>'
    if v == "FAIL":
        return '<span class="pill pill-fail">✗ FAIL</span>'
    return '<span class="pill pill-na">— N/A</span>'


def level_attrs(score: int) -> tuple[str, str, str, str]:
    """Return (level_str, badge_cls, card_cls, score_cls)."""
    if score >= 80:
        return "HOT",   "badge-hot",   "ccard-hot",   "cscore-hot"
    if score >= 65:
        return "WARM",  "badge-warm",  "ccard-warm",  "cscore-warm"
    if score >= 50:
        return "WATCH", "badge-watch", "ccard-watch", "cscore-watch"
    return "LOW", "badge-low", "ccard-low", "cscore-low"


def build_candidate_card(c: dict) -> str:
    score   = c.get("score", 0)
    symbol  = c.get("symbol", "?")
    name    = c.get("name", "")
    address = c.get("address", "")
    age     = c.get("age_minutes")
    liq     = c.get("liquidity_usd") or 0
    mcap    = c.get("mcap_usd") or 0
    vol_liq = c.get("vol_liq_ratio") or 0
    p5m     = c.get("price_change_5m") or 0
    p1h     = c.get("price_change_1h") or 0
    buys    = c.get("buys_5m") or 0
    sells   = c.get("sells_5m") or 0
    verdict = c.get("legit_verdict")
    mint_ok = c.get("legit_mint_revoked")
    frz_ok  = c.get("legit_freeze_revoked")
    lp_lock = c.get("legit_lp_locked_pct")
    top1    = c.get("legit_top1_pct")
    rc      = c.get("legit_rc_score")
    dex_url = c.get("url") or f"https://dexscreener.com/solana/{address}"

    lvl, badge_cls, card_cls, score_cls = level_attrs(score)
    age_str = f"{age:.0f}m" if age is not None else "?"
    addr_short = address[:8] + "…" + address[-4:] if len(address) > 12 else address

    # LP pill
    if lp_lock is not None:
        lp_cls = "pill-pass" if lp_lock >= 90 else ("pill-warn" if lp_lock >= 50 else "pill-fail")
        lp_pill = f'<span class="pill {lp_cls}">LP {lp_lock:.0f}%</span>'
    else:
        lp_pill = '<span class="pill pill-na">LP ?</span>'

    # Top holder pill
    if top1 is not None:
        t_cls = "pill-pass" if top1 <= 10 else ("pill-warn" if top1 <= 20 else "pill-fail")
        top1_pill = f'<span class="pill {t_cls}">Top {top1:.1f}%</span>'
    else:
        top1_pill = ""

    rc_str = f'<span class="mono" style="font-size:10px;color:#7d8590;">RC:{rc}</span>' if rc is not None else ""

    # JS clipboard copy (works on localhost secure context)
    js_copy = (
        f"navigator.clipboard.writeText('{address}')"
        ".then(function(){var b=this;b.textContent='✓ Copied';"
        "setTimeout(function(){b.textContent='📋 Copy CA'},2000)}).bind(this)()"
    )
    copy_btn = f'<button class="btn-copy" onclick="{js_copy}">📋 Copy CA</button>'

    return f"""
<div class="ccard {card_cls}">
  <div class="crow1">
    <span class="badge {badge_cls}">{lvl}</span>
    <span class="csym">{symbol}</span>
    <span class="cname">{name}</span>
    <span class="{score_cls} cscore">Score: {score}/100</span>
  </div>
  <div class="crow2">
    Age: <span class="mono">{age_str}</span> &nbsp;│&nbsp;
    Liq: <span class="mono">{fmt_usd(liq)}</span> &nbsp;│&nbsp;
    MCap: <span class="mono">{fmt_usd(mcap)}</span> &nbsp;│&nbsp;
    Vol/Liq: <span class="mono">{vol_liq:.1f}x</span>
  </div>
  <div class="crow3">
    5m: {price_span(p5m)} &nbsp;│&nbsp;
    1h: {price_span(p1h)} &nbsp;│&nbsp;
    Buys: <span style="color:#3fb950;font-family:monospace">{buys}</span>
    &nbsp;Sells: <span style="color:#f85149;font-family:monospace">{sells}</span>
  </div>
  <div class="crow4">
    {verdict_pill(verdict)}
    {legit_pill(mint_ok, "Mint")}
    {legit_pill(frz_ok, "Frz")}
    {lp_pill}
    {top1_pill}
    {rc_str}
    {copy_btn}
    <a href="{dex_url}" target="_blank" class="dex-link">DexScreener ↗</a>
    <span class="ca-addr">{addr_short}</span>
  </div>
</div>"""


def build_exit_banner(ea: dict) -> str:
    urgency = ea.get("urgency", "LOW")
    symbol  = ea.get("symbol", "?")
    reason  = ea.get("exit_reason", "")
    p5m     = ea.get("current_price_change_5m") or 0
    p1h     = ea.get("current_price_change_1h") or 0
    urg_icon = {"HIGH": "🔴", "MEDIUM": "🟠", "LOW": "🟡"}.get(urgency, "⚪")
    return f"""
<div class="exit-banner">
  <div class="exit-title">🚨 EXIT SIGNAL: {symbol} — {reason} &nbsp;|&nbsp; {urg_icon} {urgency} urgency</div>
  <div class="exit-detail">5m: {fmt_pct(p5m)} &nbsp;│&nbsp; 1h: {fmt_pct(p1h)}</div>
</div>"""


def dark_plotly_layout(fig: go.Figure, **kwargs) -> go.Figure:
    fig.update_layout(
        paper_bgcolor="#0d1117",
        plot_bgcolor="#161b22",
        font=dict(color="#7d8590", family="Courier New, monospace", size=11),
        margin=dict(l=40, r=20, t=40, b=30),
        xaxis=dict(gridcolor="#21262d", zerolinecolor="#21262d", tickfont=dict(size=10)),
        yaxis=dict(gridcolor="#21262d", zerolinecolor="#21262d", tickfont=dict(size=10)),
        legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor="#21262d"),
        **kwargs,
    )
    return fig


# ─── Load state ───────────────────────────────────────────────────────────────
state = load_state()

# ─── Waiting screen ───────────────────────────────────────────────────────────
if state is None:
    st.markdown("""
<div class="waiting-wrap">
  <div class="waiting-title waiting-blink">🛰 SOLDAR</div>
  <div class="waiting-sub">Waiting for screener data…</div>
  <div class="waiting-sub" style="margin-top:8px;font-size:11px;">
    <code>data/state.json</code> not found — screener will create it on next cycle.<br>
    Auto-refreshes every 90 seconds.
  </div>
</div>""", unsafe_allow_html=True)
    st.stop()

# ─── Extract state ────────────────────────────────────────────────────────────
candidates:    list[dict] = state.get("candidates", [])
hard_rejected: list[dict] = state.get("hard_rejected", [])
cycle_history: list[dict] = state.get("cycle_history", [])
exit_alerts:   list[dict] = state.get("exit_alerts", [])

last_updated_str = state.get("last_updated", "")
try:
    updated_dt  = datetime.fromisoformat(last_updated_str)
    seconds_ago = int((datetime.now(timezone.utc) - updated_dt).total_seconds())
except (ValueError, TypeError):
    seconds_ago = 0

cycles_run    = state.get("cycle_count", 0)
scanned_count = state.get("candidates_scanned", 0)
passed_count  = state.get("passed_filters", len(candidates))

candidates.sort(key=lambda c: c.get("score", 0), reverse=True)
top_score = candidates[0]["score"] if candidates else 0

hot_count  = sum(1 for c in candidates if c.get("score", 0) >= 80)
warm_count = sum(1 for c in candidates if 65 <= c.get("score", 0) < 80)

if seconds_ago < 120:
    age_text = f"{seconds_ago}s ago"
else:
    age_text = f"{seconds_ago // 60}m ago"

api_ok = seconds_ago < 300

# ─── Header bar ───────────────────────────────────────────────────────────────
status_dot = "dot-green" if api_ok else "dot-red"
st.markdown(f"""
<div class="soldar-header">
  <span class="soldar-title">🛰 SOLDAR</span>
  <div class="live-indicator">
    <span class="live-dot"></span>LIVE
  </div>
  <span class="hstat-sep">│</span>
  <div class="hstat"><span class="hstat-val">{age_text}</span><span class="hstat-lbl">Last Scan</span></div>
  <span class="hstat-sep">│</span>
  <div class="hstat"><span class="hstat-val mono">{scanned_count}</span><span class="hstat-lbl">Scanned</span></div>
  <span class="hstat-sep">│</span>
  <div class="hstat"><span class="hstat-val mono">{passed_count}</span><span class="hstat-lbl">Passed</span></div>
  <span class="hstat-sep">│</span>
  <div class="hstat"><span class="hstat-val mono" style="color:#ff4444">{hot_count}</span><span class="hstat-lbl">HOT</span></div>
  <span class="hstat-sep">│</span>
  <div class="hstat"><span class="hstat-val mono" style="color:#ff8c00">{warm_count}</span><span class="hstat-lbl">WARM</span></div>
  <span class="hstat-sep">│</span>
  <div class="hstat"><span class="hstat-val mono">{top_score}</span><span class="hstat-lbl">Top Score</span></div>
  <span class="hstat-sep">│</span>
  <div class="hstat"><span class="hstat-val mono">{cycles_run}</span><span class="hstat-lbl">Cycles</span></div>
</div>
""", unsafe_allow_html=True)

# ─── Main layout: left col (filters) + right col (tabs) ──────────────────────
left_col, main_col = st.columns([1, 3.2], gap="medium")

# ══════════════════════════════════════════════════════════════════════════════
# LEFT COLUMN — filters + stats
# ══════════════════════════════════════════════════════════════════════════════
with left_col:
    st.markdown('<div class="fpanel"><div class="fpanel-title">Filters</div>', unsafe_allow_html=True)
    min_score = st.slider("Min score", 0, 100, 0, key="min_score")
    max_age   = st.slider("Max age (min)", 0, 360, 360, key="max_age")
    legit_filter = st.radio(
        "Legitimacy",
        ["All", "PASS only", "PASS + WARN"],
        index=0,
        key="legit_filter",
    )
    st.markdown('</div>', unsafe_allow_html=True)

    # Quick stats
    trades = load_trades_summary()
    daily_pnl = trades["daily_pnl"]
    pnl_cls = "up" if daily_pnl > 0 else ("dn" if daily_pnl < 0 else "")
    wr_str = f"{trades['win_rate']:.0f}%" if trades["win_rate"] is not None else "—"

    st.markdown(f"""
<div class="fpanel">
  <div class="fpanel-title">Quick Stats</div>
  <div class="mini-stat">
    <span class="mini-stat-lbl">HOT Alerts</span>
    <span class="mini-stat-val" style="color:#ff4444">{hot_count}</span>
  </div>
  <div class="mini-stat">
    <span class="mini-stat-lbl">Open Trades</span>
    <span class="mini-stat-val">{trades['open_count']}</span>
  </div>
  <div class="mini-stat">
    <span class="mini-stat-lbl">Daily PnL</span>
    <span class="mini-stat-val {pnl_cls}">${daily_pnl:+.2f}</span>
  </div>
  <div class="mini-stat">
    <span class="mini-stat-lbl">Win Rate</span>
    <span class="mini-stat-val">{wr_str}</span>
  </div>
</div>
""", unsafe_allow_html=True)

    # Screener health
    api_indicator = f'<span class="{status_dot}"></span>LIVE' if api_ok else '<span class="dot-red"></span>STALE'
    exit_count = len(exit_alerts)

    st.markdown(f"""
<div class="fpanel">
  <div class="fpanel-title">Screener Health</div>
  <div class="health-row">
    <span class="health-lbl">API Status</span>
    <span class="health-val">{api_indicator}</span>
  </div>
  <div class="health-row">
    <span class="health-lbl">Last scan</span>
    <span class="health-val mono">{age_text}</span>
  </div>
  <div class="health-row">
    <span class="health-lbl">Candidates</span>
    <span class="health-val mono">{len(candidates)}</span>
  </div>
  <div class="health-row">
    <span class="health-lbl">Rejected</span>
    <span class="health-val mono">{len(hard_rejected)}</span>
  </div>
  <div class="health-row">
    <span class="health-lbl">Exit alerts</span>
    <span class="health-val mono" style="color:{'#f85149' if exit_count else '#7d8590'}">{exit_count}</span>
  </div>
</div>
""", unsafe_allow_html=True)

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

# ══════════════════════════════════════════════════════════════════════════════
# RIGHT COLUMN — tabs
# ══════════════════════════════════════════════════════════════════════════════
with main_col:
    rej_label = f"🚫 REJECTED ({len(hard_rejected)})" if hard_rejected else "🚫 REJECTED"
    ea_label  = f"🔥 LIVE ({len(filtered)})"
    tab_live, tab_stats, tab_rejected, tab_positions = st.tabs(
        [ea_label, "📊 STATS", rej_label, "💰 POSITIONS"]
    )

    # ── Tab: LIVE ─────────────────────────────────────────────────────────────
    with tab_live:
        # Exit signal banners
        if exit_alerts:
            for ea in exit_alerts:
                st.markdown(build_exit_banner(ea), unsafe_allow_html=True)
            st.markdown("<hr>", unsafe_allow_html=True)

        if not filtered:
            st.markdown('<div class="nodata">No candidates match current filters.</div>', unsafe_allow_html=True)
        else:
            st.markdown(
                f'<div class="sec-label">Showing {len(filtered)} candidate(s)</div>',
                unsafe_allow_html=True,
            )
            for c in filtered:
                st.markdown(build_candidate_card(c), unsafe_allow_html=True)

    # ── Tab: STATS ────────────────────────────────────────────────────────────
    with tab_stats:
        scores = [c.get("score", 0) for c in candidates]
        avg_score  = sum(scores) / len(scores) if scores else 0
        watch_plus = sum(1 for s in scores if s >= 50)

        # Today's HOT count from cycle history
        today_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        hot_today = hot_count  # current scan

        # Win rate
        wr_display = f"{trades['win_rate']:.1f}%" if trades["win_rate"] is not None else "—"

        # Metric row
        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.markdown(f'<div class="smetric"><div class="smetric-val">{cycles_run}</div><div class="smetric-lbl">Cycles Run</div></div>', unsafe_allow_html=True)
        with m2:
            st.markdown(f'<div class="smetric"><div class="smetric-val">{scanned_count}</div><div class="smetric-lbl">Coins Scanned</div></div>', unsafe_allow_html=True)
        with m3:
            st.markdown(f'<div class="smetric"><div class="smetric-val" style="color:#ff4444">{hot_today}</div><div class="smetric-lbl">HOT Alerts Now</div></div>', unsafe_allow_html=True)
        with m4:
            st.markdown(f'<div class="smetric"><div class="smetric-val">{wr_display}</div><div class="smetric-lbl">Paper Win Rate</div></div>', unsafe_allow_html=True)

        st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)

        chart_left, chart_right = st.columns(2)

        with chart_left:
            if cycle_history:
                recent = cycle_history[-50:]
                times  = [h.get("timestamp", "")[-8:] for h in recent]  # HH:MM:SS
                tscores = [h.get("top_score", 0) for h in recent]

                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=list(range(len(tscores))), y=tscores,
                    mode="lines+markers",
                    line=dict(color="#58a6ff", width=2),
                    marker=dict(size=4, color="#58a6ff"),
                    fill="tozeroy",
                    fillcolor="rgba(88,166,255,0.08)",
                    name="Top Score",
                ))
                fig.add_hline(y=80, line_dash="dot", line_color="rgba(255,68,68,0.4)",
                              annotation_text="HOT", annotation_font_color="#ff4444")
                fig.add_hline(y=65, line_dash="dot", line_color="rgba(255,140,0,0.4)",
                              annotation_text="WARM", annotation_font_color="#ff8c00")
                dark_plotly_layout(fig, title="Top Score / Cycle (last 50)",
                                   yaxis_range=[0, 100])
                fig.update_layout(title_font=dict(color="#7d8590", size=12), showlegend=False)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.markdown('<div class="nodata">No cycle history yet.</div>', unsafe_allow_html=True)

        with chart_right:
            if scores:
                bins  = list(range(0, 110, 10))
                labels = [f"{b}–{b+9}" for b in bins[:-1]]
                counts = [0] * (len(bins) - 1)
                for s in scores:
                    idx = min(int(s) // 10, len(counts) - 1)
                    counts[idx] += 1

                colors = []
                for i, b in enumerate(bins[:-1]):
                    if b >= 80:   colors.append("rgba(255,68,68,0.8)")
                    elif b >= 60: colors.append("rgba(255,140,0,0.8)")
                    elif b >= 50: colors.append("rgba(255,215,0,0.8)")
                    else:         colors.append("rgba(48,54,61,0.8)")

                fig2 = go.Figure(go.Bar(
                    x=labels, y=counts,
                    marker_color=colors,
                    marker_line=dict(color="#0d1117", width=1),
                ))
                dark_plotly_layout(fig2, title="Score Distribution")
                fig2.update_layout(title_font=dict(color="#7d8590", size=12), showlegend=False)
                st.plotly_chart(fig2, use_container_width=True)
            else:
                st.markdown('<div class="nodata">No candidates to chart.</div>', unsafe_allow_html=True)

        # Top candidates table
        if candidates:
            st.markdown('<div class="sec-label">Top Candidates This Cycle</div>', unsafe_allow_html=True)
            table_data = []
            for c in candidates[:20]:
                lvl_str, _, _, _ = level_attrs(c.get("score", 0))
                table_data.append({
                    "Symbol":   c.get("symbol", "?"),
                    "Score":    c.get("score", 0),
                    "Level":    lvl_str,
                    "Age(m)":   f"{c.get('age_minutes', 0):.0f}",
                    "Liq":      fmt_usd(c.get("liquidity_usd")),
                    "MCap":     fmt_usd(c.get("mcap_usd")),
                    "5m%":      fmt_pct(c.get("price_change_5m")),
                    "1h%":      fmt_pct(c.get("price_change_1h")),
                    "Vol/Liq":  f"{c.get('vol_liq_ratio', 0):.1f}x",
                    "Legit":    c.get("legit_verdict") or "—",
                })
            st.dataframe(table_data, use_container_width=True, hide_index=True)

    # ── Tab: REJECTED ─────────────────────────────────────────────────────────
    with tab_rejected:
        if not hard_rejected:
            st.markdown('<div class="nodata">No hard rejections this cycle.</div>', unsafe_allow_html=True)
        else:
            rows_html = '<div class="rej-row header"><span>Symbol</span><span>Address</span><span>Rejection Reason</span><span>Time</span></div>'
            for r in hard_rejected:
                sym   = r.get("symbol", "?")
                addr  = r.get("address", "?")
                addr_s = addr[:8] + "…" + addr[-4:] if len(addr) > 12 else addr
                reasons = "; ".join(r.get("reasons", []))
                ts    = r.get("timestamp", "")
                ts_s  = ts[11:16] if len(ts) >= 16 else ts
                rows_html += f"""
<div class="rej-row hard">
  <span class="rej-sym">{sym}</span>
  <span class="rej-addr">{addr_s}</span>
  <span class="rej-reason">{reasons}</span>
  <span class="rej-time">{ts_s}</span>
</div>"""
            st.markdown(
                f'<div style="background:#161b22;border:1px solid #21262d;border-radius:6px;overflow:hidden">{rows_html}</div>',
                unsafe_allow_html=True,
            )

    # ── Tab: POSITIONS ────────────────────────────────────────────────────────
    with tab_positions:
        if not TRADES_DB.exists():
            st.markdown('<div class="nodata">No trades database — paper trading hasn\'t executed any trades yet.</div>', unsafe_allow_html=True)
        else:
            try:
                conn = sqlite3.connect(str(TRADES_DB))
                conn.row_factory = sqlite3.Row

                # Summary bar
                all_sells = conn.execute(
                    "SELECT pnl_usd, pnl_pct FROM trades WHERE action!='buy'"
                ).fetchall()
                open_rows = conn.execute(
                    "SELECT * FROM positions WHERE status='open' ORDER BY entry_time DESC"
                ).fetchall()

                total_pnl    = sum(r["pnl_usd"] for r in all_sells) if all_sells else 0
                total_trades = len(all_sells)
                wins         = sum(1 for r in all_sells if r["pnl_usd"] > 0)
                win_rate_pct = wins / total_trades * 100 if total_trades else 0
                best_trade   = max((r["pnl_pct"] for r in all_sells), default=0)

                pnl_color = "#3fb950" if total_pnl > 0 else ("#f85149" if total_pnl < 0 else "#7d8590")

                st.markdown(f"""
<div class="pos-summary">
  <div class="pos-summary-card">
    <div class="pos-summary-val" style="color:{pnl_color}">${total_pnl:+.2f}</div>
    <div class="pos-summary-lbl">Total PnL</div>
  </div>
  <div class="pos-summary-card">
    <div class="pos-summary-val">{win_rate_pct:.1f}%</div>
    <div class="pos-summary-lbl">Win Rate</div>
  </div>
  <div class="pos-summary-card">
    <div class="pos-summary-val">{len(open_rows)}</div>
    <div class="pos-summary-lbl">Open Positions</div>
  </div>
  <div class="pos-summary-card">
    <div class="pos-summary-val" style="color:#3fb950">{best_trade:+.1f}%</div>
    <div class="pos-summary-lbl">Best Trade</div>
  </div>
</div>
""", unsafe_allow_html=True)

                # Open positions
                st.markdown('<div class="sec-label">Open Positions</div>', unsafe_allow_html=True)
                if not open_rows:
                    st.markdown('<div class="nodata">No open positions.</div>', unsafe_allow_html=True)
                else:
                    open_data = []
                    for r in open_rows:
                        ep = r["entry_price"] or 0
                        cp = r["current_price"] or 0
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
                            "Symbol":   r["symbol"],
                            "Entry":    f"${ep:.8f}",
                            "Current":  f"${cp:.8f}",
                            "PnL%":     f"{pnl_pct:+.1f}%",
                            "Size":     f"${r['entry_amount_usd']:.2f}",
                            "Held":     held_s,
                            "Score":    r["screener_score"],
                        })
                    st.dataframe(open_data, use_container_width=True, hide_index=True)

                # Closed trades
                st.markdown('<div class="sec-label">Recent Closed Trades (last 20)</div>', unsafe_allow_html=True)
                closed_rows = conn.execute(
                    """SELECT t.*, p.symbol, p.entry_price, p.entry_amount_usd
                       FROM trades t
                       JOIN positions p ON t.position_id = p.id
                       WHERE t.action != 'buy'
                       ORDER BY t.timestamp DESC LIMIT 20"""
                ).fetchall()

                if not closed_rows:
                    st.markdown('<div class="nodata">No closed trades yet.</div>', unsafe_allow_html=True)
                else:
                    closed_data = []
                    for r in closed_rows:
                        pnl_pct = r["pnl_pct"] or 0
                        closed_data.append({
                            "Pos#":    r["position_id"],
                            "Symbol":  r["symbol"],
                            "Action":  r["action"],
                            "Entry":   f"${r['entry_price']:.8f}",
                            "Exit":    f"${r['price']:.8f}",
                            "PnL $":   f"${r['pnl_usd']:+.2f}",
                            "PnL%":    f"{pnl_pct:+.1f}%",
                            "Reason":  r["exit_reason"] or "—",
                            "Time":    (r["timestamp"] or "")[:16],
                        })
                    st.dataframe(closed_data, use_container_width=True, hide_index=True)

                conn.close()

            except Exception as e:
                st.markdown(
                    f'<div class="nodata">Error loading trades: {e}</div>',
                    unsafe_allow_html=True,
                )
