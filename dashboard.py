#!/usr/bin/env python3
"""Streamlit dashboard for the Solana meme coin screener."""

from __future__ import annotations

import json
import time
from pathlib import Path

import plotly.express as px
import streamlit as st
from streamlit_autorefresh import st_autorefresh

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(page_title="Meme Screener", layout="wide", page_icon="🪙")

# Auto-refresh every 90 seconds
st_autorefresh(interval=90_000, key="data_refresh")

STATE_FILE = Path(__file__).parent / "data" / "state.json"


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_state() -> dict | None:
    if not STATE_FILE.exists():
        return None
    try:
        with open(STATE_FILE) as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return None


state = load_state()

if state is None:
    st.title("🪙 Solana Meme Screener")
    st.info(
        "Waiting for screener data…\n\n"
        "`data/state.json` not found — the screener will create it on its next cycle. "
        "This page auto-refreshes every 90 seconds.",
        icon="⏳",
    )
    st.stop()


# ---------------------------------------------------------------------------
# Extract data
# ---------------------------------------------------------------------------
candidates: list[dict] = state.get("candidates", [])
hard_rejected: list[dict] = state.get("hard_rejected", [])
cycle_history: list[dict] = state.get("cycle_history", [])
exit_alerts: list[dict] = state.get("exit_alerts", [])

# Parse ISO timestamp to compute "X seconds ago"
last_updated_str = state.get("last_updated", "")
try:
    from datetime import datetime, timezone
    updated_dt = datetime.fromisoformat(last_updated_str)
    seconds_ago = int((datetime.now(timezone.utc) - updated_dt).total_seconds())
except (ValueError, TypeError):
    seconds_ago = 0

cycles_run = state.get("cycle_count", 0)
scanned_count = state.get("candidates_scanned", 0)
passed_count = state.get("passed_filters", len(candidates))

# Sort candidates by score descending
candidates.sort(key=lambda c: c.get("score", 0), reverse=True)
top_score = candidates[0]["score"] if candidates else 0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def score_badge(score: int) -> str:
    if score >= 80:
        return f"🔴 {score}"
    if score >= 65:
        return f"🟠 {score}"
    if score >= 50:
        return f"🟡 {score}"
    return f"⚫ {score}"


def fmt_usd(val: float | int | None) -> str:
    if val is None:
        return "$0"
    return f"${val:,.0f}"


def fmt_pct(val: float | int | None) -> str:
    if val is None:
        return "+0.0%"
    return f"{val:+.1f}%"


def legit_tag(verdict: str | None) -> str:
    if verdict == "PASS":
        return "✅ PASS"
    if verdict == "WARN":
        return "⚠️ WARN"
    if verdict == "FAIL":
        return "❌ FAIL"
    return "❓ N/A"


# ---------------------------------------------------------------------------
# Header row
# ---------------------------------------------------------------------------
st.title("🪙 Solana Meme Screener")

if seconds_ago < 120:
    age_text = f"{seconds_ago}s ago"
else:
    age_text = f"{seconds_ago // 60}m ago"

header_cols = st.columns([2, 1, 1, 1])
with header_cols[0]:
    st.caption(f"Last updated: **{age_text}**")
with header_cols[1]:
    st.metric("Scanned", scanned_count)
with header_cols[2]:
    st.metric("Passed", passed_count)
with header_cols[3]:
    st.metric("Top Score", top_score)


# ---------------------------------------------------------------------------
# Sidebar filters
# ---------------------------------------------------------------------------
st.sidebar.header("Filters")
min_score = st.sidebar.slider("Min score", 0, 100, 0)
max_age = st.sidebar.slider("Max age (min)", 0, 360, 360)
legit_filter = st.sidebar.radio(
    "Legitimacy filter",
    ["All", "PASS only", "PASS + WARN"],
    index=0,
)

# Apply filters
filtered = []
for c in candidates:
    if c.get("score", 0) < min_score:
        continue
    age = c.get("age_minutes")
    if age is not None and age > max_age:
        continue
    verdict = c.get("legit_verdict")
    if legit_filter == "PASS only" and verdict != "PASS":
        continue
    if legit_filter == "PASS + WARN" and verdict not in ("PASS", "WARN"):
        continue
    filtered.append(c)


# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------
tab_live, tab_stats, tab_rejected, tab_positions = st.tabs(
    ["🔥 Live Candidates", "📊 Stats", "🚫 Rejected", "💰 Positions"]
)


# ---------------------------------------------------------------------------
# Tab 1 — Live Candidates
# ---------------------------------------------------------------------------
with tab_live:
    # --- Exit Signals (shown at top when present) ---
    if exit_alerts:
        st.markdown("### 🚨 Exit Signals")
        for ea in exit_alerts:
            urgency = ea.get("urgency", "LOW")
            symbol = ea.get("symbol", "?")
            reason = ea.get("exit_reason", "")
            all_sigs = ea.get("all_signals", [])
            p5m = ea.get("current_price_change_5m", 0)
            p1h = ea.get("current_price_change_1h", 0)
            urgency_icon = {"HIGH": "🔴", "MEDIUM": "🟠", "LOW": "🟡"}.get(urgency, "⚪")
            st.error(
                f"**{urgency_icon} {urgency}** — **{symbol}**: {reason}  \n"
                f"5m: {fmt_pct(p5m)} | 1h: {fmt_pct(p1h)}  \n"
                f"All signals: {', '.join(all_sigs)}",
                icon="🚨",
            )
        st.markdown("---")

    if not filtered:
        st.info("No candidates match current filters.")
    else:
        st.caption(f"Showing {len(filtered)} candidate(s)")
        for c in filtered:
            score = c.get("score", 0)
            symbol = c.get("symbol", "?")
            age = c.get("age_minutes")
            age_s = f"{age:.0f}m" if age is not None else "?"
            liq = c.get("liquidity_usd", 0)
            p5m = c.get("price_change_5m", 0)
            p1h = c.get("price_change_1h", 0)
            verdict = c.get("legit_verdict", "N/A")

            badge = score_badge(score)
            header = (
                f"{badge}  **{symbol}** — "
                f"Age: {age_s} | Liq: {fmt_usd(liq)} | "
                f"5m: {fmt_pct(p5m)} | 1h: {fmt_pct(p1h)} | "
                f"Legit: {legit_tag(verdict)}"
            )

            with st.expander(header, expanded=(score >= 80)):
                # Stats table
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.markdown("**Market Stats**")
                    mcap = c.get("mcap_usd", 0)
                    vol_liq = c.get("vol_liq_ratio", 0)
                    st.text(f"MCap:      {fmt_usd(mcap)}")
                    st.text(f"Liquidity: {fmt_usd(liq)}")
                    st.text(f"Vol/Liq:   {vol_liq:.2f}x")
                with col2:
                    st.markdown("**Price Change**")
                    p6h = c.get("price_change_6h", 0)
                    st.text(f"5m:  {fmt_pct(p5m)}")
                    st.text(f"1h:  {fmt_pct(p1h)}")
                    st.text(f"6h:  {fmt_pct(p6h)}")
                with col3:
                    st.markdown("**Transactions (5m)**")
                    buys = c.get("buys_5m", 0)
                    sells = c.get("sells_5m", 0)
                    st.text(f"🟢 Buys:  {buys}")
                    st.text(f"🔴 Sells: {sells}")

                # Legitimacy details (flat fields from state writer)
                mint_ok = c.get("legit_mint_revoked")
                has_legit = mint_ok is not None
                if has_legit:
                    st.markdown("---")
                    st.markdown("**Legitimacy Details**")
                    ld1, ld2 = st.columns(2)
                    with ld1:
                        freeze_ok = c.get("legit_freeze_revoked")
                        top1 = c.get("legit_top1_pct")
                        lp_lock = c.get("legit_lp_locked_pct")
                        st.text(
                            f"Mint revoked:    {'✅ Yes' if mint_ok else '❌ No' if mint_ok is not None else '? Unknown'}"
                        )
                        st.text(
                            f"Freeze revoked:  {'✅ Yes' if freeze_ok else '❌ No' if freeze_ok is not None else '? Unknown'}"
                        )
                        if top1 is not None:
                            st.text(f"Top holder:      {top1:.1f}%")
                        if lp_lock is not None:
                            st.text(f"LP locked:       {lp_lock:.0f}%")
                    with ld2:
                        rc_score = c.get("legit_rc_score")
                        socials = c.get("legit_socials", [])
                        if rc_score is not None:
                            st.text(f"RugCheck score:  {rc_score}")
                        st.text(
                            f"Socials:         {', '.join(socials) if socials else 'None'}"
                        )

                # Legit reasons
                legit_reasons = c.get("legit_reasons", [])
                if legit_reasons:
                    st.markdown("---")
                    st.markdown("**Legitimacy Flags**")
                    for reason in legit_reasons:
                        st.text(f"  {reason}")

                # DexScreener link
                dex_url = c.get("url", "")
                address = c.get("address", "")
                if dex_url:
                    st.markdown(f"[View on DexScreener]({dex_url})")
                elif address:
                    st.markdown(
                        f"[View on DexScreener](https://dexscreener.com/solana/{address})"
                    )


# ---------------------------------------------------------------------------
# Tab 2 — Stats
# ---------------------------------------------------------------------------
with tab_stats:
    st_cols = st.columns(3)
    scores = [c.get("score", 0) for c in candidates]
    avg_score = sum(scores) / len(scores) if scores else 0
    watch_plus = sum(1 for s in scores if s >= 50)

    with st_cols[0]:
        st.metric("Avg Score", f"{avg_score:.1f}")
    with st_cols[1]:
        st.metric("WATCH+ Candidates", watch_plus)
    with st_cols[2]:
        st.metric("Cycles Run", cycles_run)

    # Top score over time
    if cycle_history:
        times = [h.get("timestamp", "") for h in cycle_history]
        top_scores = [h.get("top_score", 0) for h in cycle_history]
        fig_line = px.line(
            x=times,
            y=top_scores,
            labels={"x": "Time", "y": "Top Score"},
            title="Top Score Over Time",
        )
        fig_line.update_layout(
            xaxis_title="Cycle",
            yaxis_title="Score",
            yaxis_range=[0, 100],
        )
        st.plotly_chart(fig_line, use_container_width=True)
    else:
        st.info("No cycle history available yet.")

    # Score distribution bar chart
    if scores:
        bins = list(range(0, 110, 10))
        bin_labels = [f"{b}-{b+9}" for b in bins[:-1]]
        counts = [0] * (len(bins) - 1)
        for s in scores:
            idx = min(s // 10, len(counts) - 1)
            counts[idx] += 1
        fig_bar = px.bar(
            x=bin_labels,
            y=counts,
            labels={"x": "Score Range", "y": "Count"},
            title="Score Distribution",
        )
        fig_bar.update_layout(xaxis_title="Score Range", yaxis_title="Candidates")
        st.plotly_chart(fig_bar, use_container_width=True)
    else:
        st.info("No candidates to chart.")


# ---------------------------------------------------------------------------
# Tab 3 — Rejected
# ---------------------------------------------------------------------------
with tab_rejected:
    if not hard_rejected:
        st.info("No hard rejections this session.")
    else:
        rows = []
        for r in hard_rejected:
            sym = r.get("symbol", "?")
            addr = r.get("address", "?")
            addr_short = addr[:8] + "…" + addr[-4:] if len(addr) > 12 else addr
            reasons = "; ".join(r.get("reasons", []))
            ts = r.get("timestamp", "")
            rows.append(
                {
                    "Symbol": sym,
                    "Address": addr_short,
                    "Reasons": reasons,
                    "Timestamp": ts,
                }
            )
        st.dataframe(rows, use_container_width=True)


# ---------------------------------------------------------------------------
# Tab 4 — Positions
# ---------------------------------------------------------------------------
with tab_positions:
    import sqlite3
    from datetime import datetime, timezone, timedelta

    TRADES_DB = Path(__file__).parent / "data" / "trades.db"

    if not TRADES_DB.exists():
        st.info("No trades database yet — paper trading hasn't executed any trades.")
    else:
        _conn = sqlite3.connect(str(TRADES_DB))
        _conn.row_factory = sqlite3.Row

        # --- Open positions ---
        open_rows = _conn.execute(
            "SELECT * FROM positions WHERE status = 'open' ORDER BY entry_time DESC"
        ).fetchall()

        st.markdown("### Open Positions")
        if not open_rows:
            st.info("No open positions.")
        else:
            open_data = []
            for r in open_rows:
                entry_price = r["entry_price"] or 0
                current_price = r["current_price"] or 0
                pnl_pct = ((current_price - entry_price) / entry_price * 100) if entry_price else 0
                # Time held
                try:
                    entry_dt = datetime.fromisoformat(r["entry_time"])
                    held = datetime.now(timezone.utc) - entry_dt
                    held_str = f"{held.seconds // 3600}h {(held.seconds % 3600) // 60}m"
                    if held.days > 0:
                        held_str = f"{held.days}d {held_str}"
                except (ValueError, TypeError):
                    held_str = "?"
                open_data.append({
                    "ID": r["id"],
                    "Symbol": r["symbol"],
                    "Entry Price": f"${entry_price:.8f}",
                    "Current Price": f"${current_price:.8f}",
                    "PnL%": f"{pnl_pct:+.1f}%",
                    "Size": f"${r['entry_amount_usd']:.2f}",
                    "Time Held": held_str,
                    "Score": r["screener_score"],
                })
            st.dataframe(open_data, use_container_width=True)

        # --- Recent closed trades ---
        st.markdown("### Recent Closed Trades")
        closed_rows = _conn.execute(
            """SELECT t.*, p.symbol, p.entry_price, p.entry_amount_usd
               FROM trades t
               JOIN positions p ON t.position_id = p.id
               WHERE t.action != 'buy'
               ORDER BY t.timestamp DESC
               LIMIT 20"""
        ).fetchall()

        if not closed_rows:
            st.info("No closed trades yet.")
        else:
            closed_data = []
            for r in closed_rows:
                closed_data.append({
                    "Pos#": r["position_id"],
                    "Symbol": r["symbol"],
                    "Action": r["action"],
                    "Entry": f"${r['entry_price']:.8f}",
                    "Exit": f"${r['price']:.8f}",
                    "PnL $": f"${r['pnl_usd']:+.2f}",
                    "PnL %": f"{r['pnl_pct']:+.1f}%",
                    "Reason": r["exit_reason"] or "",
                    "Time": r["timestamp"],
                })
            st.dataframe(closed_data, use_container_width=True)

        # --- Summary stats ---
        st.markdown("### Trading Summary")
        all_sells = _conn.execute(
            "SELECT pnl_usd, pnl_pct, timestamp FROM trades WHERE action != 'buy'"
        ).fetchall()

        total_trades = len(all_sells)
        if total_trades > 0:
            pnls = [r["pnl_usd"] for r in all_sells]
            wins = sum(1 for p in pnls if p > 0)
            total_pnl = sum(pnls)

            # Daily PnL
            today_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
            daily_sells = _conn.execute(
                "SELECT COALESCE(SUM(pnl_usd), 0) FROM trades WHERE action != 'buy' AND timestamp >= ?",
                (today_str,),
            ).fetchone()
            daily_pnl = daily_sells[0]

            s1, s2, s3, s4 = st.columns(4)
            with s1:
                st.metric("Total Trades", total_trades)
            with s2:
                st.metric("Win Rate", f"{wins / total_trades * 100:.1f}%")
            with s3:
                st.metric("Total PnL", f"${total_pnl:+.2f}")
            with s4:
                st.metric("Daily PnL", f"${daily_pnl:+.2f}")
        else:
            st.info("No completed trades to summarize.")

        _conn.close()
