"""Telegram alert sender."""

from __future__ import annotations

import logging
import os
import time
from typing import Any

import requests
from dotenv import load_dotenv

import config as cfg

load_dotenv()

log = logging.getLogger(__name__)

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")
TELEGRAM_SEND_URL = (
    f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
)


def _age_str(age_min: float | None) -> str:
    if age_min is None:
        return "unknown"
    if age_min < 60:
        return f"{age_min:.0f}m"
    hours = age_min / 60
    if hours < 24:
        return f"{hours:.1f}h"
    return f"{hours / 24:.1f}d"


def _get_alert_level(score: int) -> tuple[str, str] | None:
    for threshold, level, emoji in cfg.ALERT_LEVELS:
        if score >= threshold:
            return level, emoji
    return None


def format_alert(pair: dict[str, Any], result: dict[str, Any], legit: dict[str, Any] | None = None) -> str | None:
    """Format a Telegram alert message. Returns None if score is below threshold."""
    score = result["score"]
    level_info = _get_alert_level(score)
    if level_info is None:
        return None

    level, emoji = level_info
    base = pair.get("baseToken") or {}
    name = base.get("name", "???")
    symbol = base.get("symbol", "???")
    address = base.get("address", "???")
    age_min = result.get("age_minutes")

    liq = (pair.get("liquidity") or {}).get("usd", 0) or 0
    mcap = pair.get("marketCap") or pair.get("fdv") or 0

    vol = pair.get("volume") or {}
    vol_5m = vol.get("m5", 0) or 0
    if vol_5m <= 0:
        vol_5m = (vol.get("h1", 0) or 0) / 12
    vol_liq = vol_5m / liq if liq > 0 else 0

    pc = pair.get("priceChange") or {}
    p5m = pc.get("m5", 0) or 0
    p1h = pc.get("h1", 0) or 0
    p6h = pc.get("h6", 0) or 0

    txns_5m = (pair.get("txns") or {}).get("m5", {})
    buys = txns_5m.get("buys", 0) or 0
    sells = txns_5m.get("sells", 0) or 0

    dex_name = pair.get("dexId", "unknown")
    dex_url = pair.get("url", "")

    # Legitimacy section
    legit_verdict = ""
    legit_lines = ""
    if legit:
        verdict = legit.get("verdict", "?")
        verdict_emoji = {"PASS": "✅", "WARN": "⚠️", "FAIL": "❌"}.get(verdict, "❓")
        legit_verdict = f"{verdict_emoji} Legit check: *{verdict}*"
        details = legit.get("details", {})
        d_lines = []
        top1 = details.get("top1_pct")
        lp_lock = details.get("lp_locked_pct")
        rc_score = details.get("rc_score")
        social = details.get("social", {})
        mint_ok = details.get("mint_revoked")
        freeze_ok = details.get("freeze_revoked")
        if mint_ok is not None:
            d_lines.append(f"  {'✅' if mint_ok else '🚨'} Mint {'revoked' if mint_ok else 'NOT revoked'}")
        if freeze_ok is not None:
            d_lines.append(f"  {'✅' if freeze_ok else '🚨'} Freeze {'revoked' if freeze_ok else 'NOT revoked'}")
        if top1 is not None:
            d_lines.append(f"  {'✅' if top1 < 20 else ('⚠️' if top1 < 50 else '🚨')} Top holder: {top1:.1f}%")
        if lp_lock is not None:
            d_lines.append(f"  {'✅' if lp_lock >= 80 else ('⚠️' if lp_lock >= 50 else '🚨')} LP locked: {lp_lock:.0f}%")
        if rc_score is not None:
            d_lines.append(f"  {'✅' if rc_score < 500 else '⚠️'} RugCheck score: {rc_score}")
        soc_found = [k for k, v in social.items() if v]
        if soc_found:
            d_lines.append(f"  🌐 Socials: {', '.join(soc_found)}")
        else:
            d_lines.append("  ⚠️ No socials found")
        legit_lines = "\n".join(d_lines)

    msg = (
        f"{emoji} MEME ALERT [{level}]\n"
        f"\n"
        f"🪙 {name} (${symbol})\n"
        f"📍 CA: `{address}`\n"
        f"\n"
        f"⏰ Age: {_age_str(age_min)}\n"
        f"💧 Liquidity: ${liq:,.0f}\n"
        f"📊 Market Cap: ${mcap:,.0f}\n"
        f"📈 Vol/Liq: {vol_liq:.1f}x\n"
        f"\n"
        f"Price Change:\n"
        f"  5m: {p5m:+.1f}%\n"
        f"  1h: {p1h:+.1f}%\n"
        f"  6h: {p6h:+.1f}%\n"
        f"\n"
        f"Transactions (5m):\n"
        f"  🟢 Buys: {buys} | 🔴 Sells: {sells}\n"
        f"\n"
        f"⚡ Score: {score}/100\n"
        f"🔗 DEX: {dex_name}\n"
    )

    if legit_verdict:
        msg += f"\n{legit_verdict}\n{legit_lines}\n"

    msg += f"\n[View on DexScreener]({dex_url})"
    return msg


def send_telegram(text: str, dry_run: bool = False) -> bool:
    """Send a message via Telegram Bot API. Returns True on success."""
    if dry_run:
        log.info("[DRY RUN] Would send Telegram alert")
        return True

    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        log.warning("Telegram credentials not set – skipping alert")
        return False

    try:
        resp = requests.post(
            TELEGRAM_SEND_URL,
            json={
                "chat_id": TELEGRAM_CHAT_ID,
                "text": text,
                "parse_mode": "Markdown",
                "disable_web_page_preview": True,
            },
            timeout=cfg.REQUEST_TIMEOUT_SECONDS,
        )
        if resp.status_code == 200:
            log.info("Telegram alert sent successfully")
            return True
        log.warning("Telegram API returned %s: %s", resp.status_code, resp.text)
        return False
    except requests.RequestException as exc:
        log.error("Telegram send failed: %s", exc)
        return False
