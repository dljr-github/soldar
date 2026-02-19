"""
Legitimacy analysis via RugCheck.xyz API + DEX Screener social data.

Returns a structured assessment before any alert is sent.
Hard rejects prevent the alert entirely.
"""

from __future__ import annotations

import logging
import time
from typing import Any

import requests

import config as cfg

log = logging.getLogger("legitimacy")

RUGCHECK_URL = "https://api.rugcheck.xyz/v1/tokens/{mint}/report"


# ── RugCheck fetch ────────────────────────────────────────────────────────────

def _fetch_rugcheck(mint: str) -> dict | None:
    url = RUGCHECK_URL.format(mint=mint)
    try:
        r = requests.get(url, timeout=cfg.REQUEST_TIMEOUT_SECONDS)
        if r.status_code == 200:
            return r.json()
        log.warning("RugCheck returned %s for %s", r.status_code, mint)
    except Exception as e:
        log.warning("RugCheck fetch failed for %s: %s", mint, e)
    return None


# ── Individual checks ─────────────────────────────────────────────────────────

def _check_mint_freeze(rc: dict) -> tuple[bool, bool]:
    """Returns (mint_revoked, freeze_revoked)."""
    token = rc.get("token") or {}
    mint_ok = token.get("mintAuthority") is None
    freeze_ok = token.get("freezeAuthority") is None
    return mint_ok, freeze_ok


def _check_top_holder(rc: dict) -> float:
    """Returns top single holder % (0-100)."""
    holders = rc.get("topHolders") or []
    if not holders:
        return 0.0
    return holders[0].get("pct", 0.0)


def _check_top10_concentration(rc: dict) -> float:
    """Returns combined % held by top 10 holders."""
    holders = rc.get("topHolders") or []
    return sum(h.get("pct", 0.0) for h in holders[:10])


def _check_lp_locked(rc: dict) -> float | None:
    """Returns LP locked % (0-100), or None if no market data available."""
    markets = rc.get("markets") or []
    pcts = []
    for m in markets:
        lp = (m.get("lp") or {})
        pct = lp.get("lpLockedPct")
        if pct is not None:
            pcts.append(float(pct))
    return max(pcts) if pcts else None  # None = no data, not 0%


def _check_rugcheck_risks(rc: dict) -> tuple[int, list[str], bool]:
    """
    Returns (risk_score, risk_labels, has_danger).
    risk_score from RugCheck (higher = worse).
    """
    risks = rc.get("risks") or []
    labels = [f"{r.get('name', 'unknown')} ({r.get('level', 'unknown')})" for r in risks]
    has_danger = any(r.get("level") == "danger" for r in risks)
    return rc.get("score", 0), labels, has_danger


def _check_metadata_mutable(rc: dict) -> bool:
    """Returns True if metadata is LOCKED (good)."""
    meta = rc.get("tokenMeta") or {}
    return not meta.get("mutable", True)


def _check_social(pair: dict) -> dict[str, str | None]:
    """Extract social links from DEX Screener pair/profile data."""
    info = pair.get("info") or {}
    socials = {s.get("type", ""): s.get("url") for s in (info.get("socials") or [])}
    websites = [w.get("url") for w in (info.get("websites") or []) if w.get("url")]
    return {
        "website": websites[0] if websites else None,
        "twitter": socials.get("twitter"),
        "telegram": socials.get("telegram"),
    }


# ── Main analysis ─────────────────────────────────────────────────────────────

def analyse(pair: dict[str, Any]) -> dict[str, Any] | None:
    """
    Run full legitimacy analysis on a token pair.

    Returns a dict with:
      - verdict: "PASS" | "WARN" | "FAIL"
      - reasons: list of human-readable flags (good and bad)
      - hard_reject: bool — if True, do NOT alert under any circumstances
      - details: dict of raw metrics for alert formatting

    Returns None if RugCheck API is unreachable (don't block on API failure).
    """
    base = pair.get("baseToken") or {}
    mint = base.get("address", "")
    if not mint:
        return None

    rc = _fetch_rugcheck(mint)
    if rc is None:
        # Can't verify — don't block, but flag it
        return {
            "verdict": "WARN",
            "hard_reject": False,
            "reasons": ["⚠️ RugCheck unavailable — unverified"],
            "details": {},
        }

    mint_ok, freeze_ok = _check_mint_freeze(rc)
    top1_pct = _check_top_holder(rc)
    top10_pct = _check_top10_concentration(rc)
    lp_locked = _check_lp_locked(rc)
    rc_score, risk_labels, has_danger = _check_rugcheck_risks(rc)
    meta_locked = _check_metadata_mutable(rc)
    social = _check_social(pair)

    reasons = []
    hard_reject = False
    fail_count = 0
    warn_count = 0

    # ── Hard reject conditions ────────────────────────────────────────────────
    if not mint_ok:
        reasons.append("🚨 Mint authority NOT revoked (dev can print tokens)")
        hard_reject = True
        fail_count += 1
    else:
        reasons.append("✅ Mint authority revoked")

    if not freeze_ok:
        reasons.append("🚨 Freeze authority NOT revoked (wallets can be frozen)")
        hard_reject = True
        fail_count += 1
    else:
        reasons.append("✅ Freeze authority revoked")

    if top1_pct > cfg.LEGIT_TOP1_HOLDER_REJECT:
        reasons.append(f"🚨 Top holder owns {top1_pct:.1f}% of supply")
        hard_reject = True
        fail_count += 1
    elif top1_pct > cfg.LEGIT_TOP1_HOLDER_WARN:
        reasons.append(f"⚠️ Top holder owns {top1_pct:.1f}% of supply")
        warn_count += 1
    else:
        reasons.append(f"✅ Top holder {top1_pct:.1f}% — healthy distribution")

    if has_danger:
        reasons.append("🚨 RugCheck flagged DANGER-level risks")
        hard_reject = True
        fail_count += 1

    # ── Warn conditions ───────────────────────────────────────────────────────
    if lp_locked is None:
        reasons.append("⚠️ LP lock data unavailable")
        warn_count += 1
    elif lp_locked < cfg.LEGIT_LP_LOCKED_REJECT:
        reasons.append(f"🚨 LP only {lp_locked:.0f}% locked — rug risk")
        hard_reject = True
        fail_count += 1
    elif lp_locked < cfg.LEGIT_LP_LOCKED_WARN:
        reasons.append(f"⚠️ LP {lp_locked:.0f}% locked (partial)")
        warn_count += 1
    else:
        reasons.append(f"✅ LP {lp_locked:.0f}% locked")

    if rc_score > cfg.LEGIT_RUGCHECK_SCORE_WARN:
        reasons.append(f"⚠️ RugCheck risk score: {rc_score} (elevated)")
        warn_count += 1
    else:
        reasons.append(f"✅ RugCheck score: {rc_score}")

    if not meta_locked:
        reasons.append("⚠️ Token metadata is mutable (can be changed)")
        warn_count += 1
    else:
        reasons.append("✅ Metadata locked")

    if top10_pct > 60:
        reasons.append(f"⚠️ Top 10 holders own {top10_pct:.1f}%")
        warn_count += 1

    # Social presence
    social_present = sum(1 for v in social.values() if v)
    if social_present == 0:
        reasons.append("⚠️ No social links found (website/Twitter/Telegram)")
        warn_count += 1
    else:
        soc_str = ", ".join(k for k, v in social.items() if v)
        reasons.append(f"✅ Socials: {soc_str}")

    # Any extra RugCheck risk flags
    if risk_labels:
        reasons.append(f"⚠️ Risks: {'; '.join(risk_labels[:3])}")

    # ── Verdict ───────────────────────────────────────────────────────────────
    if hard_reject:
        verdict = "FAIL"
    elif warn_count >= 3:
        verdict = "WARN"
    elif warn_count >= 1:
        verdict = "WARN"
    else:
        verdict = "PASS"

    return {
        "verdict": verdict,
        "hard_reject": hard_reject,
        "reasons": reasons,
        "details": {
            "mint_revoked": mint_ok,
            "freeze_revoked": freeze_ok,
            "top1_pct": top1_pct,
            "top10_pct": top10_pct,
            "lp_locked_pct": lp_locked,
            "rc_score": rc_score,
            "meta_locked": meta_locked,
            "social": social,
        },
    }
