"""
Pump.fun direct integration for early meme coin detection.
Fetches new token listings before they appear on DEX Screener.
"""

from __future__ import annotations

import logging
import time
from typing import Any

import requests

import config as cfg

log = logging.getLogger("pumpfun")

BASE_URL = "https://frontend-api-v3.pump.fun"

# Target SOL reserves for bonding curve graduation (~85 SOL historically)
BONDING_CURVE_TARGET_SOL = 85.0

_HEADERS = {
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36",
    "Accept": "application/json",
}


def fetch_new_tokens(limit: int = 50) -> list[dict]:
    """Fetch newest Pump.fun tokens sorted by creation time (newest first)."""
    url = (
        f"{BASE_URL}/coins"
        f"?offset=0&limit={limit}"
        f"&sort=created_timestamp&order=DESC&includeNsfw=false"
    )
    try:
        resp = requests.get(url, headers=_HEADERS, timeout=cfg.REQUEST_TIMEOUT_SECONDS)
        resp.raise_for_status()
        data = resp.json()
        # API returns a list directly
        if isinstance(data, list):
            return data
        # Some endpoints wrap in an object
        return data.get("coins", data.get("data", []))
    except requests.RequestException as exc:
        log.warning("Pump.fun fetch_new_tokens failed: %s", exc)
        return []


def fetch_token_detail(mint: str) -> dict:
    """Fetch detailed info for a single Pump.fun token by mint address."""
    url = f"{BASE_URL}/coins/{mint}"
    try:
        resp = requests.get(url, headers=_HEADERS, timeout=cfg.REQUEST_TIMEOUT_SECONDS)
        resp.raise_for_status()
        return resp.json()
    except requests.RequestException as exc:
        log.warning("Pump.fun fetch_token_detail(%s) failed: %s", mint[:8], exc)
        return {}


def fetch_recent_trades(mint: str, limit: int = 50) -> list[dict]:
    """Fetch recent trades for a Pump.fun token."""
    url = f"{BASE_URL}/trades/all/{mint}?limit={limit}&offset=0"
    try:
        resp = requests.get(url, headers=_HEADERS, timeout=cfg.REQUEST_TIMEOUT_SECONDS)
        resp.raise_for_status()
        data = resp.json()
        if isinstance(data, list):
            return data
        return data.get("trades", data.get("data", []))
    except requests.RequestException as exc:
        log.warning("Pump.fun fetch_recent_trades(%s) failed: %s", mint[:8], exc)
        return []


def compute_pumpfun_features(token: dict) -> dict:
    """Compute features from a Pump.fun token listing data.

    Uses fields available from the /coins listing endpoint directly
    (trades endpoint is unavailable on v3 API).
    """
    now = time.time()

    # Age in minutes from created_timestamp (epoch milliseconds)
    created_ts = token.get("created_timestamp", 0)
    if created_ts > 1e12:
        age_minutes = (now - created_ts / 1000) / 60
    elif created_ts > 0:
        age_minutes = (now - created_ts) / 60
    else:
        age_minutes = 999

    market_cap_usd = token.get("usd_market_cap", 0) or 0
    sol_reserves = token.get("virtual_sol_reserves", 0) or 0
    # Normalize from lamports if needed (> 1B likely lamports)
    if sol_reserves > 1_000_000_000:
        sol_reserves = sol_reserves / 1e9

    # Activity proxy: reply_count and last_trade_timestamp
    reply_count = token.get("reply_count", 0) or 0

    # Trade velocity proxy: replies per minute (community engagement correlates
    # with trading activity on pump.fun)
    trade_velocity = reply_count / max(age_minutes, 0.5) if age_minutes > 0 else 0

    # Buy pressure proxy: use real_sol_reserves growth vs initial
    # Higher real_sol_reserves relative to initial (~0 SOL) = net buying
    real_sol = token.get("real_sol_reserves", 0) or 0
    if real_sol > 1_000_000_000:
        real_sol = real_sol / 1e9
    # Estimate buy pressure from how much SOL is in the curve
    # A fresh token starts at ~0 real SOL; buying pushes it up
    # Normalize: 30 SOL in curve = strong buying (target is ~85 SOL for graduation)
    buy_pressure = min(real_sol / 30.0, 1.0) if real_sol > 0 else 0.5

    # Social presence
    has_social = bool(
        token.get("website")
        or token.get("twitter")
        or token.get("telegram")
    )

    description = token.get("description", "") or ""
    description_length = len(description)

    # Bonding curve progress
    bonding_curve_pct = 0.0
    if BONDING_CURVE_TARGET_SOL > 0 and sol_reserves > 0:
        bonding_curve_pct = min((sol_reserves / BONDING_CURVE_TARGET_SOL) * 100, 100)

    graduated = bool(token.get("complete", False))

    return {
        "age_minutes": round(age_minutes, 1),
        "market_cap_usd": market_cap_usd,
        "sol_reserves": round(sol_reserves, 2),
        "buy_pressure": round(buy_pressure, 3),
        "trade_velocity": round(trade_velocity, 1),
        "has_social": has_social,
        "description_length": description_length,
        "bonding_curve_pct": round(bonding_curve_pct, 1),
        "graduated": graduated,
        "reply_count": reply_count,
    }


def score_pumpfun_token(features: dict) -> int:
    """Score a Pump.fun token 0-100 based on bonding curve stage features.

    Scoring adapted for pre-Raydium bonding curve tokens:
    - Age < 5 min: 30 pts | 5-15 min: 20 pts | 15-30 min: 10 pts
    - Trade velocity > 10/min: 20 pts | 5-10: 12 pts | 2-5: 6 pts
    - Buy pressure > 0.7: 15 pts | 0.6: 10 pts | 0.5: 5 pts
    - Has social: 10 pts
    - Description length > 50: 5 pts
    - Bonding curve > 20%: 10 pts (real traction)
    - Graduated (on Raydium): 10 pts bonus
    """
    score = 0

    # Age freshness (30 pts max)
    age = features.get("age_minutes", 999)
    if age < 5:
        score += 30
    elif age < 15:
        score += 20
    elif age < 30:
        score += 10

    # Trade velocity (20 pts max)
    velocity = features.get("trade_velocity", 0)
    if velocity > 10:
        score += 20
    elif velocity > 5:
        score += 12
    elif velocity > 2:
        score += 6

    # Buy pressure (15 pts max)
    bp = features.get("buy_pressure", 0)
    if bp > 0.7:
        score += 15
    elif bp > 0.6:
        score += 10
    elif bp > 0.5:
        score += 5

    # Social presence (10 pts)
    if features.get("has_social", False):
        score += 10

    # Description quality (5 pts)
    if features.get("description_length", 0) > 50:
        score += 5

    # Bonding curve traction (10 pts)
    if features.get("bonding_curve_pct", 0) > 20:
        score += 10

    # Graduated bonus (10 pts)
    if features.get("graduated", False):
        score += 10

    return min(score, 100)


def scan_pumpfun(limit: int = 50, max_age_minutes: float = 30) -> list[dict[str, Any]]:
    """Fetch, score, and return Pump.fun candidates under max_age_minutes.

    Returns a list of dicts compatible with screener.py state format,
    each with source="pumpfun".
    """
    tokens = fetch_new_tokens(limit=limit)
    if not tokens:
        log.info("Pump.fun: no tokens returned")
        return []

    candidates = []
    for token in tokens:
        mint = token.get("mint", "")
        if not mint:
            continue

        # Quick age check before fetching trades
        created_ts = token.get("created_timestamp", 0)
        now = time.time()
        if created_ts > 1e12:
            age_min = (now - created_ts / 1000) / 60
        elif created_ts > 0:
            age_min = (now - created_ts) / 60
        else:
            continue

        if age_min > max_age_minutes:
            continue

        features = compute_pumpfun_features(token)
        score = score_pumpfun_token(features)

        symbol = token.get("symbol", "?")
        name = token.get("name", "?")

        candidates.append({
            "symbol": symbol,
            "name": name,
            "address": mint,
            "score": score,
            "level": _get_level(score),
            "age_minutes": features["age_minutes"],
            "liquidity_usd": 0,  # No DEX liquidity yet (bonding curve)
            "mcap_usd": round(features["market_cap_usd"], 2),
            "vol_liq_ratio": 0,
            "price_change_5m": 0,
            "price_change_1h": 0,
            "price_change_6h": 0,
            "buys_5m": 0,
            "sells_5m": 0,
            "dex": "pump.fun",
            "url": f"https://pump.fun/coin/{mint}",
            "source": "pumpfun",
            "pumpfun_features": features,
            "legit_verdict": None,
            "legit_hard_rejected": False,
            "legit_top1_pct": None,
            "legit_lp_locked_pct": None,
            "legit_rc_score": None,
            "legit_mint_revoked": None,
            "legit_freeze_revoked": None,
            "legit_socials": [],
            "legit_reasons": [],
            "first_seen": None,  # Will be set by screener
        })

    candidates.sort(key=lambda c: c["score"], reverse=True)
    log.info(
        "Pump.fun: %d tokens fetched, %d under %dm, top score %d",
        len(tokens), len(candidates), max_age_minutes,
        candidates[0]["score"] if candidates else 0,
    )
    return candidates


def _get_level(score: int) -> str | None:
    for threshold, level, _emoji in cfg.ALERT_LEVELS:
        if score >= threshold:
            return level
    return None
