"""Scoring engine – assigns 0-100 score to each token pair."""

from __future__ import annotations

import time
from typing import Any

import config as cfg


def _age_minutes(pair: dict[str, Any]) -> float | None:
    created = pair.get("pairCreatedAt")
    if created is None:
        return None
    return (time.time() * 1000 - created) / 60_000


def _score_age(age_min: float | None) -> int:
    if age_min is None:
        return 0
    for threshold_min, pts in cfg.AGE_THRESHOLDS:
        if age_min < threshold_min:
            return pts
    return cfg.AGE_DEFAULT_POINTS


def _score_vol_liq(pair: dict[str, Any]) -> int:
    vol = pair.get("volume") or {}
    liq = (pair.get("liquidity") or {}).get("usd", 0) or 0
    if liq <= 0:
        return 0
    # Prefer 5m volume; fall back to h1/6 normalised to a 5m window
    vol_5m = vol.get("m5", 0) or 0
    if vol_5m <= 0:
        vol_h1 = vol.get("h1", 0) or 0
        vol_5m = vol_h1 / 12  # h1 → estimated 5m equivalent
    if vol_5m <= 0:
        vol_h6 = vol.get("h6", 0) or 0
        vol_5m = vol_h6 / 72
    ratio = vol_5m / liq
    for threshold, pts in cfg.VOL_LIQ_THRESHOLDS:
        if ratio > threshold:
            return pts
    return cfg.VOL_LIQ_DEFAULT_POINTS


def _score_momentum(pair: dict[str, Any]) -> int:
    pc = pair.get("priceChange") or {}
    p5m = pc.get("m5", 0) or 0
    p1h = pc.get("h1", 0) or 0

    # 5m momentum (up to 12 pts) — freshest signal
    score_5m = cfg.MOMENTUM_DEFAULT_POINTS
    for threshold, pts in cfg.MOMENTUM_5M_THRESHOLDS:
        if p5m > threshold:
            score_5m = pts
            break

    # 1h momentum (up to 8 pts) — catching early movers
    # Cap benefit at 500% — beyond that you've probably missed it
    score_1h = cfg.MOMENTUM_DEFAULT_POINTS
    p1h_capped = min(p1h, 500)
    for threshold, pts in cfg.MOMENTUM_1H_THRESHOLDS:
        if p1h_capped > threshold:
            score_1h = pts
            break

    # Bonus if both directions agree (both positive)
    bonus = cfg.MOMENTUM_BONUS if (p5m > 0 and p1h > 0) else 0

    return score_5m + score_1h + bonus


def _score_buy_pressure(pair: dict[str, Any]) -> int:
    txns_5m = (pair.get("txns") or {}).get("m5", {})
    buys = txns_5m.get("buys", 0) or 0
    sells = txns_5m.get("sells", 0) or 0
    total = buys + sells
    if total == 0:
        return 0
    ratio = buys / total
    for threshold, pts in cfg.BUY_PRESSURE_THRESHOLDS:
        if ratio > threshold:
            return pts
    return cfg.BUY_PRESSURE_DEFAULT_POINTS


def _score_liquidity_sweet_spot(pair: dict[str, Any]) -> int:
    liq = (pair.get("liquidity") or {}).get("usd", 0) or 0
    for lo, hi, pts in cfg.LIQ_SWEET_SPOT:
        if lo <= liq <= hi:
            return pts
    return cfg.LIQ_SWEET_DEFAULT_POINTS


def _score_mcap(pair: dict[str, Any]) -> int:
    mcap = pair.get("marketCap") or pair.get("fdv") or 0
    if mcap <= 0:
        return 0
    for threshold, pts in cfg.MCAP_THRESHOLDS:
        if mcap < threshold:
            return pts
    return cfg.MCAP_DEFAULT_POINTS


def score_pair(pair: dict[str, Any]) -> dict[str, Any]:
    """Return a dict with breakdown and total score for *pair*."""
    age_min = _age_minutes(pair)
    breakdown = {
        "age": _score_age(age_min),
        "vol_liq": _score_vol_liq(pair),
        "momentum": _score_momentum(pair),
        "buy_pressure": _score_buy_pressure(pair),
        "liq_sweet": _score_liquidity_sweet_spot(pair),
        "mcap": _score_mcap(pair),
    }
    total = sum(breakdown.values())
    # Cap at 100
    total = min(total, 100)
    return {"score": total, "breakdown": breakdown, "age_minutes": age_min}


# ---------------------------------------------------------------------------
# Exit signal detection
# ---------------------------------------------------------------------------

def _get_vol_liq_ratio(pair: dict[str, Any]) -> float:
    """Compute vol/liq ratio the same way as _score_vol_liq."""
    vol = pair.get("volume") or {}
    liq = (pair.get("liquidity") or {}).get("usd", 0) or 0
    if liq <= 0:
        return 0.0
    vol_5m = vol.get("m5", 0) or 0
    if vol_5m <= 0:
        vol_h1 = vol.get("h1", 0) or 0
        vol_5m = vol_h1 / 12
    if vol_5m <= 0:
        vol_h6 = vol.get("h6", 0) or 0
        vol_5m = vol_h6 / 72
    return vol_5m / liq


def score_exit(pair: dict[str, Any], seen_entry: dict[str, Any]) -> dict[str, Any]:
    """Check exit signals for a previously-alerted coin.

    *seen_entry* must contain first_seen_price_change_1h and
    first_seen_vol_liq from when the coin was first alerted.

    Returns a dict with should_exit, exit_reason, exit_signals, urgency.
    """
    signals: list[tuple[str, str]] = []  # (reason, urgency)

    pc = pair.get("priceChange") or {}
    p5m = pc.get("m5", 0) or 0
    p1h = pc.get("h1", 0) or 0
    txns_5m = (pair.get("txns") or {}).get("m5", {})
    buys = txns_5m.get("buys", 0) or 0
    sells = txns_5m.get("sells", 0) or 0
    total_txns = buys + sells
    liq = (pair.get("liquidity") or {}).get("usd", 0) or 0
    age_min = _age_minutes(pair)
    score = score_pair(pair)["score"]

    # 1. Sell pressure flip — more than 65% sells
    if total_txns > 0 and buys / total_txns < 0.35:
        signals.append(("Sell pressure dominant", "HIGH"))

    # 2. Volume collapse — vol/liq dropped >70% vs first alert
    first_vol_liq = seen_entry.get("first_seen_vol_liq") or 0
    if first_vol_liq > 0:
        current_vol_liq = _get_vol_liq_ratio(pair)
        drop_pct = (first_vol_liq - current_vol_liq) / first_vol_liq
        if drop_pct > 0.70:
            signals.append(("Volume drying up", "MEDIUM"))

    # 3. Price reversal — sharp 5m drop after a prior pump
    first_p1h = seen_entry.get("first_seen_price_change_1h") or 0
    if p5m < -15 and first_p1h > 20:
        signals.append(("Sharp 5m reversal after pump", "HIGH"))

    # 4. Liquidity drain — liquidity dropped >40% from first-seen
    first_liq = seen_entry.get("first_seen_liq") or 0
    if first_liq > 0 and liq > 0:
        liq_drop = (first_liq - liq) / first_liq
        if liq_drop > 0.40:
            signals.append(("Liquidity being pulled", "HIGH"))

    # 5. Age-based soft exit — >2h old and score < 30
    if age_min is not None and age_min > 120 and score < 30:
        signals.append(("Momentum faded — too old", "LOW"))

    # 6. Dump pattern — 5m change < -25%
    if p5m < -25:
        signals.append(("Dump in progress", "HIGH"))

    if not signals:
        return {
            "should_exit": False,
            "exit_reason": "",
            "exit_signals": [],
            "urgency": "LOW",
        }

    # Pick the highest urgency as the primary
    urgency_rank = {"HIGH": 3, "MEDIUM": 2, "LOW": 1}
    signals.sort(key=lambda s: urgency_rank.get(s[1], 0), reverse=True)
    primary_reason = signals[0][0]
    primary_urgency = signals[0][1]

    return {
        "should_exit": True,
        "exit_reason": primary_reason,
        "exit_signals": [s[0] for s in signals],
        "urgency": primary_urgency,
    }
