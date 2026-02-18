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
    vol_5m = (pair.get("volume") or {}).get("m5", 0) or 0
    liq = (pair.get("liquidity") or {}).get("usd", 0) or 0
    if liq <= 0:
        return 0
    ratio = vol_5m / liq
    for threshold, pts in cfg.VOL_LIQ_THRESHOLDS:
        if ratio > threshold:
            return pts
    return cfg.VOL_LIQ_DEFAULT_POINTS


def _score_momentum(pair: dict[str, Any]) -> int:
    pc = pair.get("priceChange") or {}
    p5m = pc.get("m5", 0) or 0
    p1h = pc.get("h1", 0) or 0
    score = cfg.MOMENTUM_DEFAULT_POINTS
    for threshold, pts in cfg.MOMENTUM_5M_THRESHOLDS:
        if p5m > threshold:
            score = pts
            break
    if p5m > 0 and p1h > 0:
        score += cfg.MOMENTUM_BONUS
    return score


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
