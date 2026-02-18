"""Hard rejection filters applied before scoring."""

from __future__ import annotations

import json
import os
import re
import time
from typing import Any

import config as cfg


def _load_seen() -> dict[str, Any]:
    if not os.path.exists(cfg.SEEN_FILE):
        return {}
    try:
        with open(cfg.SEEN_FILE) as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return {}


def _save_seen(seen: dict[str, Any]) -> None:
    with open(cfg.SEEN_FILE, "w") as f:
        json.dump(seen, f, indent=2)


def purge_expired(seen: dict[str, Any]) -> dict[str, Any]:
    """Remove entries older than SEEN_EXPIRY_HOURS."""
    cutoff = time.time() - cfg.SEEN_EXPIRY_HOURS * 3600
    return {
        addr: info
        for addr, info in seen.items()
        if info.get("alerted_at", 0) > cutoff
    }


def load_seen() -> dict[str, Any]:
    seen = _load_seen()
    seen = purge_expired(seen)
    _save_seen(seen)
    return seen


def save_seen(seen: dict[str, Any]) -> None:
    _save_seen(seen)


def mark_seen(
    seen: dict[str, Any], address: str, score: int, level: str
) -> None:
    seen[address] = {
        "score": score,
        "alerted_at": time.time(),
        "alert_level": level,
    }


def _level_rank(level: str) -> int:
    """Higher rank = hotter alert."""
    return {"WATCH": 1, "WARM": 2, "HOT": 3}.get(level, 0)


def should_alert(seen: dict[str, Any], address: str, score: int, level: str) -> bool:
    """Return True if this token should fire an alert."""
    prev = seen.get(address)
    if prev is None:
        return True
    # Re-alert if it jumped to a hotter level
    return _level_rank(level) > _level_rank(prev.get("alert_level", ""))


def _is_scam_name(name: str, symbol: str) -> bool:
    text = f"{name} {symbol}"
    return any(re.search(pat, text) for pat in cfg.SCAM_PATTERNS)


def apply_filters(pair: dict[str, Any]) -> str | None:
    """Return a rejection reason string, or None if the pair passes."""
    # Liquidity check
    liq = (pair.get("liquidity") or {}).get("usd", 0) or 0
    if liq < cfg.MIN_LIQUIDITY_USD:
        return f"low liquidity (${liq:,.0f})"

    # Age check
    created = pair.get("pairCreatedAt")
    if created:
        age_hours = (time.time() * 1000 - created) / 3_600_000
        if age_hours > cfg.MAX_AGE_HOURS:
            return f"too old ({age_hours:.1f}h)"

    # Scam name check
    base = pair.get("baseToken") or {}
    name = base.get("name", "")
    symbol = base.get("symbol", "")
    if _is_scam_name(name, symbol):
        return "scam pattern in name/symbol"

    return None
