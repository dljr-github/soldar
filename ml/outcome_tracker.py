"""Outcome tracker: records post-alert price performance for ML training labels."""

from __future__ import annotations

import json
import logging
import os
import tempfile
import time
from datetime import datetime, timezone
from typing import Any

import requests

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SEEN_FILE = os.path.join(_ROOT, "seen.json")
OUTCOMES_FILE = os.path.join(_ROOT, "data", "outcomes.json")

DEXSCREENER_TOKEN_URL = "https://api.dexscreener.com/latest/dex/tokens/{address}"

# Checkpoint offsets in seconds from first alert
CHECKPOINTS: dict[str, int] = {
    "15m": 15 * 60,
    "30m": 30 * 60,
    "60m": 60 * 60,
    "2h":  2 * 3600,
    "4h":  4 * 3600,
}

# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def _load_json(path: str) -> dict:
    if not os.path.exists(path):
        return {}
    try:
        with open(path) as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return {}


def _save_json_atomic(path: str, data: dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=os.path.dirname(path), suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(data, f, indent=2)
        os.replace(tmp, path)
    except Exception:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise


# ---------------------------------------------------------------------------
# DexScreener price fetch
# ---------------------------------------------------------------------------

def _fetch_price(address: str) -> float | None:
    """Return current USD price from DexScreener, or None on failure."""
    url = DEXSCREENER_TOKEN_URL.format(address=address)
    try:
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        pairs = data.get("pairs") or []
        # Pick the most-liquid Solana pair
        solana_pairs = [p for p in pairs if p.get("chainId") == "solana"]
        if not solana_pairs:
            return None
        best = max(
            solana_pairs,
            key=lambda p: (p.get("liquidity") or {}).get("usd", 0) or 0,
        )
        price_str = best.get("priceUsd")
        return float(price_str) if price_str else None
    except Exception as exc:
        log.debug("Price fetch failed for %s: %s", address[:8], exc)
        return None


# ---------------------------------------------------------------------------
# Final status classifier
# ---------------------------------------------------------------------------

def _classify_final(outcomes: dict[str, Any | None], price_at_alert: float) -> str:
    """Compute final_status from completed outcomes."""
    changes: list[float] = []
    for cp in CHECKPOINTS:
        o = outcomes.get(cp)
        if o is not None:
            changes.append(o["change_pct"])

    if not changes:
        return "pending"

    peak = max(changes)
    last = changes[-1]

    if peak >= 900:
        return "pumped_10x"
    if peak >= 400:
        return "pumped_5x"
    if peak >= 100:
        return "pumped_2x"
    if peak >= 30:
        return "modest_gain"
    if last <= -70:
        return "dumped"
    # All checkpoints filled with small movement
    if all(outcomes.get(cp) is not None for cp in CHECKPOINTS):
        return "flat"
    return "pending"


# ---------------------------------------------------------------------------
# OutcomeTracker
# ---------------------------------------------------------------------------

class OutcomeTracker:
    def __init__(self) -> None:
        self.outcomes: dict[str, Any] = _load_json(OUTCOMES_FILE)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run_checkpoint(self) -> None:
        """Check all unresolved tokens and record any due checkpoints."""
        seen = _load_json(SEEN_FILE)
        changed = False

        for address, entry in seen.items():
            if not entry.get("alert_level"):
                continue  # never alerted

            alerted_at_ts = entry.get("alerted_at")
            if not alerted_at_ts:
                continue

            score = entry.get("score")
            level = entry.get("alert_level")
            alerted_at_iso = datetime.fromtimestamp(
                alerted_at_ts, tz=timezone.utc
            ).isoformat()

            # Bootstrap record if missing
            if address not in self.outcomes:
                self.outcomes[address] = {
                    "symbol": None,  # filled on first price fetch
                    "alerted_at": alerted_at_iso,
                    "alert_score": score,
                    "alert_level": level,
                    "price_at_alert": None,
                    "outcomes": {cp: None for cp in CHECKPOINTS},
                    "peak_gain_pct": None,
                    "peak_gain_at": None,
                    "final_status": "pending",
                }
                changed = True

            rec = self.outcomes[address]

            # Skip if already fully resolved
            if rec["final_status"] not in ("pending", None):
                continue

            now_ts = time.time()
            elapsed = now_ts - alerted_at_ts

            # Check which checkpoints are due
            needs_fetch = any(
                rec["outcomes"].get(cp) is None
                and elapsed >= offset
                for cp, offset in CHECKPOINTS.items()
            )
            if not needs_fetch:
                continue

            # Fetch price once per token per cycle
            price = _fetch_price(address)
            if price is None:
                log.debug("No price for %s, skipping checkpoint", address[:8])
                continue

            if rec["price_at_alert"] is None:
                rec["price_at_alert"] = price
                log.debug("Bootstrapped alert price %.8f for %s", price, address[:8])

            alert_price = rec["price_at_alert"]
            now_iso = datetime.now(timezone.utc).isoformat()

            for cp, offset in CHECKPOINTS.items():
                if rec["outcomes"][cp] is not None:
                    continue
                if elapsed < offset:
                    continue

                change_pct = (
                    ((price - alert_price) / alert_price * 100)
                    if alert_price and alert_price > 0
                    else 0.0
                )
                rec["outcomes"][cp] = {
                    "price": price,
                    "change_pct": round(change_pct, 2),
                    "recorded_at": now_iso,
                }
                log.info(
                    "Outcome %s @%s: price=%.8f change=%.1f%%",
                    address[:8], cp, price, change_pct,
                )
                changed = True

            # Update peak gain
            filled = [
                o for o in rec["outcomes"].values() if o is not None
            ]
            if filled:
                best = max(filled, key=lambda o: o["change_pct"])
                rec["peak_gain_pct"] = round(best["change_pct"], 2)
                # Find which checkpoint had the peak
                for cp_name, o in rec["outcomes"].items():
                    if o is not None and o["change_pct"] == best["change_pct"]:
                        rec["peak_gain_at"] = cp_name
                        break

            # Classify final status if all checkpoints filled
            all_filled = all(rec["outcomes"][cp] is not None for cp in CHECKPOINTS)
            if all_filled:
                rec["final_status"] = _classify_final(rec["outcomes"], alert_price or 0)
                log.info(
                    "Final status for %s: %s (peak=%.1f%%)",
                    address[:8], rec["final_status"], rec.get("peak_gain_pct") or 0,
                )
                changed = True

        if changed:
            _save_json_atomic(OUTCOMES_FILE, self.outcomes)
            log.debug("Saved outcomes.json (%d entries)", len(self.outcomes))

    def get_stats(self) -> dict[str, Any]:
        """Return summary stats for state.json outcome_stats block."""
        counts: dict[str, int] = {}
        for rec in self.outcomes.values():
            status = rec.get("final_status", "pending")
            counts[status] = counts.get(status, 0) + 1

        total = len(self.outcomes)
        pumped_2x = counts.get("pumped_2x", 0)
        pumped_5x = counts.get("pumped_5x", 0)
        pumped_10x = counts.get("pumped_10x", 0)
        modest_gain = counts.get("modest_gain", 0)
        dumped = counts.get("dumped", 0)
        flat = counts.get("flat", 0)
        pending = counts.get("pending", 0)

        # accuracy = any meaningful gain (≥ 2x) out of all resolved
        resolved = total - pending
        gainers = pumped_2x + pumped_5x + pumped_10x
        accuracy_pct = round(gainers / resolved * 100, 1) if resolved > 0 else 0.0

        return {
            "total_tracked": total,
            "pumped_10x": pumped_10x,
            "pumped_5x": pumped_5x,
            "pumped_2x": pumped_2x,
            "modest_gain": modest_gain,
            "dumped": dumped,
            "flat": flat,
            "pending": pending,
            "accuracy_pct": accuracy_pct,
        }

    def to_training_df(self):  # noqa: ANN201
        """Return a DataFrame merging seen.json features with outcome labels."""
        try:
            import pandas as pd
        except ImportError:
            raise ImportError("pandas is required for to_training_df()")

        seen = _load_json(SEEN_FILE)
        rows = []
        for address, rec in self.outcomes.items():
            if rec.get("final_status") == "pending":
                continue
            seen_entry = seen.get(address, {})
            row = {
                "address": address,
                "symbol": rec.get("symbol"),
                "alerted_at": rec.get("alerted_at"),
                "alert_score": rec.get("alert_score"),
                "alert_level": rec.get("alert_level"),
                "price_at_alert": rec.get("price_at_alert"),
                "peak_gain_pct": rec.get("peak_gain_pct"),
                "peak_gain_at": rec.get("peak_gain_at"),
                "final_status": rec.get("final_status"),
                # Seen.json signal features
                "first_seen_price_change_1h": seen_entry.get("first_seen_price_change_1h"),
                "first_seen_vol_liq": seen_entry.get("first_seen_vol_liq"),
                "first_seen_liq": seen_entry.get("first_seen_liq"),
            }
            # Flatten checkpoint outcomes
            for cp in CHECKPOINTS:
                o = (rec.get("outcomes") or {}).get(cp)
                row[f"outcome_{cp}_price"] = o["price"] if o else None
                row[f"outcome_{cp}_change_pct"] = o["change_pct"] if o else None
            rows.append(row)

        return pd.DataFrame(rows)
