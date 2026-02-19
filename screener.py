#!/usr/bin/env python3
"""Solana meme coin screener – polls DEX Screener for early pumps."""

from __future__ import annotations

import argparse
import json
import logging
import signal
import sys
import time
from typing import Any

import requests
from tabulate import tabulate

import config as cfg
from alerts import format_alert, send_telegram
from filters import apply_filters, load_seen, mark_seen, save_seen, should_alert
from legitimacy import analyse as legit_analyse
from signals import score_pair

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.FileHandler(cfg.LOG_FILE),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger("screener")

# ---------------------------------------------------------------------------
# Graceful shutdown
# ---------------------------------------------------------------------------
_shutdown = False


def _handle_signal(signum: int, frame: Any) -> None:
    global _shutdown
    log.info("Shutdown signal received – finishing current cycle…")
    _shutdown = True


signal.signal(signal.SIGINT, _handle_signal)
signal.signal(signal.SIGTERM, _handle_signal)

# ---------------------------------------------------------------------------
# API fetchers
# ---------------------------------------------------------------------------

def _get_json(url: str, retries: int = 2) -> Any:
    """Fetch JSON from *url* with retries and exponential backoff."""
    for attempt in range(retries + 1):
        try:
            resp = requests.get(url, timeout=cfg.REQUEST_TIMEOUT_SECONDS)
            resp.raise_for_status()
            return resp.json()
        except requests.RequestException as exc:
            log.warning("Request to %s failed (attempt %d): %s", url, attempt + 1, exc)
            if attempt < retries:
                time.sleep(2 ** attempt)
    return None


def _extract_solana_addresses_from_boosts(data: list[dict]) -> list[str]:
    """Return Solana token addresses from the boosts/profiles endpoint."""
    addrs: list[str] = []
    for item in data or []:
        if item.get("chainId") == "solana":
            addr = item.get("tokenAddress")
            if addr:
                addrs.append(addr)
    return addrs


def _fetch_pairs_for_addresses(addresses: list[str]) -> list[dict]:
    """Fetch full pair data for a batch of token addresses."""
    pairs: list[dict] = []
    # DEX Screener allows comma-separated addresses (up to ~30)
    batch_size = 30
    for i in range(0, len(addresses), batch_size):
        batch = addresses[i : i + batch_size]
        url = cfg.DEXSCREENER_TOKEN_URL.format(address=",".join(batch))
        data = _get_json(url)
        if data and "pairs" in data:
            pairs.extend(data["pairs"])
    return pairs


def fetch_candidates() -> list[dict[str, Any]]:
    """Gather candidate pairs from multiple DEX Screener endpoints."""
    all_pairs: dict[str, dict] = {}  # keyed by baseToken address

    # 1. Search endpoint (general Solana pairs sorted by creation)
    search_data = _get_json(cfg.DEXSCREENER_SEARCH_URL)
    if search_data and "pairs" in search_data:
        for p in search_data["pairs"]:
            if p.get("chainId") == "solana":
                addr = (p.get("baseToken") or {}).get("address", "")
                if addr:
                    all_pairs[addr] = p

    # 2. Token boosts (newly boosted tokens – often early memes)
    boosts = _get_json(cfg.DEXSCREENER_BOOSTS_URL)
    if isinstance(boosts, list):
        addrs = _extract_solana_addresses_from_boosts(boosts)
        if addrs:
            for p in _fetch_pairs_for_addresses(addrs):
                if p.get("chainId") == "solana":
                    addr = (p.get("baseToken") or {}).get("address", "")
                    if addr and addr not in all_pairs:
                        all_pairs[addr] = p

    # 3. Token profiles (newly profiled tokens)
    profiles = _get_json(cfg.DEXSCREENER_PROFILES_URL)
    if isinstance(profiles, list):
        addrs = _extract_solana_addresses_from_boosts(profiles)  # same shape
        if addrs:
            # Only fetch addresses we don't already have
            new_addrs = [a for a in addrs if a not in all_pairs]
            if new_addrs:
                for p in _fetch_pairs_for_addresses(new_addrs):
                    if p.get("chainId") == "solana":
                        addr = (p.get("baseToken") or {}).get("address", "")
                        if addr and addr not in all_pairs:
                            all_pairs[addr] = p

    log.info("Fetched %d unique Solana candidates", len(all_pairs))
    return list(all_pairs.values())


# ---------------------------------------------------------------------------
# Processing
# ---------------------------------------------------------------------------

def _get_level(score: int) -> tuple[str, str] | None:
    for threshold, level, emoji in cfg.ALERT_LEVELS:
        if score >= threshold:
            return level, emoji
    return None


def process_cycle(dry_run: bool = False) -> None:
    """Run one screening cycle."""
    pairs = fetch_candidates()
    if not pairs:
        log.warning("No candidates fetched this cycle")
        return

    seen = load_seen()
    results: list[tuple[dict, dict]] = []

    for pair in pairs:
        # Hard filters
        rejection = apply_filters(pair)
        if rejection:
            base = (pair.get("baseToken") or {})
            log.debug(
                "Filtered %s (%s): %s",
                base.get("symbol", "?"),
                base.get("address", "?")[:8],
                rejection,
            )
            continue

        # Score
        result = score_pair(pair)
        results.append((pair, result))

        # Alert logic
        score = result["score"]
        level_info = _get_level(score)
        if level_info is None:
            continue

        level, emoji = level_info
        address = (pair.get("baseToken") or {}).get("address", "")
        if not should_alert(seen, address, score, level):
            log.debug("Already alerted %s at same or higher level", address[:8])
            continue

        # Legitimacy check — hard reject blocks the alert entirely
        base_sym = (pair.get("baseToken") or {}).get("symbol", "?")
        log.info("Running legitimacy check for %s (%s)…", base_sym, address[:8])
        legit = legit_analyse(pair)

        if legit and legit.get("hard_reject"):
            log.info(
                "REJECTED %s — legitimacy FAIL: %s",
                base_sym,
                "; ".join(r for r in legit["reasons"] if "🚨" in r),
            )
            mark_seen(seen, address, score, level)  # mark so we don't re-check constantly
            continue

        verdict = (legit or {}).get("verdict", "UNKNOWN")
        log.info("Legitimacy verdict for %s: %s", base_sym, verdict)

        msg = format_alert(pair, result, legit)
        if msg:
            if dry_run:
                print(f"\n{'='*60}")
                print(f"[DRY RUN] Would send alert:")
                print(msg)
                print(f"{'='*60}\n")
            else:
                send_telegram(msg)

            mark_seen(seen, address, score, level)

    save_seen(seen)

    # Console summary table
    results.sort(key=lambda r: r[1]["score"], reverse=True)
    top = results[: cfg.CONSOLE_TOP_N]
    if top:
        table_rows = []
        for pair, result in top:
            base = pair.get("baseToken") or {}
            liq = (pair.get("liquidity") or {}).get("usd", 0) or 0
            mcap = pair.get("marketCap") or pair.get("fdv") or 0
            pc = pair.get("priceChange") or {}
            age = result.get("age_minutes")
            age_s = f"{age:.0f}m" if age is not None else "?"
            level_info = _get_level(result["score"])
            tag = level_info[0] if level_info else "-"
            table_rows.append([
                base.get("symbol", "?"),
                result["score"],
                tag,
                age_s,
                f"${liq:,.0f}",
                f"${mcap:,.0f}",
                f"{pc.get('m5', 0) or 0:+.1f}%",
                f"{pc.get('h1', 0) or 0:+.1f}%",
            ])

        print(f"\n--- Top {len(table_rows)} Candidates ---")
        print(tabulate(
            table_rows,
            headers=["Symbol", "Score", "Level", "Age", "Liq", "MCap", "5m%", "1h%"],
            tablefmt="simple",
        ))
        print()
    else:
        print("\nNo candidates passed filters this cycle.\n")

    log.info(
        "Cycle complete: %d candidates, %d passed filters, top score %d",
        len(pairs),
        len(results),
        results[0][1]["score"] if results else 0,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Solana meme coin screener")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print alerts to console instead of sending to Telegram",
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Run a single cycle and exit",
    )
    args = parser.parse_args()

    mode = "DRY RUN" if args.dry_run else "LIVE"
    log.info("Starting meme screener in %s mode", mode)

    while not _shutdown:
        try:
            process_cycle(dry_run=args.dry_run)
        except Exception:
            log.exception("Error during cycle")

        if args.once:
            break

        log.info("Sleeping %ds until next cycle…", cfg.POLL_INTERVAL_SECONDS)
        # Sleep in small increments so Ctrl+C is responsive
        for _ in range(cfg.POLL_INTERVAL_SECONDS):
            if _shutdown:
                break
            time.sleep(1)

    log.info("Screener stopped.")


if __name__ == "__main__":
    main()
