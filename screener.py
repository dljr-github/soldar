#!/usr/bin/env python3
"""Solana meme coin screener – polls DEX Screener for early pumps."""

from __future__ import annotations

import argparse
import json
import logging
import os
import signal
import sys
import tempfile
import time
from datetime import datetime, timezone
from typing import Any

import requests
from tabulate import tabulate

import config as cfg
from alerts import format_alert, send_telegram
from execution.paper_trader import PaperTrader
from execution.position_manager import PositionManager
from filters import apply_filters, load_seen, mark_seen, save_seen, should_alert
from legitimacy import analyse as legit_analyse
from ml.outcome_tracker import OutcomeTracker
from signals import _get_vol_liq_ratio, score_exit, score_pair

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
# Paper / live trading
# ---------------------------------------------------------------------------
_position_manager = PositionManager()
_paper_trader = PaperTrader() if cfg.PAPER_TRADING_ENABLED else None

# ---------------------------------------------------------------------------
# Outcome tracker (ML training labels)
# ---------------------------------------------------------------------------
_outcome_tracker = OutcomeTracker()

# ---------------------------------------------------------------------------
# State file for dashboard
# ---------------------------------------------------------------------------
STATE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
STATE_FILE = os.path.join(STATE_DIR, "state.json")
_cycle_count = 0

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
# State file writer
# ---------------------------------------------------------------------------

def _load_prior_state() -> dict:
    """Load existing state.json to preserve rolling history."""
    try:
        with open(STATE_FILE) as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def _write_state(
    pairs_total: int,
    results: list[tuple[dict, dict]],
    hard_rejected: list[dict],
    exit_alerts: list[dict] | None = None,
    outcome_stats: dict | None = None,
) -> None:
    """Atomically write data/state.json with current cycle data."""
    global _cycle_count
    _cycle_count += 1
    now = datetime.now(timezone.utc).isoformat()

    prior = _load_prior_state()
    cycle_history = prior.get("cycle_history", [])
    prior_rejected = prior.get("hard_rejected", [])

    # Build candidate list from scored results
    prior_candidates = {
        c["address"]: c for c in prior.get("candidates", [])
    }
    candidates = []
    for pair, result in results:
        base = pair.get("baseToken") or {}
        liq = (pair.get("liquidity") or {}).get("usd", 0) or 0
        mcap = pair.get("marketCap") or pair.get("fdv") or 0
        pc = pair.get("priceChange") or {}
        txns_5m = (pair.get("txns") or {}).get("m5") or {}
        vol = pair.get("volume") or {}
        vol_liq = round(vol.get("h1", 0) / liq, 1) if liq else 0

        level_info = _get_level(result["score"])
        level = level_info[0] if level_info else None

        # Legitimacy data stored during cycle
        legit = result.get("legit") or {}
        details = legit.get("details") or {}
        social = details.get("social") or {}
        socials = [k for k in ("website", "twitter", "telegram") if social.get(k)]

        # first_seen: use prior state if available, else now
        addr = base.get("address", "")
        first_seen = prior_candidates.get(addr, {}).get("first_seen", now)

        candidates.append({
            "symbol": base.get("symbol", "?"),
            "name": base.get("name", "?"),
            "address": addr,
            "score": result["score"],
            "level": level,
            "age_minutes": round(result.get("age_minutes") or 0, 1),
            "liquidity_usd": round(liq, 2),
            "mcap_usd": round(mcap, 2),
            "vol_liq_ratio": vol_liq,
            "price_change_5m": round(pc.get("m5", 0) or 0, 1),
            "price_change_1h": round(pc.get("h1", 0) or 0, 1),
            "price_change_6h": round(pc.get("h6", 0) or 0, 1),
            "buys_5m": txns_5m.get("buys", 0) or 0,
            "sells_5m": txns_5m.get("sells", 0) or 0,
            "dex": pair.get("dexId", ""),
            "url": pair.get("url", ""),
            "legit_verdict": legit.get("verdict"),
            "legit_hard_rejected": legit.get("hard_reject", False),
            "legit_top1_pct": details.get("top1_pct"),
            "legit_lp_locked_pct": details.get("lp_locked_pct"),
            "legit_rc_score": details.get("rc_score"),
            "legit_mint_revoked": details.get("mint_revoked"),
            "legit_freeze_revoked": details.get("freeze_revoked"),
            "legit_socials": socials,
            "legit_reasons": legit.get("reasons", []),
            "first_seen": first_seen,
        })

    candidates.sort(key=lambda c: c["score"], reverse=True)
    top_score = candidates[0]["score"] if candidates else 0

    # Append to cycle history (rolling last 100)
    cycle_history.append({
        "timestamp": now,
        "scanned": pairs_total,
        "passed": len(results),
        "top_score": top_score,
    })
    cycle_history = cycle_history[-100:]

    # Merge hard_rejected (rolling last 50)
    all_rejected = prior_rejected + hard_rejected
    all_rejected = all_rejected[-50:]

    # Merge exit alerts — keep from prior state if no new ones this cycle
    from datetime import timedelta
    current_exits = exit_alerts if exit_alerts else []
    prior_exits = prior.get("exit_alerts", [])
    # Replace with current cycle's alerts; keep prior ones that aren't superseded
    current_addrs = {e["address"] for e in current_exits}
    merged_exits = current_exits + [
        e for e in prior_exits if e["address"] not in current_addrs
    ]
    # Only keep exit alerts from the last 4 hours
    cutoff = (datetime.now(timezone.utc) - timedelta(hours=4)).isoformat()
    merged_exits = [
        e for e in merged_exits
        if e.get("detected_at", "") > cutoff
    ]

    state = {
        "last_updated": now,
        "cycle_count": _cycle_count,
        "candidates_scanned": pairs_total,
        "passed_filters": len(results),
        "top_score": top_score,
        "candidates": candidates,
        "hard_rejected": all_rejected,
        "exit_alerts": merged_exits,
        "cycle_history": cycle_history,
        "outcome_stats": outcome_stats or {},
    }

    os.makedirs(STATE_DIR, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(dir=STATE_DIR, suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(state, f, indent=2)
        os.replace(tmp_path, STATE_FILE)
        log.info("Wrote %s (%d candidates)", STATE_FILE, len(candidates))
    except Exception:
        log.exception("Failed to write state file")
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


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
    hard_rejected: list[dict] = []

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
            hard_rejected.append({
                "symbol": base_sym,
                "address": address,
                "reasons": [r for r in legit["reasons"] if "🚨" in r],
                "timestamp": datetime.now(timezone.utc).isoformat(),
            })
            mark_seen(seen, address, score, level)  # mark so we don't re-check constantly
            continue

        # Attach legit data to result for state.json
        result["legit"] = legit

        verdict = (legit or {}).get("verdict", "UNKNOWN")
        log.info("Legitimacy verdict for %s: %s", base_sym, verdict)

        # Compute first-seen values for exit signal tracking
        pc = pair.get("priceChange") or {}
        fs_p1h = pc.get("h1", 0) or 0
        fs_vliq = _get_vol_liq_ratio(pair)
        fs_liq = (pair.get("liquidity") or {}).get("usd", 0) or 0

        msg = format_alert(pair, result, legit)
        if msg:
            if dry_run:
                print(f"\n{'='*60}")
                print(f"[DRY RUN] Would send alert:")
                print(msg)
                print(f"{'='*60}\n")
                mark_seen(seen, address, score, level,
                          first_seen_price_change_1h=fs_p1h,
                          first_seen_vol_liq=fs_vliq,
                          first_seen_liq=fs_liq)
            elif cfg.AUTO_ALERTS_ENABLED:
                send_telegram(msg)
                mark_seen(seen, address, score, level,
                          first_seen_price_change_1h=fs_p1h,
                          first_seen_vol_liq=fs_vliq,
                          first_seen_liq=fs_liq)
            else:
                log.info("AUTO_ALERTS_ENABLED=False — logged %s (score=%d, verdict=%s), no Telegram sent", base_sym, score, verdict)
                mark_seen(seen, address, score, level,
                          first_seen_price_change_1h=fs_p1h,
                          first_seen_vol_liq=fs_vliq,
                          first_seen_liq=fs_liq)

        # --- Paper trading trigger ---
        if (
            score >= cfg.MIN_SCORE_TO_TRADE
            and legit is not None
            and not legit.get("hard_reject")
            and cfg.PAPER_TRADING_ENABLED
            and _paper_trader is not None
        ):
            can_open, reason = _position_manager.can_open_position(cfg.MAX_POSITION_SIZE_USD)
            if can_open:
                pool_addr = pair.get("pairAddress", "")
                buy_result = _paper_trader.simulate_buy(
                    symbol=base_sym,
                    token_mint=address,
                    pool_address=pool_addr,
                    amount_usd=cfg.MAX_POSITION_SIZE_USD,
                    score=score,
                    legit_verdict=verdict,
                )
                if buy_result["success"]:
                    log.info(
                        "PAPER TRADE OPENED: %s score=%d pos=#%d @ $%s (slippage %.1f%%)",
                        base_sym, score, buy_result["position_id"],
                        buy_result["fill_price"], buy_result["slippage_pct"],
                    )
                else:
                    log.warning("Paper buy failed for %s: %s", base_sym, buy_result.get("error"))
            else:
                log.info("Cannot open position for %s: %s", base_sym, reason)

    # --- Update open position prices and check stop losses ---
    if _paper_trader is not None:
        _paper_trader.update_all_prices()
        triggered = _position_manager.check_stop_losses()
        for pos in triggered:
            sell_result = _paper_trader.simulate_sell(pos["id"], reason="stop_loss")
            if sell_result["success"]:
                log.warning(
                    "STOP LOSS CLOSED #%d %s: PnL $%.2f (%.1f%%)",
                    pos["id"], pos["symbol"], sell_result["pnl_usd"], sell_result["pnl_pct"],
                )

    # --- Exit signal detection for previously-alerted coins ---
    exit_alerts: list[dict] = []
    # Build a lookup of current-cycle pair data by address
    pair_by_addr: dict[str, dict] = {}
    for pair, _result in results:
        addr = (pair.get("baseToken") or {}).get("address", "")
        if addr:
            pair_by_addr[addr] = pair

    for addr, entry in seen.items():
        # Only check coins that were actually alerted (have an alert_level)
        if not entry.get("alert_level"):
            continue
        # Need current pair data from this cycle
        pair = pair_by_addr.get(addr)
        if pair is None:
            continue
        exit_result = score_exit(pair, entry)
        if exit_result["should_exit"]:
            base = pair.get("baseToken") or {}
            pc = pair.get("priceChange") or {}
            symbol = base.get("symbol", "?")
            log.warning(
                "EXIT SIGNAL %s (%s): %s [%s] — signals: %s",
                symbol, addr[:8],
                exit_result["exit_reason"],
                exit_result["urgency"],
                ", ".join(exit_result["exit_signals"]),
            )
            exit_alerts.append({
                "symbol": symbol,
                "address": addr,
                "exit_reason": exit_result["exit_reason"],
                "urgency": exit_result["urgency"],
                "all_signals": exit_result["exit_signals"],
                "current_price_change_5m": round(pc.get("m5", 0) or 0, 1),
                "current_price_change_1h": round(pc.get("h1", 0) or 0, 1),
                "detected_at": datetime.now(timezone.utc).isoformat(),
                "alerted_at": datetime.fromtimestamp(
                    entry.get("alerted_at", 0), tz=timezone.utc
                ).isoformat(),
            })

    save_seen(seen)

    # --- Outcome tracking (ML training labels) ---
    try:
        _outcome_tracker.run_checkpoint()
        outcome_stats = _outcome_tracker.get_stats()
    except Exception:
        log.exception("Outcome tracker error (non-fatal)")
        outcome_stats = {}

    # Write dashboard state file
    _write_state(len(pairs), results, hard_rejected, exit_alerts, outcome_stats)

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
