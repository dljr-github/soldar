"""
On-demand "what's hot?" report generator.

Reads data/state.json and returns a formatted Telegram message
showing current top candidates with scores and legit status.

Usage:
    python -m ml.report          # Print to console
    python -m ml.report --send   # Print + send to Telegram
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from typing import Any

# Add project root to path for imports
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from alerts import send_telegram

STATE_FILE = os.path.join(_project_root, "data", "state.json")


def _load_state() -> dict:
    """Load the latest state.json."""
    try:
        with open(STATE_FILE) as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def _age_str(age_min: float) -> str:
    if age_min < 60:
        return f"{age_min:.0f}m"
    hours = age_min / 60
    if hours < 24:
        return f"{hours:.1f}h"
    return f"{hours / 24:.1f}d"


def _shorten_address(addr: str) -> str:
    if len(addr) > 10:
        return f"{addr[:6]}...{addr[-4:]}"
    return addr


def generate_hot_report() -> str:
    """Generate a formatted Telegram-ready report of current top candidates."""
    state = _load_state()
    if not state:
        return "No state data found. Is the screener running?"

    candidates = state.get("candidates", [])
    last_updated = state.get("last_updated", "")
    cycle_count = state.get("cycle_count", 0)
    scanned = state.get("candidates_scanned", 0)

    # Parse last_updated for display
    try:
        dt = datetime.fromisoformat(last_updated)
        time_str = dt.strftime("%I:%M %p")
        # Time since last scan
        age_sec = (datetime.now(timezone.utc) - dt).total_seconds()
        if age_sec < 120:
            ago_str = f"{age_sec:.0f}s ago"
        else:
            ago_str = f"{age_sec / 60:.0f}m ago"
    except (ValueError, TypeError):
        time_str = "?"
        ago_str = "?"

    # Group by level
    hot = [c for c in candidates if c.get("level") == "HOT"]
    warm = [c for c in candidates if c.get("level") == "WARM"]
    watch = [c for c in candidates if c.get("level") == "WATCH"]

    lines = [f"\U0001f6f0 SOLDAR REPORT \u2014 {time_str}"]
    lines.append("")

    def _format_candidate(c: dict[str, Any]) -> list[str]:
        symbol = c.get("symbol", "?")
        name = c.get("name", "?")
        score = c.get("score", 0)
        age = c.get("age_minutes", 0)
        mcap = c.get("mcap_usd", 0)
        p5m = c.get("price_change_5m", 0)
        p1h = c.get("price_change_1h", 0)
        addr = c.get("address", "")
        url = c.get("url", "")
        verdict = c.get("legit_verdict")
        source = c.get("source", "")

        source_tag = " \U0001f525" if source == "pumpfun" else ""
        verdict_str = {
            "PASS": "\u2705 Legit",
            "WARN": "\u26a0\ufe0f Caution",
            "FAIL": "\u274c Risky",
        }.get(verdict or "", "\u2753 Unchecked")

        entry_lines = [
            f"\u2022 {name} (${symbol}){source_tag} \u2014 {score}/100",
            f"  Age: {_age_str(age)} | MCap: ${mcap:,.0f} | 5m: {p5m:+.1f}% | 1h: {p1h:+.1f}%",
            f"  {verdict_str} | CA: {_shorten_address(addr)}",
        ]
        if url:
            entry_lines.append(f"  \U0001f517 {url}")
        return entry_lines

    if hot:
        lines.append(f"\U0001f534 HOT ({len(hot)})")
        for c in hot[:5]:
            lines.extend(_format_candidate(c))
            lines.append("")
    else:
        lines.append("\U0001f534 HOT \u2014 none right now")
        lines.append("")

    if warm:
        lines.append(f"\U0001f7e0 WARM ({len(warm)})")
        for c in warm[:5]:
            lines.extend(_format_candidate(c))
            lines.append("")
    else:
        lines.append("\U0001f7e0 WARM \u2014 none right now")
        lines.append("")

    if watch:
        lines.append(f"\U0001f7e1 WATCH ({len(watch)})")
        for c in watch[:5]:
            lines.extend(_format_candidate(c))
            lines.append("")
    else:
        lines.append("\U0001f7e1 WATCH \u2014 none right now")
        lines.append("")

    # Pump.fun section if any
    pf_candidates = [c for c in candidates if c.get("source") == "pumpfun"]
    if pf_candidates:
        lines.append(f"\U0001f525 PUMP.FUN ({len(pf_candidates)})")
        for c in pf_candidates[:3]:
            feats = c.get("pumpfun_features", {})
            lines.append(
                f"\u2022 {c.get('name','?')} (${c.get('symbol','?')}) \u2014 {c.get('score',0)}/100"
            )
            lines.append(
                f"  Curve: {feats.get('bonding_curve_pct',0):.0f}% | "
                f"Vel: {feats.get('trade_velocity',0):.0f}/min | "
                f"Buy%: {feats.get('buy_pressure',0)*100:.0f}%"
            )
            lines.append("")

    # Exit alerts
    exits = state.get("exit_alerts", [])
    if exits:
        lines.append(f"\u26a0\ufe0f EXIT SIGNALS ({len(exits)})")
        for e in exits[:3]:
            lines.append(
                f"\u2022 {e.get('symbol','?')} \u2014 {e.get('exit_reason','?')} [{e.get('urgency','?')}]"
            )
        lines.append("")

    # Footer
    passed = state.get("passed_filters", 0)
    lines.append(f"\U0001f4ca Scanned {scanned} tokens this cycle ({passed} passed). Last scan: {ago_str}.")

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Soldar on-demand hot report")
    parser.add_argument(
        "--send",
        action="store_true",
        help="Send report to Telegram in addition to printing",
    )
    args = parser.parse_args()

    report = generate_hot_report()
    print(report)

    if args.send:
        ok = send_telegram(report)
        if ok:
            print("\n--- Sent to Telegram ---")
        else:
            print("\n--- Failed to send to Telegram ---")


if __name__ == "__main__":
    main()
