#!/usr/bin/env python3
"""Collect historical Solana meme coin pool data from GeckoTerminal."""

from __future__ import annotations

import argparse
import json
import os
import time
from datetime import datetime, timezone

import requests
from tqdm import tqdm

BASE = "https://api.geckoterminal.com/api/v2"

STABLECOIN_SYMBOLS = {
    "USDC", "USDT", "BUSD", "DAI", "TUSD", "FRAX", "USDP", "GUSD",
    "PYUSD", "USDD", "LUSD", "CRVUSD", "GHO", "EURC",
}


def api_get(url: str, params: dict | None = None, delay: float = 2.0) -> dict | None:
    """GET with rate-limit delay. Returns JSON or None on failure."""
    max_retries = 3
    for attempt in range(max_retries):
        try:
            resp = requests.get(url, params=params, timeout=30)
            if resp.status_code == 429:
                wait = int(resp.headers.get("Retry-After", 60))
                tqdm.write(f"  Rate limited (HTTP 429) – waiting {wait}s")
                time.sleep(wait)
                continue
            resp.raise_for_status()
            data = resp.json()
            # GeckoTerminal sometimes returns 200 with a 429 error in the body
            if isinstance(data, dict) and "status" in data:
                err = data["status"]
                if isinstance(err, dict) and err.get("error_code") == 429:
                    tqdm.write(f"  Rate limited (body 429) – waiting 60s")
                    time.sleep(60)
                    continue
            return data
        except requests.RequestException as e:
            tqdm.write(f"  API error: {e}")
            return None
        finally:
            time.sleep(delay)
    tqdm.write("  Max retries reached, skipping.")
    return None


def fetch_pools(pages: int, delay: float) -> list[dict]:
    """Fetch new pool listings across N pages."""
    pools = []
    for page in tqdm(range(1, pages + 1), desc="Fetching pool pages"):
        url = f"{BASE}/networks/solana/new_pools"
        data = api_get(url, params={"page": page}, delay=delay)
        if not data or "data" not in data:
            tqdm.write(f"  No data on page {page}, stopping.")
            break
        for pool in data["data"]:
            pools.append(pool)
    return pools


def filter_pools(pools: list[dict], max_age_days: int = 60) -> list[dict]:
    """Keep Solana pools created within max_age_days, skip stablecoin bases."""
    cutoff = datetime.now(timezone.utc).timestamp() - max_age_days * 86400
    kept = []
    for p in pools:
        attrs = p.get("attributes", {})
        created = attrs.get("pool_created_at")
        if not created:
            continue
        try:
            ts = datetime.fromisoformat(created.replace("Z", "+00:00")).timestamp()
        except (ValueError, TypeError):
            continue
        if ts < cutoff:
            continue
        # Skip stablecoin base tokens
        name = (attrs.get("name") or "").upper()
        base_symbol = name.split("/")[0].strip() if "/" in name else ""
        if base_symbol in STABLECOIN_SYMBOLS:
            continue
        kept.append(p)
    return kept


def fetch_ohlcv(pool_address: str, delay: float) -> list | None:
    """Fetch minute OHLCV candles for a pool."""
    url = f"{BASE}/networks/solana/pools/{pool_address}/ohlcv/minute"
    data = api_get(url, params={"limit": 1000, "aggregate": 1}, delay=delay)
    if not data:
        return None
    attrs = data.get("data", {}).get("attributes", {})
    return attrs.get("ohlcv_list")


def fetch_pool_details(pool_address: str, delay: float) -> dict | None:
    """Fetch detailed pool info."""
    url = f"{BASE}/networks/solana/pools/{pool_address}"
    data = api_get(url, delay=delay)
    if not data:
        return None
    return data.get("data", {}).get("attributes", {})


def collect(pages: int, out_dir: str, delay: float) -> None:
    os.makedirs(out_dir, exist_ok=True)
    manifest_path = os.path.join(out_dir, "manifest.json")
    manifest: dict = {}
    if os.path.exists(manifest_path):
        with open(manifest_path) as f:
            manifest = json.load(f)

    # Step 1: fetch pool list
    raw_pools = fetch_pools(pages, delay)
    pools = filter_pools(raw_pools)
    tqdm.write(f"Found {len(pools)} pools after filtering ({len(raw_pools)} raw)")

    # Step 2: collect data for each pool
    for p in tqdm(pools, desc="Collecting pool data"):
        pool_address = p.get("attributes", {}).get("address") or p.get("id", "").split("_")[-1]
        if not pool_address:
            continue

        out_file = os.path.join(out_dir, f"{pool_address}.json")
        if os.path.exists(out_file):
            continue

        attrs = p.get("attributes", {})
        name = attrs.get("name", "")
        base_symbol = name.split("/")[0].strip() if "/" in name else name

        # Fetch OHLCV
        ohlcv = fetch_ohlcv(pool_address, delay)
        if not ohlcv:
            tqdm.write(f"  Skipping {base_symbol} – no OHLCV data")
            continue

        # Fetch pool details
        details = fetch_pool_details(pool_address, delay)

        record = {
            "name": name,
            "symbol": base_symbol,
            "base_address": (
                p.get("relationships", {})
                .get("base_token", {})
                .get("data", {})
                .get("id", "")
                .replace("solana_", "")
            ),
            "pool_address": pool_address,
            "created_at": attrs.get("pool_created_at"),
            "ohlcv": ohlcv,
            "pool_details": details,
        }

        with open(out_file, "w") as f:
            json.dump(record, f)

        manifest[pool_address] = {
            "name": name,
            "symbol": base_symbol,
            "created_at": attrs.get("pool_created_at"),
            "status": "collected",
        }

        # Save manifest incrementally
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)

    print(f"\nDone. {len(manifest)} pools in manifest at {manifest_path}")


def main():
    parser = argparse.ArgumentParser(description="Collect meme coin pool data from GeckoTerminal")
    parser.add_argument("--pages", type=int, default=50, help="Number of new_pools pages to fetch")
    parser.add_argument("--out", type=str, default="ml/raw/", help="Output directory for raw JSON")
    parser.add_argument("--delay", type=float, default=2.0, help="Seconds between API calls")
    args = parser.parse_args()
    collect(args.pages, args.out, args.delay)


if __name__ == "__main__":
    main()
