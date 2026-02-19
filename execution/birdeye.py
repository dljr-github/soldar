"""
Birdeye API integration for richer Solana token holder data.

Free tier provides holder counts and top holder details.
Used as optional enrichment in legitimacy analysis when BIRDEYE_API_KEY is set.
"""

from __future__ import annotations

import logging
import os
from typing import Any

import requests
from dotenv import load_dotenv

load_dotenv()

log = logging.getLogger("birdeye")

BASE_URL = "https://public-api.birdeye.so"
API_KEY = os.getenv("BIRDEYE_API_KEY", "")

_HEADERS = {
    "accept": "application/json",
    "x-chain": "solana",
}


def _get(endpoint: str, params: dict | None = None, timeout: int = 10) -> dict | None:
    """Make an authenticated GET request to Birdeye API.

    Returns None if API key is missing or request fails (never crashes).
    """
    if not API_KEY:
        return None

    headers = {**_HEADERS, "X-API-KEY": API_KEY}
    url = f"{BASE_URL}{endpoint}"
    try:
        resp = requests.get(url, headers=headers, params=params, timeout=timeout)
        resp.raise_for_status()
        data = resp.json()
        if data.get("success") is False:
            log.warning("Birdeye API error: %s", data.get("message", "unknown"))
            return None
        return data.get("data", data)
    except requests.RequestException as exc:
        log.warning("Birdeye request failed (%s): %s", endpoint, exc)
        return None


def get_token_overview(address: str) -> dict[str, Any] | None:
    """Fetch token overview: price, volume24h, liquidity, market cap.

    Returns None if API key is not set or request fails.
    """
    data = _get(f"/defi/token_overview", params={"address": address})
    if data is None:
        return None

    return {
        "price": data.get("price"),
        "volume24h": data.get("v24hUSD"),
        "liquidity": data.get("liquidity"),
        "mc": data.get("mc"),
        "holder_count": data.get("holder"),
    }


def get_token_holders(address: str) -> dict[str, Any] | None:
    """Fetch holder count and top holders for a token.

    Returns None if API key is not set or request fails.
    """
    # Token overview gives holder count
    overview = _get(f"/defi/token_overview", params={"address": address})
    if overview is None:
        return None

    holder_count = overview.get("holder", 0)

    # Top holders endpoint (if available on free tier)
    top_data = _get(
        f"/defi/v2/tokens/top_traders",
        params={"address": address, "time_frame": "24h"},
    )
    top_holders = []
    if top_data and isinstance(top_data, dict):
        items = top_data.get("items", [])
        for item in items[:10]:
            top_holders.append({
                "address": item.get("address", ""),
                "balance_usd": item.get("volume_buy_usd", 0),
            })

    return {
        "holder_count": holder_count,
        "top_holders": top_holders,
    }


def is_available() -> bool:
    """Check if Birdeye API key is configured."""
    return bool(API_KEY)
