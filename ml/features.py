#!/usr/bin/env python3
"""Extract ML features from raw pool data."""

from __future__ import annotations


def _safe_float(val, default=None):
    """Convert to float, returning default if not possible."""
    if val is None:
        return default
    try:
        return float(val)
    except (ValueError, TypeError):
        return default


def extract_features(pool_data: dict) -> list[dict]:
    """Extract features at multiple detection points from a single pool's data.

    Returns a list of feature dicts, one per detection point T.
    """
    ohlcv = pool_data.get("ohlcv") or []
    details = pool_data.get("pool_details") or {}

    if len(ohlcv) < 5:
        return []

    # OHLCV is [timestamp, open, high, low, close, volume_usd]
    # Sort by timestamp ascending (earliest first)
    ohlcv = sorted(ohlcv, key=lambda c: c[0])

    open_price = _safe_float(ohlcv[0][1])
    if not open_price or open_price <= 0:
        return []

    survived_30min = len(ohlcv) >= 30

    # Pool-level features
    initial_liquidity = _safe_float(details.get("reserve_in_usd"))
    mcap = _safe_float(details.get("fdv_usd")) or _safe_float(details.get("market_cap_usd"))
    liq_to_mcap = None
    if initial_liquidity and mcap and mcap > 0:
        liq_to_mcap = initial_liquidity / mcap

    symbol = pool_data.get("symbol", "")
    pool_address = pool_data.get("pool_address", "")

    rows = []
    for T in [1, 3, 5, 10, 15]:
        if len(ohlcv) < T:
            continue

        candles_t = ohlcv[:T]
        price_at_t = _safe_float(candles_t[-1][4])
        if not price_at_t or price_at_t <= 0:
            continue

        # Volume per minute (bounds-check candle length)
        vols = [_safe_float(c[5] if len(c) > 5 else None, 0) for c in candles_t]
        vol_per_min = sum(vols) / T

        # Price velocity: rate of change over last 2 candles before T
        if T >= 2:
            c_prev = _safe_float(ohlcv[T - 2][4], price_at_t)
            price_velocity = (price_at_t - c_prev) / c_prev * 100 if c_prev else 0
        else:
            price_velocity = 0

        # High-low range
        highs = [_safe_float(c[2], 0) for c in candles_t]
        lows = [_safe_float(c[3], 0) for c in candles_t if _safe_float(c[3], 0) > 0]
        if lows and min(lows) > 0:
            hl_range_pct = (max(highs) - min(lows)) / min(lows) * 100
        else:
            hl_range_pct = 0

        # Candles up percentage
        ups = sum(1 for c in candles_t if _safe_float(c[4], 0) > _safe_float(c[1], 0))
        candles_up_pct = ups / T

        # Labels computed FORWARD from detection point T (not from candle 0)
        horizon_1h = ohlcv[T:T + 60]
        horizon_2h = ohlcv[T:T + 120]
        closes_1h = [_safe_float(c[4]) for c in horizon_1h if _safe_float(c[4])]
        closes_2h = [_safe_float(c[4]) for c in horizon_2h if _safe_float(c[4])]
        max_close_1h = max(closes_1h) if closes_1h else price_at_t
        max_close_2h = max(closes_2h) if closes_2h else price_at_t
        max_gain_1h_pct = (max_close_1h / price_at_t - 1) * 100
        max_gain_2h_pct = (max_close_2h / price_at_t - 1) * 100

        if closes_2h:
            peak_idx = closes_2h.index(max(closes_2h))
            time_to_peak_min = peak_idx + 1  # minutes after detection
        else:
            time_to_peak_min = None

        rows.append({
            "symbol": symbol,
            "pool_address": pool_address,
            "detection_minute": T,
            "open_price": open_price,
            "price_at_T": price_at_t,
            "price_change_T_pct": (price_at_t / open_price - 1) * 100,
            "vol_per_min_T": vol_per_min,
            "price_velocity": price_velocity,
            "high_low_range_pct": hl_range_pct,
            "candles_up_pct": candles_up_pct,
            "initial_liquidity_usd": initial_liquidity,
            "mcap_at_listing": mcap,
            "liq_to_mcap_ratio": liq_to_mcap,
            # Labels — computed forward from T
            "max_gain_1h_pct": max_gain_1h_pct,
            "max_gain_2h_pct": max_gain_2h_pct,
            "pumped_2x": max_gain_1h_pct >= 100,
            "pumped_5x": max_gain_1h_pct >= 400,
            "pumped_10x": max_gain_2h_pct >= 900,
            "time_to_peak_min": time_to_peak_min,
            "survived_30min": survived_30min,
        })

    return rows
