#!/usr/bin/env python3
"""
Run trained model on live screener candidates.
Called by screener.py to enhance scoring with ML prediction.
"""

from __future__ import annotations

import logging
import os
import pickle
import time

import numpy as np
import pandas as pd

log = logging.getLogger("ml.predict")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
XGB_MODEL_PATH = os.path.join(MODEL_DIR, "xgb_baseline.pkl")

# Same feature columns as training
FEATURE_COLS = [
    "price_change_T_pct",
    "vol_per_min_T",
    "price_velocity",
    "high_low_range_pct",
    "candles_up_pct",
    "initial_liquidity_usd",
    "mcap_at_listing",
    "liq_to_mcap_ratio",
]

# Cached model (loaded once, reused across calls)
_cached_model = None
_cached_model_mtime = 0.0


def _load_model():
    """Load the XGBoost model, caching it for reuse. Reloads if file changed."""
    global _cached_model, _cached_model_mtime

    if not os.path.exists(XGB_MODEL_PATH):
        return None

    mtime = os.path.getmtime(XGB_MODEL_PATH)
    if _cached_model is not None and mtime == _cached_model_mtime:
        return _cached_model

    try:
        with open(XGB_MODEL_PATH, "rb") as f:
            data = pickle.load(f)

        # Handle both old format (bare model) and new format (dict with metadata)
        if isinstance(data, dict):
            _cached_model = data["model"]
        else:
            _cached_model = data

        _cached_model_mtime = mtime
        log.info("Loaded XGBoost model from %s", XGB_MODEL_PATH)
        return _cached_model
    except Exception as e:
        log.warning("Failed to load model: %s", e)
        return None


def _extract_features_from_pair(pair_data: dict) -> dict:
    """Extract ML features from a live DexScreener pair dict.

    Maps live DEX Screener data fields to the same feature space
    used during training (from ml/features.py).
    """
    pc = pair_data.get("priceChange") or {}
    vol = pair_data.get("volume") or {}
    liq_data = pair_data.get("liquidity") or {}
    txns = pair_data.get("txns") or {}
    txns_5m = txns.get("m5") or {}

    # Liquidity and market cap
    liq_usd = liq_data.get("usd") or 0
    mcap = pair_data.get("marketCap") or pair_data.get("fdv") or 0

    # price_change_T_pct: use 5m price change as proxy for T=5min detection
    price_change = pc.get("m5") or 0

    # vol_per_min_T: volume in last 5 minutes / 5
    vol_5m = vol.get("m5") or 0
    if vol_5m <= 0:
        vol_h1 = vol.get("h1") or 0
        vol_5m = vol_h1 / 12  # Estimate 5m from 1h
    vol_per_min = vol_5m / 5 if vol_5m > 0 else 0

    # price_velocity: approximate from 5m vs 1h changes
    p5m = pc.get("m5") or 0
    p1h = pc.get("h1") or 0
    # Velocity = recent acceleration (5m change is more recent than 1h trend)
    price_velocity = p5m - (p1h / 12) if p1h else p5m

    # high_low_range_pct: approximate from available price change data
    # In live data we don't have exact H/L, but can estimate from price changes
    p6h = abs(pc.get("h6") or pc.get("h1") or 0)
    high_low_range = max(abs(p5m), p6h) * 1.5  # Rough estimate

    # candles_up_pct: use buy/sell ratio as proxy
    buys = txns_5m.get("buys") or 0
    sells = txns_5m.get("sells") or 0
    total_txns = buys + sells
    candles_up_pct = buys / total_txns if total_txns > 0 else 0.5

    # liq_to_mcap_ratio
    liq_to_mcap = liq_usd / mcap if mcap > 0 else 0

    return {
        "price_change_T_pct": price_change,
        "vol_per_min_T": vol_per_min,
        "price_velocity": price_velocity,
        "high_low_range_pct": high_low_range,
        "candles_up_pct": candles_up_pct,
        "initial_liquidity_usd": liq_usd,
        "mcap_at_listing": mcap,
        "liq_to_mcap_ratio": liq_to_mcap,
    }


def predict_pump_probability(pair_data: dict) -> float:
    """
    Given a pair dict from DexScreener, return probability of 2x pump.
    Uses XGBoost model if available, else returns -1.0 (model not trained).

    Args:
        pair_data: DexScreener pair dict with priceChange, volume, liquidity, txns

    Returns:
        float: 0.0-1.0 pump probability, or -1.0 if model not available
    """
    model = _load_model()
    if model is None:
        return -1.0

    try:
        features = _extract_features_from_pair(pair_data)
        X = pd.DataFrame([features])[FEATURE_COLS]

        # Convert to numeric and handle NaN
        for col in X.columns:
            X[col] = pd.to_numeric(X[col], errors="coerce")
        X = X.fillna(0)

        proba = model.predict_proba(X)[0, 1]
        return float(proba)
    except Exception as e:
        log.warning("Prediction failed: %s", e)
        return -1.0


if __name__ == "__main__":
    # Quick test with sample data
    sample_pair = {
        "priceChange": {"m5": 25.3, "h1": 120.5, "h6": 450.0},
        "volume": {"m5": 85000, "h1": 320000},
        "liquidity": {"usd": 45000},
        "txns": {"m5": {"buys": 95, "sells": 25}},
        "marketCap": 250000,
    }

    prob = predict_pump_probability(sample_pair)
    if prob < 0:
        print(f"Model not loaded (no trained model at {XGB_MODEL_PATH})")
        print("Run: python ml/train_xgboost.py  to train first")
    else:
        print(f"Pump probability: {prob:.3f}")
