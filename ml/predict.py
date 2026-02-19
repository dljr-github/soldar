#!/usr/bin/env python3
"""
Run trained model on live screener candidates.
Called by screener.py to enhance scoring with ML prediction.

NOTE: pickle is used here because XGBoost/sklearn models are saved as pickle
by the training pipeline. The model files are local training artifacts, not
untrusted external data. See HIGH-2 in the audit for future hardening (hash
verification or native XGBoost .json format).
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

# Prefer tuned model, fall back to baseline
XGB_MODEL_PATHS = [
    os.path.join(MODEL_DIR, "xgb_tuned.pkl"),
    os.path.join(MODEL_DIR, "xgb_baseline.pkl"),
]

# Cached model state (loaded once, reused across calls, reloads on file change)
_cached_model = None
_cached_model_mtime = 0.0
_cached_model_path = ""
_cached_feature_cols: list[str] | None = None
_cached_imputer_medians: dict[str, float] | None = None


def _load_model():
    """Load the XGBoost model, caching it for reuse. Reloads if file changed."""
    global _cached_model, _cached_model_mtime, _cached_model_path
    global _cached_feature_cols, _cached_imputer_medians

    # Find first available model file
    model_path = None
    for path in XGB_MODEL_PATHS:
        if os.path.exists(path):
            model_path = path
            break

    if model_path is None:
        log.debug("No XGBoost model found in %s", MODEL_DIR)
        return None

    mtime = os.path.getmtime(model_path)
    if _cached_model is not None and mtime == _cached_model_mtime and model_path == _cached_model_path:
        return _cached_model

    try:
        with open(model_path, "rb") as f:
            data = pickle.load(f)  # noqa: S301 — local training artifact, not untrusted

        # Handle both old format (bare model) and new format (dict with metadata)
        if isinstance(data, dict):
            _cached_model = data["model"]
            _cached_feature_cols = data.get("feature_cols")
            # Load imputer medians if saved during training
            _cached_imputer_medians = data.get("imputer_medians")
        else:
            _cached_model = data
            _cached_feature_cols = None
            _cached_imputer_medians = None

        _cached_model_mtime = mtime
        _cached_model_path = model_path
        log.info("Loaded XGBoost model from %s (features: %s)",
                 model_path,
                 len(_cached_feature_cols) if _cached_feature_cols else "unknown")
        return _cached_model
    except Exception as e:
        log.warning("Failed to load model: %s", e)
        return None


def _extract_features_from_pair(pair_data: dict) -> dict:
    """Extract ML features from a live DexScreener pair dict.

    When the model was trained on data_pipeline.py features (MemeTrans),
    computes the 4 aligned features that are available from DexScreener.
    Missing features (holder data, on-chain metrics) are set to NaN.
    """
    pc = pair_data.get("priceChange") or {}
    vol = pair_data.get("volume") or {}
    liq_data = pair_data.get("liquidity") or {}
    txns = pair_data.get("txns") or {}
    txns_5m = txns.get("m5") or {}

    liq_usd = liq_data.get("usd") or 0
    mcap = pair_data.get("marketCap") or pair_data.get("fdv") or 0
    vol_5m = vol.get("m5") or 0
    if vol_5m <= 0:
        vol_h1 = vol.get("h1") or 0
        vol_5m = vol_h1 / 12
    buys = txns_5m.get("buys") or 0
    sells = txns_5m.get("sells") or 0
    total_txns = buys + sells
    p5m = pc.get("m5") or 0
    p1h = pc.get("h1") or 0
    p6h = abs(pc.get("h6") or pc.get("h1") or 0)

    # Features aligned with data_pipeline.py training features:
    features = {
        # These 4 are the "aligned concepts" from data_pipeline.py
        "volume_intensity": vol_5m / liq_usd if liq_usd > 0 else 0,
        "buy_sell_pressure": buys / total_txns if total_txns > 0 else 0.5,
        "momentum": p5m,
        "price_range": max(abs(p5m), p6h),
    }

    # Also compute build_dataset.py (features.py) features for older models
    vol_per_min = vol_5m / 5 if vol_5m > 0 else 0
    features.update({
        "price_change_T_pct": p5m,
        "vol_per_min_T": vol_per_min,
        "price_velocity": p5m - (p1h / 12) if p1h else p5m,
        "high_low_range_pct": max(abs(p5m), p6h) * 1.5,
        "candles_up_pct": buys / total_txns if total_txns > 0 else 0.5,
        "initial_liquidity_usd": liq_usd,
        "mcap_at_listing": mcap,
        "liq_to_mcap_ratio": liq_usd / mcap if mcap > 0 else 0,
    })

    return features


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
        all_features = pd.DataFrame([features])

        # Use the model's training feature columns if available
        if _cached_feature_cols:
            # Add missing columns as NaN
            for col in _cached_feature_cols:
                if col not in all_features.columns:
                    all_features[col] = np.nan
            X = all_features[_cached_feature_cols]
        else:
            # Fallback: try features.py columns for older baseline models
            fallback_cols = [
                "price_change_T_pct", "vol_per_min_T", "price_velocity",
                "high_low_range_pct", "candles_up_pct",
                "initial_liquidity_usd", "mcap_at_listing", "liq_to_mcap_ratio",
            ]
            X = all_features[[c for c in fallback_cols if c in all_features.columns]]

        # Convert to numeric and handle NaN
        for col in X.columns:
            X[col] = pd.to_numeric(X[col], errors="coerce")

        # Use training-time medians if available, else 0
        if _cached_imputer_medians:
            for col in X.columns:
                if col in _cached_imputer_medians:
                    X[col] = X[col].fillna(_cached_imputer_medians[col])
            X = X.fillna(0)
        else:
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
        print("Model not loaded (no trained model found)")
        print("Run: python ml/train_xgboost.py  to train first")
    else:
        print(f"Pump probability: {prob:.3f}")
