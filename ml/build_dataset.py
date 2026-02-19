#!/usr/bin/env python3
"""Build ML-ready dataset from collected raw pool data."""

from __future__ import annotations

import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd

from features import extract_features

RAW_DIR = "ml/raw"
OUT_PARQUET = "ml/dataset.parquet"
OUT_CSV = "ml/dataset.csv"


def build():
    files = glob.glob(os.path.join(RAW_DIR, "*.json"))
    files = [f for f in files if not f.endswith("manifest.json")]

    if not files:
        print("No raw data files found. Run collect.py first.")
        return

    all_rows = []
    skipped = 0
    for fpath in files:
        with open(fpath) as f:
            pool_data = json.load(f)
        rows = extract_features(pool_data)
        if rows:
            all_rows.extend(rows)
        else:
            skipped += 1

    if not all_rows:
        print("No features extracted from any pool. Check raw data quality.")
        return

    df = pd.DataFrame(all_rows)

    os.makedirs(os.path.dirname(OUT_PARQUET) or ".", exist_ok=True)
    df.to_parquet(OUT_PARQUET, index=False)
    df.to_csv(OUT_CSV, index=False)

    # --- Stats ---
    print(f"\n{'='*60}")
    print(f"Dataset built: {len(df)} rows from {len(files)} pools ({skipped} skipped)")
    print(f"Saved to: {OUT_PARQUET} and {OUT_CSV}")
    print(f"{'='*60}")

    # Per detection minute stats
    for t in sorted(df["detection_minute"].unique()):
        subset = df[df["detection_minute"] == t]
        n = len(subset)
        if n == 0:
            continue
        pct_2x = subset["pumped_2x"].mean() * 100
        pct_5x = subset["pumped_5x"].mean() * 100
        pct_10x = subset["pumped_10x"].mean() * 100
        print(f"\nT={t:>2d}min | {n:>4d} pools | 2x: {pct_2x:.1f}% | 5x: {pct_5x:.1f}% | 10x: {pct_10x:.1f}%")

    # Feature correlations with pumped_2x at T=5
    t5 = df[df["detection_minute"] == 5]
    if len(t5) > 5:
        feature_cols = [
            "price_change_T_pct", "vol_per_min_T", "price_velocity",
            "high_low_range_pct", "candles_up_pct", "initial_liquidity_usd",
            "mcap_at_listing", "liq_to_mcap_ratio",
        ]
        print(f"\nFeature correlations with pumped_2x (T=5min, n={len(t5)}):")
        for col in feature_cols:
            if col in t5.columns and t5[col].notna().sum() > 5:
                corr = t5[col].corr(t5["pumped_2x"].astype(float))
                print(f"  {col:<25s} {corr:+.3f}")


if __name__ == "__main__":
    build()
