#!/usr/bin/env python3
"""
Unified data pipeline for Soldar ML training.
Loads, aligns, and merges all available datasets:
  1. MemeTrans — 41K Solana meme coins, 124 features + label
  2. Sapienza  — 584K CEX pump events at 15s intervals
  3. Mehrnoom  — 63K Telegram pump signal timestamps
"""

from __future__ import annotations

import json
import os
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

warnings.filterwarnings("ignore", category=FutureWarning)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
os.makedirs(DATA_DIR, exist_ok=True)

MEMETRANS_PATH = os.path.expanduser("~/repos/MemeTrans/data/feat_label.csv")
SAPIENZA_PATH = os.path.expanduser(
    "~/repos/sapienza-dataset/labeled_features/features_15S.csv.gz"
)
MEHRNOOM_PATH = os.path.expanduser(
    "~/repos/mehrnoom-social/Telegram/classified/coin-pump.csv"
)

# MemeTrans metadata columns (not features)
MEMETRANS_META = ["Unnamed: 0", "mint_ts", "mint_address", "token", "time", "label",
                  "min_ratio", "pred_proba", "return_ratio"]

# Live screener features (from features.py)
SCREENER_FEATURES = [
    "price_change_T_pct", "vol_per_min_T", "price_velocity",
    "high_low_range_pct", "candles_up_pct", "initial_liquidity_usd",
    "mcap_at_listing", "liq_to_mcap_ratio",
]


def banner(msg: str) -> None:
    print(f"\n{'='*70}")
    print(f"  {msg}")
    print(f"{'='*70}")


# ===================================================================
# STEP 1 — Load & inspect all three datasets
# ===================================================================

def step1_inspect() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    banner("STEP 1: Load & Inspect All Datasets")

    # MemeTrans
    print("\n--- MemeTrans ---")
    mt = pd.read_csv(MEMETRANS_PATH)
    print(f"  Shape: {mt.shape}")
    print(f"  Columns: {mt.columns.tolist()[:10]} ... ({len(mt.columns)} total)")
    print(f"  Label distribution:\n{mt['label'].value_counts().to_string()}")
    missing = mt.isnull().sum()
    missing_pct = (missing / len(mt) * 100).sort_values(ascending=False)
    cols_with_missing = missing_pct[missing_pct > 0]
    print(f"  Columns with missing values: {len(cols_with_missing)}")
    if len(cols_with_missing) > 0:
        print(f"  Top 5 missing:\n{cols_with_missing.head().to_string()}")

    # Sapienza
    print("\n--- Sapienza ---")
    sap = pd.read_csv(SAPIENZA_PATH)
    print(f"  Shape: {sap.shape}")
    print(f"  Columns: {sap.columns.tolist()}")
    print(f"  Label (gt) distribution:\n{sap['gt'].value_counts().to_string()}")
    print(f"  pump_index range: {sap['pump_index'].min()} to {sap['pump_index'].max()}")
    missing_s = sap.isnull().sum()
    print(f"  Total missing values: {missing_s.sum()}")

    # Mehrnoom
    print("\n--- Mehrnoom Telegram ---")
    meh = pd.read_csv(MEHRNOOM_PATH)
    print(f"  Shape: {meh.shape}")
    print(f"  Columns: {meh.columns.tolist()}")
    print(f"  Unique coins: {meh['Coin'].nunique()}")
    print(f"  Date range: {meh['Date'].min()} to {meh['Date'].max()}")
    missing_m = meh.isnull().sum()
    print(f"  Total missing values: {missing_m.sum()}")

    return mt, sap, meh


# ===================================================================
# STEP 2 — MemeTrans feature analysis & cleaning
# ===================================================================

def step2_memetrans_clean(mt: pd.DataFrame) -> pd.DataFrame:
    banner("STEP 2: MemeTrans Feature Analysis & Cleaning")

    # Identify feature groups
    groups = {
        "group1_context": [c for c in mt.columns if c.startswith("group1_")],
        "group2_holder_dist": [c for c in mt.columns if c.startswith("group2_")],
        "group3_trading": [c for c in mt.columns if c.startswith("group3_")],
        "group4_clustering": [c for c in mt.columns if c.startswith("group4_")],
    }
    for name, cols in groups.items():
        print(f"  {name}: {len(cols)} features")

    # Label analysis
    print(f"\n  Label column: 'label'")
    print(f"  Label distribution:\n{mt['label'].value_counts().to_string()}")

    # Convert label to binary: 'high' risk = 1, else = 0
    mt["label_binary"] = (mt["label"] == "high").astype(int)
    print(f"\n  Binary label (high=1): {mt['label_binary'].value_counts().to_dict()}")

    # Drop columns with >50% missing
    feature_cols = [c for c in mt.columns if c not in MEMETRANS_META + ["label_binary"]]
    missing_pct = mt[feature_cols].isnull().mean()
    drop_cols = missing_pct[missing_pct > 0.5].index.tolist()
    if drop_cols:
        print(f"\n  Dropping {len(drop_cols)} columns with >50% missing: {drop_cols}")
        mt = mt.drop(columns=drop_cols)
    else:
        print(f"\n  No columns with >50% missing — all retained")

    # Recount features
    clean_features = [c for c in mt.columns if c not in MEMETRANS_META + ["label_binary"]]
    print(f"  Clean feature count: {len(clean_features)}")

    # Save
    out_path = os.path.join(DATA_DIR, "memetrans_clean.parquet")
    mt.to_parquet(out_path, index=False)
    print(f"  Saved: {out_path} ({mt.shape})")

    return mt


# ===================================================================
# STEP 3 — Sapienza feature extraction (aggregate per pump event)
# ===================================================================

def _extract_window_features(window: pd.DataFrame) -> dict | None:
    """Extract aggregate features from a time-series window of Sapienza data."""
    if len(window) < 5:
        return None

    last5 = window.tail(5)
    last10 = window.tail(10)
    last20 = window.tail(20)

    max_rush_order_spike = last5["std_rush_order"].max()

    # Volume trend: slope via simple linear regression
    if len(last10) >= 2:
        x = np.arange(len(last10))
        y = last10["avg_volume"].values
        avg_volume_trend = float(np.polyfit(x, y, 1)[0]) if np.std(y) > 0 else 0.0
    else:
        avg_volume_trend = 0.0

    price_volatility = float(last10["std_price"].std()) if len(last10) >= 2 else 0.0
    mean_vol_20 = last20["avg_volume"].mean()
    last_vol = window.iloc[-1]["avg_volume"]
    volume_spike_ratio = float(last_vol / mean_vol_20) if mean_vol_20 != 0 else 1.0

    time_sin = float(window.iloc[-1]["hour_sin"])
    time_cos = float(window.iloc[-1]["hour_cos"])

    return {
        "max_rush_order_spike": float(max_rush_order_spike),
        "avg_volume_trend": avg_volume_trend,
        "price_volatility": price_volatility,
        "volume_spike_ratio": volume_spike_ratio,
        "time_of_day_sin": time_sin,
        "time_of_day_cos": time_cos,
    }


def step3_sapienza_features(sap: pd.DataFrame) -> pd.DataFrame:
    banner("STEP 3: Sapienza Feature Extraction")

    # Sort by symbol and date to maintain temporal order
    sap = sap.sort_values(["symbol", "date"]).reset_index(drop=True)

    total_pumps = int(sap["gt"].sum())
    print(f"  Total symbols: {sap['symbol'].nunique()}")
    print(f"  Total pump events (gt=1): {total_pumps}")

    rng = np.random.RandomState(42)
    records = []
    window_size = 20  # rows to look back before a pump

    for sym, sym_data in sap.groupby("symbol"):
        sym_data = sym_data.reset_index(drop=True)

        # --- Positive samples: extract pre-pump windows around each gt=1 row ---
        pump_indices = sym_data.index[sym_data["gt"] == 1].tolist()
        for pidx in pump_indices:
            start = max(0, pidx - window_size)
            if pidx - start < 5:
                continue
            window = sym_data.iloc[start:pidx]
            feats = _extract_window_features(window)
            if feats:
                feats["symbol"] = sym
                feats["label"] = 1
                records.append(feats)

        # --- Negative samples: random windows far from any pump ---
        # Mark rows within ±30 of any pump as "near pump"
        near_pump = set()
        for pidx in pump_indices:
            for offset in range(-30, 31):
                near_pump.add(pidx + offset)

        safe_indices = [i for i in range(window_size, len(sym_data))
                        if i not in near_pump]

        # Sample up to 3x the number of pump events for balance
        n_neg = min(len(pump_indices) * 3, len(safe_indices))
        if n_neg > 0:
            sampled = rng.choice(safe_indices, size=n_neg, replace=False)
            for idx in sampled:
                window = sym_data.iloc[idx - window_size:idx]
                feats = _extract_window_features(window)
                if feats:
                    feats["symbol"] = sym
                    feats["label"] = 0
                    records.append(feats)

    sap_feat = pd.DataFrame(records)
    print(f"\n  Sapienza aggregated dataset: {sap_feat.shape}")
    if len(sap_feat) > 0:
        print(f"  Label distribution:\n{sap_feat['label'].value_counts().to_string()}")

        # Replace infinities
        feat_cols = [c for c in sap_feat.columns if c not in ("symbol", "label")]
        sap_feat[feat_cols] = sap_feat[feat_cols].replace([np.inf, -np.inf], np.nan)
    else:
        print("  WARNING: No features extracted from Sapienza data")

    out_path = os.path.join(DATA_DIR, "sapienza_features.parquet")
    sap_feat.to_parquet(out_path, index=False)
    print(f"  Saved: {out_path}")

    return sap_feat


# ===================================================================
# STEP 4 — Mehrnoom timing analysis
# ===================================================================

def step4_mehrnoom_timing(meh: pd.DataFrame) -> dict:
    banner("STEP 4: Mehrnoom Telegram Timing Analysis")

    # Parse datetime
    meh = meh.copy()
    meh["datetime"] = pd.to_datetime(meh["Date"] + " " + meh["Time"], errors="coerce")
    meh = meh.dropna(subset=["datetime"])
    meh = meh.sort_values(["Coin", "datetime"])

    # Compute inter-signal intervals per coin
    intervals = []
    for coin, group in meh.groupby("Coin"):
        if len(group) < 2:
            continue
        dts = group["datetime"].sort_values()
        diffs = dts.diff().dropna().dt.total_seconds() / 60  # minutes
        intervals.extend(diffs.tolist())

    intervals = pd.Series(intervals)
    intervals = intervals[intervals > 0]  # only positive intervals

    stats = {
        "total_signals": len(meh),
        "unique_coins": int(meh["Coin"].nunique()),
        "unique_channels": int(meh["Channel ID"].nunique()),
        "date_range": {"start": str(meh["Date"].min()), "end": str(meh["Date"].max())},
        "inter_signal_minutes": {
            "mean": round(float(intervals.mean()), 2) if len(intervals) > 0 else None,
            "median": round(float(intervals.median()), 2) if len(intervals) > 0 else None,
            "p25": round(float(intervals.quantile(0.25)), 2) if len(intervals) > 0 else None,
            "p75": round(float(intervals.quantile(0.75)), 2) if len(intervals) > 0 else None,
            "std": round(float(intervals.std()), 2) if len(intervals) > 0 else None,
        },
        "signals_per_coin": {
            "mean": round(float(meh.groupby("Coin").size().mean()), 2),
            "median": round(float(meh.groupby("Coin").size().median()), 2),
            "max": int(meh.groupby("Coin").size().max()),
        },
        "busiest_hours": meh["datetime"].dt.hour.value_counts().head(5).to_dict(),
        "note": (
            "Inter-signal intervals measure time between consecutive Telegram pump "
            "signals for the same coin. Shorter intervals suggest coordinated pump "
            "campaigns. Busiest hours indicate when pump groups are most active."
        ),
    }

    print(f"  Total signals: {stats['total_signals']}")
    print(f"  Unique coins: {stats['unique_coins']}")
    print(f"  Unique channels: {stats['unique_channels']}")
    print(f"  Date range: {stats['date_range']}")
    if stats["inter_signal_minutes"]["mean"] is not None:
        print(f"  Inter-signal interval (mean): {stats['inter_signal_minutes']['mean']} min")
        print(f"  Inter-signal interval (median): {stats['inter_signal_minutes']['median']} min")
    print(f"  Signals per coin (mean): {stats['signals_per_coin']['mean']}")
    print(f"  Busiest hours (UTC): {stats['busiest_hours']}")

    out_path = os.path.join(DATA_DIR, "telegram_timing_stats.json")
    with open(out_path, "w") as f:
        json.dump(stats, f, indent=2, default=str)
    print(f"  Saved: {out_path}")

    return stats


# ===================================================================
# STEP 5 — Feature alignment across datasets
# ===================================================================

def step5_feature_alignment(mt: pd.DataFrame) -> dict:
    banner("STEP 5: Feature Alignment Across Datasets")

    # Sapienza → MemeTrans/Live screener mapping
    mapping = {
        "sapienza_to_screener": {
            "volume_spike_ratio": {
                "screener_equiv": "liq_to_mcap_ratio",
                "memetrans_equiv": ["group3_buy_vol", "group3_sell_vol"],
                "description": "Volume concentration / spike indicator",
            },
            "max_rush_order_spike": {
                "screener_equiv": "candles_up_pct",
                "memetrans_equiv": ["group3_buy_num", "group3_sell_num", "group3_sell_pressure"],
                "description": "Buy pressure / order imbalance indicator",
            },
            "avg_volume_trend": {
                "screener_equiv": "price_velocity",
                "memetrans_equiv": ["group3_tx_per_sec", "group3_avg_buy_vol"],
                "description": "Momentum / acceleration of trading activity",
            },
            "price_volatility": {
                "screener_equiv": "high_low_range_pct",
                "memetrans_equiv": ["group1_price"],
                "description": "Price dispersion / range indicator",
            },
            "time_of_day_sin": {
                "screener_equiv": None,
                "memetrans_equiv": ["group1_migrate_hour"],
                "description": "Cyclical time encoding (sin component)",
            },
            "time_of_day_cos": {
                "screener_equiv": None,
                "memetrans_equiv": ["group1_migrate_hour"],
                "description": "Cyclical time encoding (cos component)",
            },
        },
        "memetrans_to_screener": {
            "group3_sell_pressure": {
                "screener_equiv": "candles_up_pct",
                "description": "Sell pressure ≈ inverse of candles_up ratio",
            },
            "group3_tx_per_sec": {
                "screener_equiv": "vol_per_min_T",
                "description": "Transaction frequency ≈ volume rate",
            },
            "group2_holder_gini": {
                "screener_equiv": "liq_to_mcap_ratio",
                "description": "Holder concentration ≈ liquidity concentration",
            },
            "group1_price": {
                "screener_equiv": "price_change_T_pct",
                "description": "Price level / change from listing",
            },
            "group3_buy_vol": {
                "screener_equiv": "vol_per_min_T",
                "description": "Buy volume ≈ total volume rate",
            },
            "group2_top1_pct": {
                "screener_equiv": "liq_to_mcap_ratio",
                "description": "Top holder concentration ≈ concentration ratio",
            },
        },
        "aligned_feature_pairs": [
            {
                "sapienza": "volume_spike_ratio",
                "memetrans_primary": "group3_buy_vol",
                "screener": "vol_per_min_T",
                "concept": "volume_intensity",
            },
            {
                "sapienza": "max_rush_order_spike",
                "memetrans_primary": "group3_sell_pressure",
                "screener": "candles_up_pct",
                "concept": "buy_sell_pressure",
            },
            {
                "sapienza": "avg_volume_trend",
                "memetrans_primary": "group3_tx_per_sec",
                "screener": "price_velocity",
                "concept": "momentum",
            },
            {
                "sapienza": "price_volatility",
                "memetrans_primary": "group1_price",
                "screener": "high_low_range_pct",
                "concept": "price_range",
            },
        ],
    }

    n_aligned = len(mapping["aligned_feature_pairs"])
    print(f"  Aligned feature pairs: {n_aligned}")
    for pair in mapping["aligned_feature_pairs"]:
        print(f"    {pair['concept']}: "
              f"sapienza.{pair['sapienza']} ↔ "
              f"memetrans.{pair['memetrans_primary']} ↔ "
              f"screener.{pair['screener']}")

    out_path = os.path.join(DATA_DIR, "feature_mapping.json")
    with open(out_path, "w") as f:
        json.dump(mapping, f, indent=2)
    print(f"\n  Saved: {out_path}")

    return mapping


# ===================================================================
# STEP 6 — Build unified training dataset
# ===================================================================

def step6_unified_dataset(
    mt: pd.DataFrame, sap_feat: pd.DataFrame, mapping: dict
) -> pd.DataFrame:
    banner("STEP 6: Build Unified Training Dataset")

    # Define the aligned feature concepts we'll use for merging
    aligned_pairs = mapping["aligned_feature_pairs"]

    # --- Prepare MemeTrans features ---
    mt_features = mt.copy()
    # Create conceptual features from MemeTrans columns
    mt_unified = pd.DataFrame()
    mt_unified["volume_intensity"] = mt_features.get("group3_buy_vol", pd.Series(dtype=float))
    mt_unified["buy_sell_pressure"] = mt_features.get("group3_sell_pressure", pd.Series(dtype=float))
    mt_unified["momentum"] = mt_features.get("group3_tx_per_sec", pd.Series(dtype=float))
    mt_unified["price_range"] = mt_features.get("group1_price", pd.Series(dtype=float))

    # Also include high-value MemeTrans-only features for the MemeTrans rows
    extra_mt_features = [
        "group2_holder_gini", "group2_top1_pct", "group2_top5_pct", "group2_top10_pct",
        "group2_dev_hold_pct", "group3_tx_num", "group3_trader_num", "group3_wash_ratio",
        "group3_buy_num", "group3_sell_num", "group3_avg_buy_vol", "group3_avg_sell_vol",
        "group4_holder_gini", "group4_cluster_num", "group4_cluster_holder_ratio",
        "group4_top1_pct", "group4_top5_pct",
    ]
    for col in extra_mt_features:
        if col in mt_features.columns:
            mt_unified[col] = mt_features[col]

    mt_unified["label"] = mt_features["label_binary"]
    mt_unified["source"] = "memetrans"

    # --- Prepare Sapienza features ---
    sap_unified = pd.DataFrame()
    sap_unified["volume_intensity"] = sap_feat["volume_spike_ratio"]
    sap_unified["buy_sell_pressure"] = sap_feat["max_rush_order_spike"]
    sap_unified["momentum"] = sap_feat["avg_volume_trend"]
    sap_unified["price_range"] = sap_feat["price_volatility"]

    # Sapienza doesn't have the MemeTrans-only features, fill with NaN
    for col in extra_mt_features:
        sap_unified[col] = np.nan

    sap_unified["label"] = sap_feat["label"]
    sap_unified["source"] = "sapienza"

    # --- Normalize each dataset's aligned features independently to 0-1 ---
    aligned_concepts = ["volume_intensity", "buy_sell_pressure", "momentum", "price_range"]

    scaler_mt = MinMaxScaler()
    scaler_sap = MinMaxScaler()

    mt_aligned_vals = mt_unified[aligned_concepts].replace([np.inf, -np.inf], np.nan)
    sap_aligned_vals = sap_unified[aligned_concepts].replace([np.inf, -np.inf], np.nan)

    # Fill NaN with median before scaling
    mt_aligned_vals = mt_aligned_vals.fillna(mt_aligned_vals.median())
    sap_aligned_vals = sap_aligned_vals.fillna(sap_aligned_vals.median())

    mt_unified[aligned_concepts] = scaler_mt.fit_transform(mt_aligned_vals)
    sap_unified[aligned_concepts] = scaler_sap.fit_transform(sap_aligned_vals)

    # Also scale extra MemeTrans features for MemeTrans rows
    extra_available = [c for c in extra_mt_features if c in mt_unified.columns]
    mt_extra_vals = mt_unified[extra_available].replace([np.inf, -np.inf], np.nan)
    mt_extra_vals = mt_extra_vals.fillna(mt_extra_vals.median())
    scaler_extra = MinMaxScaler()
    mt_unified[extra_available] = scaler_extra.fit_transform(mt_extra_vals)

    # --- Combine ---
    combined = pd.concat([mt_unified, sap_unified], ignore_index=True)
    print(f"  Combined dataset: {combined.shape}")
    print(f"  Source distribution:\n{combined['source'].value_counts().to_string()}")
    print(f"  Label distribution:\n{combined['label'].value_counts().to_string()}")

    # --- Stratified train/val/test split: 70/15/15 ---
    # First split: 70% train, 30% temp
    train_df, temp_df = train_test_split(
        combined, test_size=0.30, random_state=42,
        stratify=combined["label"],
    )
    # Second split: 50/50 of temp → 15% val, 15% test
    val_df, test_df = train_test_split(
        temp_df, test_size=0.50, random_state=42,
        stratify=temp_df["label"],
    )

    print(f"\n  Train: {len(train_df)} samples")
    print(f"  Val:   {len(val_df)} samples")
    print(f"  Test:  {len(test_df)} samples")

    # Save splits
    train_df.to_parquet(os.path.join(DATA_DIR, "train.parquet"), index=False)
    val_df.to_parquet(os.path.join(DATA_DIR, "val.parquet"), index=False)
    test_df.to_parquet(os.path.join(DATA_DIR, "test.parquet"), index=False)
    print(f"  Saved: train.parquet, val.parquet, test.parquet")

    # Dataset stats
    stats = {
        "total_samples": len(combined),
        "splits": {
            "train": {
                "size": len(train_df),
                "label_0": int((train_df["label"] == 0).sum()),
                "label_1": int((train_df["label"] == 1).sum()),
                "memetrans": int((train_df["source"] == "memetrans").sum()),
                "sapienza": int((train_df["source"] == "sapienza").sum()),
            },
            "val": {
                "size": len(val_df),
                "label_0": int((val_df["label"] == 0).sum()),
                "label_1": int((val_df["label"] == 1).sum()),
                "memetrans": int((val_df["source"] == "memetrans").sum()),
                "sapienza": int((val_df["source"] == "sapienza").sum()),
            },
            "test": {
                "size": len(test_df),
                "label_0": int((test_df["label"] == 0).sum()),
                "label_1": int((test_df["label"] == 1).sum()),
                "memetrans": int((test_df["source"] == "memetrans").sum()),
                "sapienza": int((test_df["source"] == "sapienza").sum()),
            },
        },
        "feature_columns": [c for c in combined.columns if c not in ("label", "source")],
        "aligned_features": aligned_concepts,
        "memetrans_only_features": extra_available,
    }

    stats_path = os.path.join(DATA_DIR, "dataset_stats.json")
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"  Saved: {stats_path}")

    return combined


# ===================================================================
# STEP 7 — Feature importance preview (RandomForest on MemeTrans)
# ===================================================================

def step7_feature_importance(mt: pd.DataFrame) -> list[tuple[str, float]]:
    banner("STEP 7: Feature Importance Preview (MemeTrans + RandomForest)")

    # Use all numeric feature columns from MemeTrans
    feature_cols = [
        c for c in mt.columns
        if c not in MEMETRANS_META + ["label_binary"]
        and mt[c].dtype in [np.float64, np.int64, float, int]
    ]
    print(f"  Using {len(feature_cols)} numeric features")

    X = mt[feature_cols].copy()
    y = mt["label_binary"]

    # Replace infinities and fill NaN
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(X.median())

    if y.nunique() < 2:
        print("  Only one class present — skipping importance analysis")
        return []

    rf = RandomForestClassifier(
        n_estimators=200, max_depth=10, random_state=42, n_jobs=-1
    )
    rf.fit(X, y)

    importances = list(zip(feature_cols, rf.feature_importances_))
    importances.sort(key=lambda x: -x[1])

    print(f"\n  Top 20 Features by Importance:")
    print(f"  {'Feature':<45s} {'Importance':>10s}")
    print(f"  {'-'*55}")
    for name, imp in importances[:20]:
        print(f"  {name:<45s} {imp:>10.4f}")

    return importances[:20]


# ===================================================================
# MAIN
# ===================================================================

def main():
    banner("SOLDAR UNIFIED DATA PIPELINE")
    print("  Loading MemeTrans + Sapienza + Mehrnoom datasets")
    print("  Building aligned training data for ML pipeline")

    # Step 1: Load & inspect
    mt, sap, meh = step1_inspect()

    # Step 2: Clean MemeTrans
    mt = step2_memetrans_clean(mt)

    # Step 3: Extract Sapienza features
    sap_feat = step3_sapienza_features(sap)

    # Step 4: Mehrnoom timing
    step4_mehrnoom_timing(meh)

    # Step 5: Feature alignment
    mapping = step5_feature_alignment(mt)

    # Step 6: Unified dataset
    combined = step6_unified_dataset(mt, sap_feat, mapping)

    # Step 7: Feature importance
    top_features = step7_feature_importance(mt)

    # --- Final Summary ---
    banner("PIPELINE COMPLETE — SUMMARY")

    stats_path = os.path.join(DATA_DIR, "dataset_stats.json")
    with open(stats_path) as f:
        stats = json.load(f)

    print(f"\n  Total training samples: {stats['total_samples']}")
    print(f"\n  Split sizes:")
    for split_name, split_stats in stats["splits"].items():
        bal = split_stats["label_1"] / split_stats["size"] * 100
        print(f"    {split_name}: {split_stats['size']} "
              f"(label_0={split_stats['label_0']}, label_1={split_stats['label_1']}, "
              f"positive={bal:.1f}%)")

    print(f"\n  Aligned features: {len(stats['aligned_features'])}")
    print(f"  MemeTrans-only features: {len(stats['memetrans_only_features'])}")
    print(f"  Total feature columns: {len(stats['feature_columns'])}")

    if top_features:
        print(f"\n  Top 5 features for Agent B to focus on:")
        for name, imp in top_features[:5]:
            print(f"    {name}: {imp:.4f}")

    print(f"\n  Output files:")
    for fname in sorted(os.listdir(DATA_DIR)):
        fpath = os.path.join(DATA_DIR, fname)
        size_mb = os.path.getsize(fpath) / (1024 * 1024)
        print(f"    {fname}: {size_mb:.2f} MB")


if __name__ == "__main__":
    main()
