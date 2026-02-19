#!/usr/bin/env python3
"""Train baseline XGBoost classifier for meme coin pump prediction."""

from __future__ import annotations

import os
import pickle

import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score
from xgboost import XGBClassifier

MODEL_DIR = "ml/models"
MODEL_PATH = os.path.join(MODEL_DIR, "xgb_baseline.pkl")
DATASET_PATH = "ml/dataset.parquet"

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

TARGET = "pumped_2x"


def train():
    if not os.path.exists(DATASET_PATH):
        print(f"Dataset not found at {DATASET_PATH}. Run build_dataset.py first.")
        return

    df = pd.read_parquet(DATASET_PATH)

    # Filter to detection_minute == 5
    df = df[df["detection_minute"] == 5].copy()
    print(f"Training on {len(df)} samples (T=5min)")

    if len(df) < 20:
        print("Not enough samples for meaningful training. Collect more data.")
        return

    X = df[FEATURE_COLS].copy()
    y = df[TARGET].astype(int)

    # Fill NaN with median for numeric features
    X = X.fillna(X.median())

    # Handle edge case: if target has only one class
    if y.nunique() < 2:
        print(f"Only one class in target ({y.unique()}). Need both positive and negative examples.")
        return

    clf = XGBClassifier(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.1,
        eval_metric="logloss",
        random_state=42,
    )

    # 5-fold cross-validation
    n_folds = min(5, y.value_counts().min())
    if n_folds < 2:
        print("Not enough samples per class for cross-validation. Fitting without CV.")
        clf.fit(X, y)
    else:
        scores = cross_val_score(clf, X, y, cv=n_folds, scoring="accuracy")
        print(f"\n{n_folds}-fold CV accuracy: {scores.mean():.3f} (+/- {scores.std():.3f})")
        clf.fit(X, y)

    # Feature importances
    importances = clf.feature_importances_
    print("\nFeature importances:")
    for name, imp in sorted(zip(FEATURE_COLS, importances), key=lambda x: -x[1]):
        print(f"  {name:<25s} {imp:.4f}")

    # Save model
    os.makedirs(MODEL_DIR, exist_ok=True)
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(clf, f)
    print(f"\nModel saved to {MODEL_PATH}")


if __name__ == "__main__":
    train()
