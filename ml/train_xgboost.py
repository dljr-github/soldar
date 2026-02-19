#!/usr/bin/env python3
"""
XGBoost baseline classifier for meme coin pump detection.
Usage:
  python ml/train_xgboost.py --train ml/data/train.parquet --val ml/data/val.parquet --test ml/data/test.parquet
  python ml/train_xgboost.py --train ml/data/train.parquet --val ml/data/val.parquet --test ml/data/test.parquet --tune
  python ml/train_xgboost.py  # uses ml/dataset.parquet with auto-split, or generates synthetic data
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
import time

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

# Ensure ml package is importable
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from evaluate import (
    compute_trading_metrics,
    plot_confusion_matrix,
    plot_roc_curve,
    print_classification_report,
    save_metrics,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
MODEL_PATH = os.path.join(MODEL_DIR, "xgb_baseline.pkl")
METRICS_PATH = os.path.join(MODEL_DIR, "xgb_metrics.json")
IMPORTANCE_PATH = os.path.join(MODEL_DIR, "xgb_feature_importance.png")
CM_PATH = os.path.join(MODEL_DIR, "xgb_confusion_matrix.png")
ROC_PATH = os.path.join(MODEL_DIR, "xgb_roc_curve.png")

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
DETECTION_MINUTE = 5  # Train on T=5min snapshot


# ---------------------------------------------------------------------------
# Synthetic data generator
# ---------------------------------------------------------------------------
def generate_synthetic_data(n_samples: int = 2000, seed: int = 42) -> pd.DataFrame:
    """Generate synthetic training data matching the MemeTrans feature schema.

    Distributions are loosely based on observed meme coin patterns:
    - Most coins don't pump (85-90% negative)
    - Pumpers tend to have higher early volume and price velocity
    """
    rng = np.random.RandomState(seed)

    # Base rate: ~12% pump
    n_pump = int(n_samples * 0.12)
    n_nopump = n_samples - n_pump

    rows = []
    for label, count in [(False, n_nopump), (True, n_pump)]:
        for _ in range(count):
            if label:  # Pump characteristics
                price_change = rng.lognormal(3.0, 1.0)  # higher gains
                vol_per_min = rng.lognormal(8.5, 1.2)    # ~5000-50000 USD/min
                price_velocity = rng.normal(15, 10)       # positive velocity
                hl_range = rng.lognormal(3.5, 0.8)        # wider range
                candles_up = rng.beta(5, 2)               # mostly up candles
                liq = rng.lognormal(10.5, 1.0)            # ~30k-100k
                mcap = liq * rng.uniform(2, 10)
            else:  # No-pump characteristics
                price_change = rng.normal(0, 20)          # centered around 0
                vol_per_min = rng.lognormal(6.5, 1.5)     # ~500-5000 USD/min
                price_velocity = rng.normal(-2, 8)        # slightly negative
                hl_range = rng.lognormal(2.5, 1.0)        # tighter range
                candles_up = rng.beta(2, 3)               # mostly down candles
                liq = rng.lognormal(9.5, 1.5)             # wider range
                mcap = liq * rng.uniform(1, 20)

            liq_mcap = liq / mcap if mcap > 0 else 0

            rows.append({
                "price_change_T_pct": price_change,
                "vol_per_min_T": vol_per_min,
                "price_velocity": price_velocity,
                "high_low_range_pct": hl_range,
                "candles_up_pct": np.clip(candles_up, 0, 1),
                "initial_liquidity_usd": liq,
                "mcap_at_listing": mcap,
                "liq_to_mcap_ratio": liq_mcap,
                "pumped_2x": label,
                "detection_minute": DETECTION_MINUTE,
            })

    df = pd.DataFrame(rows)
    # Shuffle
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    return df


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_data(
    train_path: str | None,
    val_path: str | None,
    test_path: str | None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load train/val/test splits. Falls back to auto-split or synthetic data."""

    # Case 1: explicit parquet splits provided
    if train_path and os.path.exists(train_path):
        print(f"Loading train: {train_path}")
        df_train = pd.read_parquet(train_path)
        df_val = pd.read_parquet(val_path) if val_path and os.path.exists(val_path) else None
        df_test = pd.read_parquet(test_path) if test_path and os.path.exists(test_path) else None

        # Filter to detection minute
        df_train = df_train[df_train["detection_minute"] == DETECTION_MINUTE].copy()
        if df_val is not None:
            df_val = df_val[df_val["detection_minute"] == DETECTION_MINUTE].copy()
        if df_test is not None:
            df_test = df_test[df_test["detection_minute"] == DETECTION_MINUTE].copy()

        # Auto-split if val/test missing
        if df_val is None or df_test is None:
            df_train, df_val, df_test = _auto_split(df_train)

        return df_train, df_val, df_test

    # Case 2: single dataset.parquet exists — auto-split
    dataset_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dataset.parquet")
    if os.path.exists(dataset_path):
        print(f"Loading single dataset: {dataset_path}")
        df = pd.read_parquet(dataset_path)
        df = df[df["detection_minute"] == DETECTION_MINUTE].copy()
        if len(df) >= 20:
            return _auto_split(df)
        print(f"Only {len(df)} samples at T={DETECTION_MINUTE}min — too few for real training.")

    # Case 3: generate synthetic data
    print("No training data found. Generating synthetic dataset...")
    df = generate_synthetic_data(n_samples=2000)
    return _auto_split(df)


def _auto_split(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split dataframe into 70/15/15 train/val/test."""
    y = df[TARGET].astype(int)

    # Stratified split if both classes present
    stratify = y if y.nunique() >= 2 else None
    df_train, df_temp = train_test_split(df, test_size=0.3, random_state=42, stratify=stratify)

    y_temp = df_temp[TARGET].astype(int)
    stratify_temp = y_temp if y_temp.nunique() >= 2 else None
    df_val, df_test = train_test_split(df_temp, test_size=0.5, random_state=42, stratify=stratify_temp)

    print(f"Split: train={len(df_train)}, val={len(df_val)}, test={len(df_test)}")
    return df_train, df_val, df_test


# ---------------------------------------------------------------------------
# Feature preparation
# ---------------------------------------------------------------------------
def prepare_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Extract X, y from dataframe with proper type handling."""
    X = df[FEATURE_COLS].copy()
    # Convert object columns to numeric (handles None/string values from parquet)
    for col in X.columns:
        X[col] = pd.to_numeric(X[col], errors="coerce")
    X = X.fillna(X.median())
    y = df[TARGET].astype(int)
    return X, y


# ---------------------------------------------------------------------------
# Optuna hyperparameter tuning
# ---------------------------------------------------------------------------
def run_optuna_search(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    n_trials: int = 50,
) -> dict:
    """Run Optuna hyperparameter search, return best params."""
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    neg_count = int((y_train == 0).sum())
    pos_count = int((y_train == 1).sum())
    base_spw = neg_count / pos_count if pos_count > 0 else 1.0

    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "gamma": trial.suggest_float("gamma", 0.0, 5.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
            "scale_pos_weight": trial.suggest_float("scale_pos_weight", base_spw * 0.5, base_spw * 2.0),
        }

        clf = XGBClassifier(
            **params,
            eval_metric="logloss",
            random_state=42,
            verbosity=0,
        )
        clf.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )
        y_proba = clf.predict_proba(X_val)[:, 1]
        from sklearn.metrics import f1_score as f1_fn

        y_pred = (y_proba >= 0.5).astype(int)
        return f1_fn(y_val, y_pred, zero_division=0)

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    print(f"\nBest Optuna trial: F1={study.best_value:.4f}")
    print(f"Best params: {json.dumps(study.best_params, indent=2)}")

    return study.best_params


# ---------------------------------------------------------------------------
# SHAP feature importance
# ---------------------------------------------------------------------------
def plot_shap_importance(model, X: pd.DataFrame, save_path: str) -> None:
    """Plot SHAP feature importance and save."""
    try:
        import shap

        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X)

        fig, ax = plt.subplots(figsize=(10, 6))
        shap.summary_plot(shap_values, X, show=False, plot_size=None)
        plt.tight_layout()
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close("all")
        print(f"SHAP feature importance saved to {save_path}")
    except Exception as e:
        # Fall back to built-in feature importance
        print(f"SHAP failed ({e}), using built-in feature importance")
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1]

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.barh(
            range(len(importances)),
            importances[indices],
            color="#58a6ff",
        )
        ax.set_yticks(range(len(importances)))
        ax.set_yticklabels([X.columns[i] for i in indices])
        ax.invert_yaxis()
        ax.set_xlabel("Feature Importance")
        ax.set_title("XGBoost Feature Importance")
        fig.tight_layout()
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Feature importance plot saved to {save_path}")


# ---------------------------------------------------------------------------
# Main training function
# ---------------------------------------------------------------------------
def train(
    train_path: str | None = None,
    val_path: str | None = None,
    test_path: str | None = None,
    tune: bool = False,
) -> None:
    """Train XGBoost classifier with full evaluation pipeline."""
    print("=" * 60)
    print("XGBoost Meme Coin Pump Detector")
    print("=" * 60)

    # Load data
    df_train, df_val, df_test = load_data(train_path, val_path, test_path)

    X_train, y_train = prepare_features(df_train)
    X_val, y_val = prepare_features(df_val)
    X_test, y_test = prepare_features(df_test)

    print(f"\nClass balance (train): {y_train.value_counts().to_dict()}")
    print(f"Class balance (val):   {y_val.value_counts().to_dict()}")
    print(f"Class balance (test):  {y_test.value_counts().to_dict()}")

    # Handle class imbalance
    neg_count = int((y_train == 0).sum())
    pos_count = int((y_train == 1).sum())
    scale_pos_weight = neg_count / pos_count if pos_count > 0 else 1.0
    print(f"scale_pos_weight: {scale_pos_weight:.2f}")

    # Hyperparameter tuning or defaults
    if tune:
        print("\nRunning Optuna hyperparameter search (50 trials)...")
        best_params = run_optuna_search(X_train, y_train, X_val, y_val, n_trials=50)
        clf = XGBClassifier(
            **best_params,
            eval_metric="logloss",
            early_stopping_rounds=50,
            random_state=42,
            verbosity=0,
        )
    else:
        clf = XGBClassifier(
            n_estimators=500,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=scale_pos_weight,
            eval_metric="logloss",
            early_stopping_rounds=50,
            random_state=42,
            verbosity=0,
        )

    # Train with early stopping on val set
    print("\nTraining...")
    t0 = time.time()
    clf.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )
    elapsed = time.time() - t0
    best_iter = getattr(clf, "best_iteration", clf.n_estimators)
    print(f"Training complete in {elapsed:.1f}s (best iteration: {best_iter})")

    # Evaluate on test set
    y_proba = clf.predict_proba(X_test)[:, 1]
    y_pred = (y_proba >= 0.5).astype(int)

    # Classification report
    clf_metrics = print_classification_report(y_test.values, y_pred, y_proba)

    # Trading metrics
    trading_metrics = compute_trading_metrics(y_test.values, y_proba, threshold=0.5)

    # Plots
    os.makedirs(MODEL_DIR, exist_ok=True)
    plot_confusion_matrix(y_test.values, y_pred, CM_PATH)
    auc = plot_roc_curve(y_test.values, y_proba, ROC_PATH)
    plot_shap_importance(clf, X_test, IMPORTANCE_PATH)

    # Save model
    with open(MODEL_PATH, "wb") as f:
        pickle.dump({"model": clf, "feature_cols": FEATURE_COLS, "target": TARGET}, f)
    print(f"\nModel saved to {MODEL_PATH}")

    # Save metrics
    all_metrics = {
        "classification": clf_metrics,
        "trading": trading_metrics,
        "model_params": clf.get_params(),
        "training_time_seconds": round(elapsed, 2),
        "best_iteration": best_iter,
        "feature_cols": FEATURE_COLS,
    }
    save_metrics(all_metrics, METRICS_PATH)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Train XGBoost pump detector")
    parser.add_argument("--train", type=str, default=None, help="Path to train parquet")
    parser.add_argument("--val", type=str, default=None, help="Path to validation parquet")
    parser.add_argument("--test", type=str, default=None, help="Path to test parquet")
    parser.add_argument("--tune", action="store_true", help="Run Optuna hyperparameter search")
    args = parser.parse_args()

    train(train_path=args.train, val_path=args.val, test_path=args.test, tune=args.tune)


if __name__ == "__main__":
    main()
