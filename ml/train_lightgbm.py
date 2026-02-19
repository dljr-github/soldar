#!/usr/bin/env python3
"""
LightGBM classifier for meme coin pump detection on real MemeTrans data.

Usage:
  python ml/train_lightgbm.py --train ml/data/train.parquet --val ml/data/val.parquet --test ml/data/test.parquet
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
from lightgbm import LGBMClassifier

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
TARGET = "label"
DROP_COLS = {"label", "source"}


# ---------------------------------------------------------------------------
# Data loading (shared logic with train_xgboost.py)
# ---------------------------------------------------------------------------
def get_feature_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in DROP_COLS]


def prepare_features(
    df: pd.DataFrame, feature_cols: list[str]
) -> tuple[pd.DataFrame, pd.Series]:
    X = df[feature_cols].copy()
    for col in X.columns:
        X[col] = pd.to_numeric(X[col], errors="coerce")
    X = X.fillna(X.median())
    y = df[TARGET].astype(int)
    return X, y


# ---------------------------------------------------------------------------
# Main training
# ---------------------------------------------------------------------------
def train(train_path: str, val_path: str, test_path: str) -> dict:
    """Train LightGBM and return test metrics."""
    model_path = os.path.join(MODEL_DIR, "lgbm_baseline.pkl")
    metrics_path = os.path.join(MODEL_DIR, "lgbm_metrics.json")
    cm_path = os.path.join(MODEL_DIR, "lgbm_confusion_matrix.png")
    roc_path = os.path.join(MODEL_DIR, "lgbm_roc_curve.png")
    importance_path = os.path.join(MODEL_DIR, "lgbm_feature_importance.png")

    print("=" * 60)
    print("LightGBM Meme Coin Pump Detector")
    print("=" * 60)

    df_train = pd.read_parquet(train_path)
    df_val = pd.read_parquet(val_path)
    df_test = pd.read_parquet(test_path)
    print(f"Loaded: train={len(df_train)}, val={len(df_val)}, test={len(df_test)}")

    feature_cols = get_feature_cols(df_train)
    print(f"Features ({len(feature_cols)}): {feature_cols}")

    X_train, y_train = prepare_features(df_train, feature_cols)
    X_val, y_val = prepare_features(df_val, feature_cols)
    X_test, y_test = prepare_features(df_test, feature_cols)

    print(f"\nClass balance (train): {y_train.value_counts().to_dict()}")
    print(f"Class balance (val):   {y_val.value_counts().to_dict()}")
    print(f"Class balance (test):  {y_test.value_counts().to_dict()}")

    neg_count = int((y_train == 0).sum())
    pos_count = int((y_train == 1).sum())
    scale_pos_weight = neg_count / pos_count if pos_count > 0 else 1.0
    print(f"scale_pos_weight: {scale_pos_weight:.4f}")

    clf = LGBMClassifier(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        verbosity=-1,
    )

    print("\nTraining...")
    t0 = time.time()
    clf.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        eval_metric="logloss",
        callbacks=[
            __import__("lightgbm").early_stopping(50, verbose=False),
            __import__("lightgbm").log_evaluation(0),
        ],
    )
    elapsed = time.time() - t0
    best_iter = clf.best_iteration_ if hasattr(clf, "best_iteration_") else clf.n_estimators
    print(f"Training complete in {elapsed:.1f}s (best iteration: {best_iter})")

    # Evaluate on test set
    y_proba = clf.predict_proba(X_test)[:, 1]
    y_pred = (y_proba >= 0.5).astype(int)

    clf_metrics = print_classification_report(y_test.values, y_pred, y_proba)
    trading_metrics = compute_trading_metrics(y_test.values, y_proba, threshold=0.5)

    # Plots
    os.makedirs(MODEL_DIR, exist_ok=True)
    plot_confusion_matrix(y_test.values, y_pred, cm_path)
    auc = plot_roc_curve(y_test.values, y_proba, roc_path)

    # Feature importance
    importances = clf.feature_importances_
    indices = np.argsort(importances)[::-1]
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(range(len(importances)), importances[indices], color="#3fb950")
    ax.set_yticks(range(len(importances)))
    ax.set_yticklabels([feature_cols[i] for i in indices])
    ax.invert_yaxis()
    ax.set_xlabel("Feature Importance (split)")
    ax.set_title("LightGBM Feature Importance")
    fig.tight_layout()
    fig.savefig(importance_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Feature importance plot saved to {importance_path}")

    # Save model
    with open(model_path, "wb") as f:
        pickle.dump({"model": clf, "feature_cols": feature_cols, "target": TARGET}, f)
    print(f"\nModel saved to {model_path}")

    # Save metrics
    all_metrics = {
        "classification": clf_metrics,
        "trading": trading_metrics,
        "model_params": {k: v for k, v in clf.get_params().items() if not callable(v)},
        "training_time_seconds": round(elapsed, 2),
        "best_iteration": best_iter,
        "feature_cols": feature_cols,
        "data": {
            "train_rows": len(df_train),
            "val_rows": len(df_val),
            "test_rows": len(df_test),
        },
    }
    save_metrics(all_metrics, metrics_path)
    return clf_metrics


def main():
    parser = argparse.ArgumentParser(description="Train LightGBM pump detector")
    parser.add_argument("--train", type=str, required=True, help="Path to train parquet")
    parser.add_argument("--val", type=str, required=True, help="Path to validation parquet")
    parser.add_argument("--test", type=str, required=True, help="Path to test parquet")
    args = parser.parse_args()

    lgbm_metrics = train(args.train, args.val, args.test)

    # Load XGBoost metrics for comparison
    xgb_baseline_path = os.path.join(MODEL_DIR, "xgb_metrics.json")
    xgb_tuned_path = os.path.join(MODEL_DIR, "xgb_tuned_metrics.json")

    print("\n" + "=" * 60)
    print("MODEL COMPARISON (Test Set)")
    print("=" * 60)
    fmt = "{:<25s} {:>10s} {:>10s} {:>10s} {:>10s}"
    print(fmt.format("Model", "AUC", "F1", "Precision", "Recall"))
    print("-" * 65)

    for label, path in [("XGBoost Baseline", xgb_baseline_path), ("XGBoost Tuned", xgb_tuned_path)]:
        if os.path.exists(path):
            with open(path) as f:
                m = json.load(f)["classification"]
            print(fmt.format(label, f"{m['roc_auc']:.4f}", f"{m['f1']:.4f}", f"{m['precision']:.4f}", f"{m['recall']:.4f}"))

    print(fmt.format(
        "LightGBM Baseline",
        f"{lgbm_metrics['roc_auc']:.4f}",
        f"{lgbm_metrics['f1']:.4f}",
        f"{lgbm_metrics['precision']:.4f}",
        f"{lgbm_metrics['recall']:.4f}",
    ))
    print("=" * 65)


if __name__ == "__main__":
    main()
