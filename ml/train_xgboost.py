#!/usr/bin/env python3
"""
XGBoost classifier for meme coin pump detection on real MemeTrans data.

Usage:
  # Baseline
  python ml/train_xgboost.py --train ml/data/train.parquet --val ml/data/val.parquet --test ml/data/test.parquet

  # Optuna tuning
  python ml/train_xgboost.py --train ml/data/train.parquet --val ml/data/val.parquet --test ml/data/test.parquet --tune

  # SHAP analysis on saved model
  python ml/train_xgboost.py --test ml/data/test.parquet --shap ml/models/xgb_tuned.pkl
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
TARGET = "label"
DROP_COLS = {"label", "source"}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_parquet_splits(
    train_path: str,
    val_path: str,
    test_path: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load pre-split parquet files, dropping source column."""
    df_train = pd.read_parquet(train_path)
    df_val = pd.read_parquet(val_path)
    df_test = pd.read_parquet(test_path)

    print(f"Loaded: train={len(df_train)}, val={len(df_val)}, test={len(df_test)}")
    return df_train, df_val, df_test


def get_feature_cols(df: pd.DataFrame) -> list[str]:
    """Auto-detect feature columns (everything except label and source)."""
    return [c for c in df.columns if c not in DROP_COLS]


def prepare_features(
    df: pd.DataFrame, feature_cols: list[str]
) -> tuple[pd.DataFrame, pd.Series]:
    """Extract X, y with proper type handling."""
    X = df[feature_cols].copy()
    for col in X.columns:
        X[col] = pd.to_numeric(X[col], errors="coerce")
    # TODO(audit-HIGH): Save per-column medians in model bundle so predict.py
    # can use them at inference instead of filling NaN with 0 (train/serve skew).
    # E.g., save {"imputer_medians": X.median().to_dict()} alongside the model.
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
    n_trials: int = 30,
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
            "scale_pos_weight": trial.suggest_float(
                "scale_pos_weight", base_spw * 0.5, base_spw * 2.0
            ),
        }

        clf = XGBClassifier(
            **params,
            eval_metric="logloss",
            random_state=42,
            verbosity=0,
        )
        clf.fit(
            X_train,
            y_train,
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
def run_shap_analysis(
    model,
    X: pd.DataFrame,
    save_path: str,
    top_n: int = 15,
) -> None:
    """Run SHAP analysis, save beeswarm plot and print top features."""
    import shap

    print(f"\nRunning SHAP analysis on {len(X)} samples...")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    # Mean absolute SHAP values per feature
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    feature_importance = pd.Series(mean_abs_shap, index=X.columns).sort_values(
        ascending=False
    )

    print(f"\nTop {top_n} most important features (mean |SHAP|):")
    print("-" * 50)
    for i, (feat, val) in enumerate(feature_importance.head(top_n).items(), 1):
        print(f"  {i:2d}. {feat:<35s} {val:.4f}")

    # Save beeswarm plot
    fig, ax = plt.subplots(figsize=(12, 8))
    shap.summary_plot(shap_values, X, show=False, plot_size=None, max_display=top_n)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close("all")
    print(f"SHAP importance plot saved to {save_path}")


# ---------------------------------------------------------------------------
# Main training function
# ---------------------------------------------------------------------------
def train(
    train_path: str,
    val_path: str,
    test_path: str,
    tune: bool = False,
) -> None:
    """Train XGBoost classifier with full evaluation pipeline."""
    suffix = "tuned" if tune else "baseline"
    model_path = os.path.join(MODEL_DIR, f"xgb_{suffix}.pkl")
    metrics_path = os.path.join(MODEL_DIR, f"xgb_{suffix}_metrics.json" if tune else "xgb_metrics.json")
    cm_path = os.path.join(MODEL_DIR, f"xgb_{suffix}_confusion_matrix.png")
    roc_path = os.path.join(MODEL_DIR, f"xgb_{suffix}_roc_curve.png")
    importance_path = os.path.join(MODEL_DIR, f"xgb_{suffix}_feature_importance.png")

    print("=" * 60)
    print(f"XGBoost Meme Coin Pump Detector ({suffix})")
    print("=" * 60)

    # Load data
    df_train, df_val, df_test = load_parquet_splits(train_path, val_path, test_path)
    feature_cols = get_feature_cols(df_train)
    print(f"Features ({len(feature_cols)}): {feature_cols}")

    X_train, y_train = prepare_features(df_train, feature_cols)
    X_val, y_val = prepare_features(df_val, feature_cols)
    X_test, y_test = prepare_features(df_test, feature_cols)

    print(f"\nClass balance (train): {y_train.value_counts().to_dict()}")
    print(f"Class balance (val):   {y_val.value_counts().to_dict()}")
    print(f"Class balance (test):  {y_test.value_counts().to_dict()}")

    # Handle class imbalance
    neg_count = int((y_train == 0).sum())
    pos_count = int((y_train == 1).sum())
    scale_pos_weight = neg_count / pos_count if pos_count > 0 else 1.0
    print(f"scale_pos_weight: {scale_pos_weight:.4f}")

    # Hyperparameter tuning or defaults
    if tune:
        print("\nRunning Optuna hyperparameter search (30 trials)...")
        best_params = run_optuna_search(
            X_train, y_train, X_val, y_val, n_trials=30
        )
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
        X_train,
        y_train,
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
    plot_confusion_matrix(y_test.values, y_pred, cm_path)
    auc = plot_roc_curve(y_test.values, y_proba, roc_path)

    # Built-in feature importance plot
    importances = clf.feature_importances_
    indices = np.argsort(importances)[::-1]
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(range(len(importances)), importances[indices], color="#58a6ff")
    ax.set_yticks(range(len(importances)))
    ax.set_yticklabels([feature_cols[i] for i in indices])
    ax.invert_yaxis()
    ax.set_xlabel("Feature Importance (gain)")
    ax.set_title(f"XGBoost Feature Importance ({suffix})")
    fig.tight_layout()
    fig.savefig(importance_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Feature importance plot saved to {importance_path}")

    # Save model
    with open(model_path, "wb") as f:
        pickle.dump(
            {"model": clf, "feature_cols": feature_cols, "target": TARGET}, f
        )
    print(f"\nModel saved to {model_path}")

    # Save metrics
    all_metrics = {
        "classification": clf_metrics,
        "trading": trading_metrics,
        "model_params": clf.get_params(),
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


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Train XGBoost pump detector")
    parser.add_argument("--train", type=str, default=None, help="Path to train parquet")
    parser.add_argument("--val", type=str, default=None, help="Path to validation parquet")
    parser.add_argument("--test", type=str, default=None, help="Path to test parquet")
    parser.add_argument("--tune", action="store_true", help="Run Optuna hyperparameter search (30 trials)")
    parser.add_argument("--shap", type=str, default=None, help="Path to saved model .pkl for SHAP analysis")
    args = parser.parse_args()

    # SHAP-only mode
    if args.shap:
        if not args.test:
            parser.error("--test is required for SHAP analysis")
        print("Loading model for SHAP analysis...")
        with open(args.shap, "rb") as f:
            bundle = pickle.load(f)
        model = bundle["model"]
        feature_cols = bundle["feature_cols"]

        df_test = pd.read_parquet(args.test)
        X_test, y_test = prepare_features(df_test, feature_cols)

        shap_path = os.path.join(MODEL_DIR, "xgb_shap_importance.png")
        run_shap_analysis(model, X_test, shap_path, top_n=15)
        return

    # Training mode
    if not args.train:
        parser.error("--train is required for training")
    train(
        train_path=args.train,
        val_path=args.val,
        test_path=args.test,
        tune=args.tune,
    )


if __name__ == "__main__":
    main()
