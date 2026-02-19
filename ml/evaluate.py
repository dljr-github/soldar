#!/usr/bin/env python3
"""Shared evaluation utilities for Soldar ML models."""

from __future__ import annotations

import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    save_path: str,
    labels: tuple[str, str] = ("No Pump", "Pump 2x"),
) -> None:
    """Plot and save a confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    ax.set(
        xticks=[0, 1],
        yticks=[0, 1],
        xticklabels=labels,
        yticklabels=labels,
        ylabel="True",
        xlabel="Predicted",
        title="Confusion Matrix",
    )
    # Annotate cells
    thresh = cm.max() / 2.0
    for i in range(2):
        for j in range(2):
            ax.text(
                j, i, f"{cm[i, j]}",
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black",
                fontsize=14,
            )
    fig.tight_layout()
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Confusion matrix saved to {save_path}")


def plot_roc_curve(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    save_path: str,
) -> float:
    """Plot ROC curve and return AUC score."""
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    auc = roc_auc_score(y_true, y_proba)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, color="#58a6ff", lw=2, label=f"ROC (AUC = {auc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5)
    ax.set(
        xlim=[0, 1],
        ylim=[0, 1.05],
        xlabel="False Positive Rate",
        ylabel="True Positive Rate",
        title="ROC Curve",
    )
    ax.legend(loc="lower right")
    fig.tight_layout()
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"ROC curve saved to {save_path} (AUC={auc:.3f})")
    return auc


def print_classification_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
) -> dict:
    """Print classification metrics and return as dict."""
    print("\n" + "=" * 60)
    print("CLASSIFICATION REPORT")
    print("=" * 60)
    print(classification_report(y_true, y_pred, target_names=["No Pump", "Pump 2x"]))

    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    try:
        auc = roc_auc_score(y_true, y_proba)
    except ValueError:
        auc = 0.0

    metrics = {
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1": round(f1, 4),
        "roc_auc": round(auc, 4),
        "n_samples": len(y_true),
        "n_positive": int(np.sum(y_true)),
        "n_negative": int(len(y_true) - np.sum(y_true)),
    }

    print(f"\nPrecision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    print(f"ROC-AUC:   {auc:.4f}")
    print(f"Samples:   {len(y_true)} ({np.sum(y_true)} positive, {len(y_true) - np.sum(y_true)} negative)")

    return metrics


def compute_trading_metrics(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    threshold: float = 0.5,
) -> dict:
    """Compute trading-specific evaluation metrics.

    Returns:
        dict with precision_at_threshold, recall_at_threshold,
        expected_value, coins_saved_from_rugs, etc.
    """
    y_pred = (y_proba >= threshold).astype(int)
    y_true = np.asarray(y_true, dtype=int)

    # Of coins we'd trade (predicted positive), what % actually pumped?
    predicted_positive = np.sum(y_pred)
    true_positive = np.sum((y_pred == 1) & (y_true == 1))
    trade_precision = true_positive / predicted_positive if predicted_positive > 0 else 0.0

    # Of coins that actually pumped, what % would we catch?
    actual_positive = np.sum(y_true)
    trade_recall = true_positive / actual_positive if actual_positive > 0 else 0.0

    # Expected value: assume pump = +100% gain, miss = -40% loss (stop loss)
    PUMP_GAIN = 1.0  # +100%
    RUG_LOSS = -0.4  # -40% stop loss
    if predicted_positive > 0:
        gains = true_positive * PUMP_GAIN
        losses = (predicted_positive - true_positive) * RUG_LOSS
        ev_per_trade = (gains + losses) / predicted_positive
    else:
        ev_per_trade = 0.0

    # Coins correctly rejected that were rugs (true negatives among negatives)
    actual_negative = len(y_true) - actual_positive
    true_negative = np.sum((y_pred == 0) & (y_true == 0))
    coins_saved = int(true_negative)

    metrics = {
        "threshold": threshold,
        "predicted_trades": int(predicted_positive),
        "actual_pumps": int(actual_positive),
        "true_positives": int(true_positive),
        "trade_precision": round(trade_precision, 4),
        "trade_recall": round(trade_recall, 4),
        "ev_per_trade": round(ev_per_trade, 4),
        "coins_saved_from_rugs": coins_saved,
        "total_samples": len(y_true),
    }

    print("\n" + "=" * 60)
    print(f"TRADING METRICS (threshold={threshold:.2f})")
    print("=" * 60)
    print(f"Would trade:    {predicted_positive}/{len(y_true)} coins")
    print(f"Correct trades: {true_positive}/{predicted_positive} ({trade_precision:.1%} precision)")
    print(f"Pumps caught:   {true_positive}/{actual_positive} ({trade_recall:.1%} recall)")
    print(f"EV per trade:   {ev_per_trade:+.2%} (assuming +100% pump / -40% stop)")
    print(f"Rugs avoided:   {coins_saved}/{actual_negative} correctly rejected")

    return metrics


def save_metrics(metrics: dict, save_path: str) -> None:
    """Save metrics dict to JSON."""
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    with open(save_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics saved to {save_path}")
