#!/usr/bin/env python3
"""
LSTM sequence encoder for meme coin pump detection.
Input: sequences of features at 1-min intervals for the first N minutes after token listing
Output: probability of 2x pump within 2 hours

Usage:
  python ml/train_lstm.py
  python ml/train_lstm.py --seq-len 15 --epochs 100 --batch-size 64
  python ml/train_lstm.py --data-dir ml/raw --seq-len 10
"""

from __future__ import annotations

import argparse
import glob
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
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

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
MODEL_PATH = os.path.join(MODEL_DIR, "lstm_pump.pt")
CURVES_PATH = os.path.join(MODEL_DIR, "lstm_training_curves.png")
METRICS_PATH = os.path.join(MODEL_DIR, "lstm_metrics.json")

# Per-timestep features from OHLCV data
SEQUENCE_FEATURES = [
    "price_change_pct",   # % change from open to this candle's close
    "volume_usd",         # Volume in USD for this candle
    "buy_sell_ratio",     # Approximated from candle direction
    "liquidity_usd",      # Pool reserve (constant or interpolated)
    "vol_liq_ratio",      # volume / liquidity for this candle
    "age_minutes",        # Minutes since listing
    "mcap_usd",           # Market cap at this point
    "txn_count",          # Estimated from volume patterns
]


# ---------------------------------------------------------------------------
# Model Architecture
# ---------------------------------------------------------------------------
class SelfAttention(nn.Module):
    """Learned attention over LSTM timestep outputs."""

    def __init__(self, hidden_size: int):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.Tanh(),
            nn.Linear(hidden_size // 2, 1),
        )

    def forward(self, lstm_output: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            lstm_output: (batch, seq_len, hidden_size)
        Returns:
            context: (batch, hidden_size) — weighted sum
            weights: (batch, seq_len) — attention weights
        """
        scores = self.attn(lstm_output).squeeze(-1)   # (batch, seq_len)
        weights = torch.softmax(scores, dim=1)         # (batch, seq_len)
        context = torch.bmm(weights.unsqueeze(1), lstm_output).squeeze(1)  # (batch, hidden)
        return context, weights


class PumpLSTM(nn.Module):
    """Bidirectional LSTM with self-attention for pump detection."""

    def __init__(
        self,
        input_size: int = 8,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        # Input normalization
        self.input_norm = nn.LayerNorm(input_size)

        # Bidirectional LSTM encoder
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=True,
            batch_first=True,
        )

        # Self-attention over LSTM outputs (bidirectional = 2 * hidden_size)
        self.attention = SelfAttention(hidden_size * 2)

        # FC classifier head: hidden -> 64 -> 1 (sigmoid)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size * 2, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (batch, seq_len, input_size)
        Returns:
            logits: (batch, 1) — raw logits (apply sigmoid for probability)
            attn_weights: (batch, seq_len) — which timesteps matter most
        """
        x = self.input_norm(x)
        lstm_out, _ = self.lstm(x)          # (batch, seq_len, hidden*2)
        context, attn_w = self.attention(lstm_out)  # (batch, hidden*2)
        logits = self.classifier(context)   # (batch, 1)
        return logits, attn_w


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class PumpSequenceDataset(Dataset):
    """Dataset for pump detection from minute-level sequences."""

    def __init__(
        self,
        sequences: np.ndarray,
        labels: np.ndarray,
    ):
        """
        Args:
            sequences: (N, seq_len, n_features) float array
            labels: (N,) int array (0/1)
        """
        self.sequences = torch.tensor(sequences, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.sequences[idx], self.labels[idx]


# ---------------------------------------------------------------------------
# Data loading from raw JSON files
# ---------------------------------------------------------------------------
def load_sequences_from_raw(
    data_dir: str,
    seq_len: int = 15,
) -> tuple[np.ndarray, np.ndarray]:
    """Load per-coin minute-level sequences from GeckoTerminal OHLCV JSONs.

    Each JSON has: {ohlcv: [[ts, o, h, l, c, vol], ...], pool_details: {...}}

    Returns:
        sequences: (N, seq_len, 8) array
        labels: (N,) array of 0/1
    """
    pattern = os.path.join(data_dir, "*.json")
    files = [f for f in glob.glob(pattern) if not f.endswith("manifest.json")]

    if not files:
        return np.array([]), np.array([])

    sequences = []
    labels = []

    for fpath in files:
        with open(fpath) as f:
            pool = json.load(f)

        ohlcv = pool.get("ohlcv") or []
        details = pool.get("pool_details") or {}

        if len(ohlcv) < seq_len:
            continue

        # Sort by timestamp ascending
        ohlcv = sorted(ohlcv, key=lambda c: c[0])
        candles = ohlcv[:seq_len]

        open_price = float(candles[0][1]) if candles[0][1] else 0
        if open_price <= 0:
            continue

        # Extract liquidity and mcap (constant for the pool)
        liq = float(details.get("reserve_in_usd") or 0)
        mcap = float(details.get("fdv_usd") or details.get("market_cap_usd") or 0)

        # Build sequence features per minute
        seq = []
        for i, c in enumerate(candles):
            ts, o, h, low, close, vol = c[0], float(c[1] or 0), float(c[2] or 0), float(c[3] or 0), float(c[4] or 0), float(c[5] or 0)

            price_change_pct = (close / open_price - 1) * 100 if open_price > 0 else 0
            buy_sell_ratio = 1.0 if close >= o else 0.0  # Approximation from candle color
            vol_liq_ratio = vol / liq if liq > 0 else 0
            age_minutes = i + 1
            # Approximate txn count from volume (rough heuristic)
            avg_trade_size = 50  # $50 avg trade in meme coins
            txn_count = vol / avg_trade_size if avg_trade_size > 0 else 0

            seq.append([
                price_change_pct,
                vol,
                buy_sell_ratio,
                liq,
                vol_liq_ratio,
                age_minutes,
                mcap,
                txn_count,
            ])

        # Label: did it pump 2x within 1h?
        closes_1h = [float(c[4]) for c in ohlcv[:60] if c[4]]
        max_gain = max(closes_1h) / open_price - 1 if closes_1h else 0
        label = 1 if max_gain >= 1.0 else 0  # 2x = 100% gain

        sequences.append(seq)
        labels.append(label)

    if not sequences:
        return np.array([]), np.array([])

    return np.array(sequences, dtype=np.float32), np.array(labels, dtype=np.int64)


def generate_synthetic_sequences(
    n_samples: int = 1500,
    seq_len: int = 15,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate synthetic time-series sequences for development/testing.

    Simulates meme coin price patterns:
    - Pumps: exponential growth + high volume in early minutes
    - Non-pumps: flat/declining with lower volume
    """
    rng = np.random.RandomState(seed)
    n_pump = int(n_samples * 0.12)
    n_nopump = n_samples - n_pump

    sequences = []
    labels = []

    for label, count in [(0, n_nopump), (1, n_pump)]:
        for _ in range(count):
            liq = rng.lognormal(10.5, 1.0)    # $30k-$100k
            mcap = liq * rng.uniform(2, 10)
            base_vol = rng.lognormal(7, 1.0)   # $1k-$10k/min

            seq = []
            cumulative_change = 0.0

            for t in range(seq_len):
                if label == 1:  # Pump pattern
                    # Exponential-ish growth with noise
                    growth = rng.lognormal(1.5, 0.5) * (1 + t * 0.3)
                    cumulative_change += growth
                    vol = base_vol * rng.lognormal(0.5, 0.3) * (1 + t * 0.5)
                    buy_sell = rng.beta(5, 2)  # More buys
                else:  # Non-pump pattern
                    change = rng.normal(0, 3)
                    cumulative_change += change
                    vol = base_vol * rng.lognormal(0, 0.5) * max(0.3, 1 - t * 0.03)
                    buy_sell = rng.beta(2, 3)  # More sells

                vol_liq = vol / liq if liq > 0 else 0
                age = t + 1
                txn_count = vol / 50  # ~$50 avg trade

                seq.append([
                    cumulative_change,
                    vol,
                    buy_sell,
                    liq,
                    vol_liq,
                    age,
                    mcap,
                    txn_count,
                ])

            sequences.append(seq)
            labels.append(label)

    sequences = np.array(sequences, dtype=np.float32)
    labels = np.array(labels, dtype=np.int64)

    # Shuffle
    perm = rng.permutation(len(labels))
    return sequences[perm], labels[perm]


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------
def normalize_sequences(
    train: np.ndarray,
    val: np.ndarray,
    test: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """Z-score normalize using train statistics."""
    # Flatten to (N*seq_len, features) for stats
    flat = train.reshape(-1, train.shape[-1])
    mean = flat.mean(axis=0)
    std = flat.std(axis=0)
    std[std < 1e-8] = 1.0  # Avoid division by zero

    stats = {"mean": mean.tolist(), "std": std.tolist()}

    train_norm = (train - mean) / std
    val_norm = (val - mean) / std
    test_norm = (test - mean) / std

    return train_norm, val_norm, test_norm, stats


# ---------------------------------------------------------------------------
# Sapienza dataset loading
# ---------------------------------------------------------------------------
SAPIENZA_PATH = os.path.expanduser(
    "~/repos/sapienza-dataset/labeled_features/features_15S.csv.gz"
)

SAPIENZA_FEATURES = [
    "std_rush_order",
    "avg_rush_order",
    "std_trades",
    "std_volume",
    "avg_volume",
    "std_price",
    "avg_price",
    "avg_price_max",
    "hour_sin",
    "hour_cos",
]

SAPIENZA_MODEL_PATH = os.path.join(MODEL_DIR, "lstm_sapienza.pt")
SAPIENZA_CURVES_PATH = os.path.join(MODEL_DIR, "lstm_sapienza_curves.png")
SAPIENZA_METRICS_PATH = os.path.join(MODEL_DIR, "lstm_sapienza_metrics.json")
SAPIENZA_SCALER_PATH = os.path.join(MODEL_DIR, "lstm_sapienza_scaler.pkl")


def load_sapienza_sequences(
    seq_len: int = 20,
    max_sequences: int = 100_000,
) -> tuple[np.ndarray, np.ndarray]:
    """Build pump/non-pump sequences from the Sapienza 15-second dataset.

    Pump windows: N rows immediately before a gt 0→1 transition.
    Non-pump windows: random N-row windows from contiguous gt=0 regions.

    Returns:
        sequences: (N_total, seq_len, 10) float array
        labels: (N_total,) int array (0/1)
    """
    print(f"Loading Sapienza dataset from {SAPIENZA_PATH} ...")
    df = pd.read_csv(SAPIENZA_PATH)
    print(f"  {len(df):,} rows, {df['symbol'].nunique()} symbols")

    feat_cols = SAPIENZA_FEATURES
    pump_seqs: list[np.ndarray] = []
    non_pump_seqs: list[np.ndarray] = []

    for sym, grp in df.groupby("symbol"):
        grp = grp.sort_values("date").reset_index(drop=True)
        gt = grp["gt"].values
        feats = grp[feat_cols].values.astype(np.float32)

        # --- Pump windows: rows before each gt 0→1 transition ---
        transitions = np.where((gt[:-1] == 0) & (gt[1:] == 1))[0]
        for t_idx in transitions:
            # t_idx is the last 0 before the 1; we want seq_len rows ending at t_idx
            start = t_idx - seq_len + 1
            if start < 0:
                continue
            window = feats[start : t_idx + 1]
            if len(window) == seq_len:
                pump_seqs.append(window)

        # --- Non-pump windows: random windows from gt=0 stretches ---
        # Find contiguous gt=0 runs
        is_zero = gt == 0
        # We want to sample windows of length seq_len from pure gt=0 regions
        # Build list of valid start indices where the full window is gt=0
        valid_starts: list[int] = []
        run_start = None
        for i in range(len(gt)):
            if is_zero[i]:
                if run_start is None:
                    run_start = i
            else:
                if run_start is not None and (i - run_start) >= seq_len:
                    # This run has enough rows; add valid starts
                    valid_starts.extend(range(run_start, i - seq_len + 1))
                run_start = None
        # Handle run ending at end of array
        if run_start is not None and (len(gt) - run_start) >= seq_len:
            valid_starts.extend(range(run_start, len(gt) - seq_len + 1))

        if not valid_starts:
            continue

        # Sample up to same number of non-pump windows as pump events for this symbol
        n_pump_this = len(transitions)
        rng = np.random.RandomState(hash(sym) % (2**31))
        n_sample = min(n_pump_this * 3, len(valid_starts))  # oversample 3x, trim later
        chosen = rng.choice(valid_starts, size=n_sample, replace=len(valid_starts) < n_sample)
        for s in chosen:
            window = feats[s : s + seq_len]
            if len(window) == seq_len:
                non_pump_seqs.append(window)

    n_pump = len(pump_seqs)
    n_non_pump = len(non_pump_seqs)
    print(f"  Extracted {n_pump} pump sequences, {n_non_pump} non-pump sequences")

    if n_pump == 0:
        raise ValueError("No pump sequences found in Sapienza dataset")

    # Balance: match non-pump count to pump count
    rng = np.random.RandomState(42)
    if n_non_pump > n_pump:
        idx = rng.choice(n_non_pump, size=n_pump, replace=False)
        non_pump_seqs = [non_pump_seqs[i] for i in idx]
    elif n_non_pump < n_pump:
        idx = rng.choice(n_non_pump, size=n_pump, replace=True)
        non_pump_seqs = [non_pump_seqs[i] for i in idx]

    # Combine
    sequences = np.array(pump_seqs + non_pump_seqs, dtype=np.float32)
    labels = np.array([1] * len(pump_seqs) + [0] * len(non_pump_seqs), dtype=np.int64)

    # Subsample if over max
    total = len(labels)
    if total > max_sequences:
        idx = rng.choice(total, size=max_sequences, replace=False)
        sequences = sequences[idx]
        labels = labels[idx]

    # Shuffle
    perm = rng.permutation(len(labels))
    sequences = sequences[perm]
    labels = labels[perm]

    print(f"  Final: {len(labels)} sequences ({labels.sum()} pump, {len(labels) - labels.sum()} non-pump)")
    return sequences, labels


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------
def train_lstm(
    data_dir: str | None = None,
    seq_len: int = 15,
    epochs: int = 100,
    batch_size: int = 64,
    lr: float = 1e-3,
    patience: int = 15,
) -> None:
    """Full training pipeline for PumpLSTM."""
    print("=" * 60)
    print("LSTM Meme Coin Pump Detector")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load data
    raw_dir = data_dir or os.path.join(os.path.dirname(os.path.abspath(__file__)), "raw")
    sequences, labels = load_sequences_from_raw(raw_dir, seq_len=seq_len)

    if len(sequences) < 50:
        print(f"Only {len(sequences)} raw sequences — using synthetic data for development")
        sequences, labels = generate_synthetic_sequences(n_samples=1500, seq_len=seq_len)

    print(f"Total samples: {len(labels)} ({labels.sum()} positive, {len(labels) - labels.sum()} negative)")

    # Train/val/test split (70/15/15)
    n = len(labels)
    idx = np.random.RandomState(42).permutation(n)
    n_train = int(0.7 * n)
    n_val = int(0.15 * n)

    train_idx = idx[:n_train]
    val_idx = idx[n_train:n_train + n_val]
    test_idx = idx[n_train + n_val:]

    X_train, y_train = sequences[train_idx], labels[train_idx]
    X_val, y_val = sequences[val_idx], labels[val_idx]
    X_test, y_test = sequences[test_idx], labels[test_idx]

    print(f"Split: train={len(y_train)}, val={len(y_val)}, test={len(y_test)}")

    # Normalize
    X_train, X_val, X_test, norm_stats = normalize_sequences(X_train, X_val, X_test)

    # Class imbalance: weighted sampler for training
    pos_count = int(y_train.sum())
    neg_count = len(y_train) - pos_count
    class_weights = [1.0 / neg_count if neg_count > 0 else 1.0,
                     1.0 / pos_count if pos_count > 0 else 1.0]
    sample_weights = [class_weights[int(y)] for y in y_train]
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(y_train), replacement=True)

    # pos_weight for BCE loss
    pos_weight = torch.tensor([neg_count / pos_count if pos_count > 0 else 1.0], device=device)

    # DataLoaders
    train_ds = PumpSequenceDataset(X_train, y_train)
    val_ds = PumpSequenceDataset(X_val, y_val)
    test_ds = PumpSequenceDataset(X_test, y_test)

    train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    # Model
    model = PumpLSTM(
        input_size=len(SEQUENCE_FEATURES),
        hidden_size=128,
        num_layers=2,
        dropout=0.3,
    ).to(device)

    print(f"\nModel parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Loss and optimizer
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5,
    )

    # Training loop
    best_val_loss = float("inf")
    best_epoch = 0
    train_losses = []
    val_losses = []
    val_aucs = []

    print(f"\nTraining for up to {epochs} epochs (patience={patience})...\n")
    t0 = time.time()

    for epoch in range(epochs):
        # Train
        model.train()
        epoch_loss = 0.0
        n_batches = 0

        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            logits, _ = model(X_batch)
            loss = criterion(logits.squeeze(-1), y_batch)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        avg_train_loss = epoch_loss / max(n_batches, 1)
        train_losses.append(avg_train_loss)

        # Validate
        model.eval()
        val_loss = 0.0
        val_preds = []
        val_labels = []
        n_val_batches = 0

        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)

                logits, _ = model(X_batch)
                loss = criterion(logits.squeeze(-1), y_batch)
                val_loss += loss.item()
                n_val_batches += 1

                probs = torch.sigmoid(logits.squeeze(-1))
                val_preds.extend(probs.cpu().numpy())
                val_labels.extend(y_batch.cpu().numpy())

        avg_val_loss = val_loss / max(n_val_batches, 1)
        val_losses.append(avg_val_loss)

        # AUC
        from sklearn.metrics import roc_auc_score
        try:
            val_auc = roc_auc_score(val_labels, val_preds)
        except ValueError:
            val_auc = 0.5
        val_aucs.append(val_auc)

        scheduler.step(avg_val_loss)

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(
                f"Epoch {epoch+1:3d}/{epochs} | "
                f"Train Loss: {avg_train_loss:.4f} | "
                f"Val Loss: {avg_val_loss:.4f} | "
                f"Val AUC: {val_auc:.4f} | "
                f"LR: {optimizer.param_groups[0]['lr']:.6f}"
            )

        # Early stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_epoch = epoch + 1
            # Save best model
            os.makedirs(MODEL_DIR, exist_ok=True)
            torch.save({
                "model_state_dict": model.state_dict(),
                "norm_stats": norm_stats,
                "seq_len": seq_len,
                "input_size": len(SEQUENCE_FEATURES),
                "hidden_size": 128,
                "num_layers": 2,
                "dropout": 0.3,
                "feature_names": SEQUENCE_FEATURES,
                "best_epoch": best_epoch,
                "best_val_loss": best_val_loss,
            }, MODEL_PATH)
        elif epoch - best_epoch + 1 >= patience:
            print(f"\nEarly stopping at epoch {epoch+1} (best: {best_epoch})")
            break

    elapsed = time.time() - t0
    print(f"\nTraining complete in {elapsed:.1f}s (best epoch: {best_epoch})")

    # Plot training curves
    _plot_training_curves(train_losses, val_losses, val_aucs, CURVES_PATH)

    # Load best model for evaluation
    checkpoint = torch.load(MODEL_PATH, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])

    # Evaluate on test set
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(device)
            logits, _ = model(X_batch)
            probs = torch.sigmoid(logits.squeeze(-1))
            all_preds.extend(probs.cpu().numpy())
            all_labels.extend(y_batch.numpy())

    y_true = np.array(all_labels)
    y_proba = np.array(all_preds)
    y_pred = (y_proba >= 0.5).astype(int)

    # Metrics
    clf_metrics = print_classification_report(y_true, y_pred, y_proba)
    trading_metrics = compute_trading_metrics(y_true, y_proba, threshold=0.5)

    plot_confusion_matrix(y_true, y_pred, os.path.join(MODEL_DIR, "lstm_confusion_matrix.png"))
    plot_roc_curve(y_true, y_proba, os.path.join(MODEL_DIR, "lstm_roc_curve.png"))

    all_metrics = {
        "classification": clf_metrics,
        "trading": trading_metrics,
        "training_time_seconds": round(elapsed, 2),
        "best_epoch": best_epoch,
        "best_val_loss": round(best_val_loss, 6),
        "seq_len": seq_len,
        "device": str(device),
    }
    save_metrics(all_metrics, METRICS_PATH)

    print(f"\nModel saved to {MODEL_PATH}")


def _plot_training_curves(
    train_losses: list[float],
    val_losses: list[float],
    val_aucs: list[float],
    save_path: str,
) -> None:
    """Plot loss and AUC curves over training."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    epochs = range(1, len(train_losses) + 1)

    ax1.plot(epochs, train_losses, label="Train Loss", color="#58a6ff")
    ax1.plot(epochs, val_losses, label="Val Loss", color="#f85149")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Training & Validation Loss")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(epochs, val_aucs, label="Val AUC", color="#3fb950")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("AUC")
    ax2.set_title("Validation AUC")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([0, 1])

    fig.tight_layout()
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Training curves saved to {save_path}")


# ---------------------------------------------------------------------------
# Sapienza-specific training
# ---------------------------------------------------------------------------
def train_lstm_sapienza(
    seq_len: int = 20,
    epochs: int = 30,
    batch_size: int = 64,
    lr: float = 1e-3,
    patience: int = 5,
) -> None:
    """Train PumpLSTM on real Sapienza 15-second time-series data."""
    print("=" * 60)
    print("LSTM Pump Detector — Sapienza Dataset (584K rows)")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load sequences
    sequences, labels = load_sapienza_sequences(seq_len=seq_len)

    # Train/val/test split (70/15/15)
    # TODO(audit-CRITICAL): Replace random permutation with temporal split.
    # Random shuffle leaks future data from the same symbol into training.
    # Sort sequences by symbol + time, then use chronological cutoffs.
    n = len(labels)
    idx = np.random.RandomState(42).permutation(n)
    n_train = int(0.7 * n)
    n_val = int(0.15 * n)

    train_idx = idx[:n_train]
    val_idx = idx[n_train : n_train + n_val]
    test_idx = idx[n_train + n_val :]

    X_train, y_train = sequences[train_idx], labels[train_idx]
    X_val, y_val = sequences[val_idx], labels[val_idx]
    X_test, y_test = sequences[test_idx], labels[test_idx]

    print(f"Split: train={len(y_train)}, val={len(y_val)}, test={len(y_test)}")
    print(f"  Train: {y_train.sum()} pump, {len(y_train) - y_train.sum()} non-pump")

    # Fit StandardScaler on train data and save
    n_features = X_train.shape[-1]
    flat_train = X_train.reshape(-1, n_features)
    scaler = StandardScaler()
    scaler.fit(flat_train)

    os.makedirs(MODEL_DIR, exist_ok=True)
    with open(SAPIENZA_SCALER_PATH, "wb") as f:
        pickle.dump(scaler, f)
    print(f"Scaler saved to {SAPIENZA_SCALER_PATH}")

    # Apply scaler
    X_train = scaler.transform(X_train.reshape(-1, n_features)).reshape(X_train.shape)
    X_val = scaler.transform(X_val.reshape(-1, n_features)).reshape(X_val.shape)
    X_test = scaler.transform(X_test.reshape(-1, n_features)).reshape(X_test.shape)

    # Class balance: weighted sampler
    pos_count = int(y_train.sum())
    neg_count = len(y_train) - pos_count
    class_weights = [
        1.0 / neg_count if neg_count > 0 else 1.0,
        1.0 / pos_count if pos_count > 0 else 1.0,
    ]
    sample_weights = [class_weights[int(y)] for y in y_train]
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(y_train), replacement=True)

    pos_weight = torch.tensor(
        [neg_count / pos_count if pos_count > 0 else 1.0], device=device
    )

    # DataLoaders
    train_ds = PumpSequenceDataset(X_train, y_train)
    val_ds = PumpSequenceDataset(X_val, y_val)
    test_ds = PumpSequenceDataset(X_test, y_test)

    train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    # Model: input_size=10 for Sapienza features
    model = PumpLSTM(
        input_size=len(SAPIENZA_FEATURES),
        hidden_size=128,
        num_layers=2,
        dropout=0.3,
    ).to(device)

    print(f"\nModel parameters: {sum(p.numel() for p in model.parameters()):,}")

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3,
    )

    # Training loop
    best_val_loss = float("inf")
    best_epoch = 0
    train_losses: list[float] = []
    val_losses: list[float] = []
    val_aucs: list[float] = []

    print(f"\nTraining for up to {epochs} epochs (patience={patience})...\n")
    t0 = time.time()

    for epoch in range(epochs):
        # Train
        model.train()
        epoch_loss = 0.0
        n_batches = 0

        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            logits, _ = model(X_batch)
            loss = criterion(logits.squeeze(-1), y_batch)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        avg_train_loss = epoch_loss / max(n_batches, 1)
        train_losses.append(avg_train_loss)

        # Validate
        model.eval()
        val_loss = 0.0
        val_preds: list[float] = []
        val_labels: list[float] = []
        n_val_batches = 0

        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)

                logits, _ = model(X_batch)
                loss = criterion(logits.squeeze(-1), y_batch)
                val_loss += loss.item()
                n_val_batches += 1

                probs = torch.sigmoid(logits.squeeze(-1))
                val_preds.extend(probs.cpu().numpy())
                val_labels.extend(y_batch.cpu().numpy())

        avg_val_loss = val_loss / max(n_val_batches, 1)
        val_losses.append(avg_val_loss)

        from sklearn.metrics import roc_auc_score

        try:
            val_auc = roc_auc_score(val_labels, val_preds)
        except ValueError:
            val_auc = 0.5
        val_aucs.append(val_auc)

        scheduler.step(avg_val_loss)

        print(
            f"Epoch {epoch + 1:3d}/{epochs} | "
            f"Train Loss: {avg_train_loss:.4f} | "
            f"Val Loss: {avg_val_loss:.4f} | "
            f"Val AUC: {val_auc:.4f} | "
            f"LR: {optimizer.param_groups[0]['lr']:.6f}"
        )

        # Early stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_epoch = epoch + 1
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "seq_len": seq_len,
                    "input_size": len(SAPIENZA_FEATURES),
                    "hidden_size": 128,
                    "num_layers": 2,
                    "dropout": 0.3,
                    "feature_names": SAPIENZA_FEATURES,
                    "best_epoch": best_epoch,
                    "best_val_loss": best_val_loss,
                },
                SAPIENZA_MODEL_PATH,
            )
        elif epoch + 1 - best_epoch >= patience:
            print(f"\nEarly stopping at epoch {epoch + 1} (best: {best_epoch})")
            break

    elapsed = time.time() - t0
    print(f"\nTraining complete in {elapsed:.1f}s (best epoch: {best_epoch})")

    # Plot training curves
    _plot_training_curves(train_losses, val_losses, val_aucs, SAPIENZA_CURVES_PATH)

    # Load best model for evaluation
    checkpoint = torch.load(SAPIENZA_MODEL_PATH, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])

    # Evaluate on test set
    model.eval()
    all_preds: list[float] = []
    all_labels: list[float] = []

    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(device)
            logits, _ = model(X_batch)
            probs = torch.sigmoid(logits.squeeze(-1))
            all_preds.extend(probs.cpu().numpy())
            all_labels.extend(y_batch.numpy())

    y_true = np.array(all_labels)
    y_proba = np.array(all_preds)
    y_pred = (y_proba >= 0.5).astype(int)

    # Metrics
    clf_metrics = print_classification_report(y_true, y_pred, y_proba)
    trading_metrics = compute_trading_metrics(y_true, y_proba, threshold=0.5)

    plot_confusion_matrix(
        y_true, y_pred, os.path.join(MODEL_DIR, "lstm_sapienza_confusion_matrix.png")
    )
    plot_roc_curve(y_true, y_proba, os.path.join(MODEL_DIR, "lstm_sapienza_roc_curve.png"))

    all_metrics = {
        "data_source": "sapienza",
        "classification": clf_metrics,
        "trading": trading_metrics,
        "training_time_seconds": round(elapsed, 2),
        "best_epoch": best_epoch,
        "best_val_loss": round(best_val_loss, 6),
        "seq_len": seq_len,
        "input_size": len(SAPIENZA_FEATURES),
        "device": str(device),
    }
    save_metrics(all_metrics, SAPIENZA_METRICS_PATH)

    print(f"\nModel saved to {SAPIENZA_MODEL_PATH}")
    print(f"Scaler saved to {SAPIENZA_SCALER_PATH}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Train LSTM pump detector")
    parser.add_argument("--data-dir", type=str, default=None, help="Directory with raw JSON pool files")
    parser.add_argument("--data-source", type=str, default=None, choices=["sapienza"], help="Use a named dataset")
    parser.add_argument("--seq-len", type=int, default=15, help="Sequence length in minutes")
    parser.add_argument("--epochs", type=int, default=100, help="Max training epochs")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--patience", type=int, default=15, help="Early stopping patience")
    args = parser.parse_args()

    if args.data_source == "sapienza":
        train_lstm_sapienza(
            seq_len=args.seq_len if args.seq_len != 15 else 20,
            epochs=args.epochs if args.epochs != 100 else 30,
            batch_size=args.batch_size,
            lr=args.lr,
            patience=args.patience if args.patience != 15 else 5,
        )
    else:
        train_lstm(
            data_dir=args.data_dir,
            seq_len=args.seq_len,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            patience=args.patience,
        )


if __name__ == "__main__":
    main()
