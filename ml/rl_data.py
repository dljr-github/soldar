"""Sequence data loader for the RL exit-timing environment.

Provides three loading strategies:
  1. load_raw_sequences  — parse OHLCV JSON files from ml/raw/
  2. generate_synthetic_sequences — create realistic pump/dump paths
  3. load_sequences — combined loader (raw + synthetic, with optional augmentation)
"""

from __future__ import annotations

import json
import logging
import os

import numpy as np

log = logging.getLogger(__name__)

RAW_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "raw")


# ---------------------------------------------------------------------------
# 1. Raw OHLCV loader
# ---------------------------------------------------------------------------

def load_raw_sequences(raw_dir: str = RAW_DIR) -> list[list[dict]]:
    """Load each JSON file from *raw_dir* and convert OHLCV to sequences.

    Each candle becomes ``{price, volume, vol_ratio, buy_sell_ratio}``.
    ``price`` = close, ``vol_ratio`` = volume / rolling-10-mean,
    ``buy_sell_ratio`` = 0.5 (neutral — not available in OHLCV data).

    Keeps first 120 candles (2 h of minute data) and skips files with < 10.
    """
    sequences: list[list[dict]] = []
    if not os.path.isdir(raw_dir):
        log.warning("Raw directory %s not found", raw_dir)
        return sequences

    for fname in os.listdir(raw_dir):
        if not fname.endswith(".json") or fname == "manifest.json":
            continue
        fpath = os.path.join(raw_dir, fname)
        try:
            with open(fpath) as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError):
            continue

        ohlcv = data.get("ohlcv", [])
        if len(ohlcv) < 10:
            continue

        # Sort by timestamp ascending (some files are descending)
        ohlcv = sorted(ohlcv, key=lambda c: c[0])

        # Keep first 120 candles
        ohlcv = ohlcv[:120]

        # Build volume history for rolling mean
        volumes = [float(c[5]) for c in ohlcv]

        seq: list[dict] = []
        for i, candle in enumerate(ohlcv):
            # OHLCV: [ts, open, high, low, close, volume]
            close_price = float(candle[4])
            volume = float(candle[5])

            # Rolling mean volume (last 10 candles, or fewer at start)
            lookback_start = max(0, i - 9)
            mean_vol = np.mean(volumes[lookback_start : i + 1])
            vol_ratio = volume / mean_vol if mean_vol > 0 else 1.0

            seq.append({
                "price": max(close_price, 1e-18),
                "volume": volume,
                "vol_ratio": float(vol_ratio),
                "buy_sell_ratio": 0.5,
            })

        if len(seq) >= 10:
            sequences.append(seq)

    log.info("Loaded %d raw sequences from %s", len(sequences), raw_dir)
    return sequences


# ---------------------------------------------------------------------------
# 2. Synthetic sequence generator
# ---------------------------------------------------------------------------

def generate_synthetic_sequences(n: int = 5000) -> list[list[dict]]:
    """Generate realistic synthetic pump/dump price paths.

    70% pump pattern: gradual rise -> sharp spike -> dump.
    30% flat/dead: slow decay with noise.
    """
    rng = np.random.default_rng(42)
    sequences: list[list[dict]] = []

    for _ in range(n):
        is_pump = rng.random() < 0.70
        length = int(rng.integers(30, 121))  # 30-120 minutes
        entry_price = float(rng.uniform(1e-6, 0.01))

        prices = [entry_price]
        volumes = [float(rng.uniform(500, 20000))]

        if is_pump:
            # Phase 1: gradual rise over ~15 min (10-30%)
            rise_end = min(15, length)
            rise_target = rng.uniform(1.10, 1.30)
            drift_rise = np.log(rise_target) / rise_end

            # Phase 2: sharp spike over ~5 min (2-10x)
            spike_end = min(rise_end + 5, length)
            spike_mult = rng.uniform(2.0, 10.0)
            drift_spike = np.log(spike_mult) / max(spike_end - rise_end, 1)

            # Phase 3: dump (-60% to -90%)
            dump_frac = rng.uniform(0.60, 0.90)

            for t in range(1, length):
                if t < rise_end:
                    drift = drift_rise
                    noise = 0.03
                    vol_mult = 1.0 + rng.uniform(0, 0.5)
                elif t < spike_end:
                    drift = drift_spike
                    noise = 0.06
                    vol_mult = 3.0 + rng.uniform(0, 5.0)
                else:
                    # Dump phase: exponential decay
                    remaining = length - spike_end
                    if remaining > 0:
                        drift = np.log(1 - dump_frac) / remaining
                    else:
                        drift = -0.05
                    noise = 0.08
                    vol_mult = 2.0 + rng.uniform(0, 3.0)

                ret = drift + noise * rng.standard_normal()
                prices.append(max(prices[-1] * np.exp(ret), 1e-18))
                volumes.append(float(volumes[0] * vol_mult * rng.uniform(0.5, 2.0)))

        else:
            # Flat/dead: slow decay -0.5% per minute with noise
            for t in range(1, length):
                ret = -0.005 + 0.02 * rng.standard_normal()
                prices.append(max(prices[-1] * np.exp(ret), 1e-18))
                volumes.append(float(volumes[0] * rng.uniform(0.3, 1.2)))

        # Build sequence dicts
        seq: list[dict] = []
        for i in range(len(prices)):
            lookback_start = max(0, i - 9)
            mean_vol = np.mean(volumes[lookback_start : i + 1])
            vol_ratio = volumes[i] / mean_vol if mean_vol > 0 else 1.0
            seq.append({
                "price": prices[i],
                "volume": volumes[i],
                "vol_ratio": float(vol_ratio),
                "buy_sell_ratio": 0.5,
            })
        sequences.append(seq)

    log.info("Generated %d synthetic sequences", len(sequences))
    return sequences


# ---------------------------------------------------------------------------
# 3. Combined loader
# ---------------------------------------------------------------------------

def load_sequences(
    raw_dir: str = RAW_DIR,
    augmenter=None,
    min_sequences: int = 500,
) -> list[list[dict]]:
    """Load raw + synthetic sequences, optionally augmented.

    If *augmenter* is provided and has a ``sample_exit_slippage`` method,
    each raw sequence is duplicated 3 times with varied slippage profiles
    (different vol_ratio noise applied to simulate market conditions).

    Returns a shuffled combined list.
    """
    raw = load_raw_sequences(raw_dir)
    synthetic = generate_synthetic_sequences(n=max(5000, min_sequences - len(raw)))

    all_seqs: list[list[dict]] = list(raw) + list(synthetic)

    # Augment raw sequences with slippage variants
    if augmenter is not None and len(raw) > 0:
        augmented: list[list[dict]] = []
        rng = np.random.default_rng(123)
        for seq in raw:
            for _ in range(3):
                variant = []
                for step in seq:
                    d = dict(step)
                    # Jitter vol_ratio to simulate different market conditions
                    d["vol_ratio"] = float(d["vol_ratio"] * rng.uniform(0.5, 2.0))
                    variant.append(d)
                augmented.append(variant)
        all_seqs.extend(augmented)
        log.info("Created %d augmented variants from %d raw sequences",
                 len(augmented), len(raw))

    # Shuffle
    rng = np.random.default_rng(0)
    rng.shuffle(all_seqs)

    log.info("Total sequences: %d (raw=%d, synthetic=%d, augmented=%d)",
             len(all_seqs), len(raw), len(synthetic),
             len(all_seqs) - len(raw) - len(synthetic))
    return all_seqs
