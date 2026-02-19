# ML Data Collector & Training Pipeline

Collect historical Solana meme coin launch data, extract features, and train a classifier to predict which launches will pump.

## Data Source

All data comes from the [GeckoTerminal API](https://www.geckoterminal.com/) (free tier, ~30 req/min):
- **New pools endpoint** – lists recently created Solana DEX pools
- **OHLCV endpoint** – minute-level price candles per pool
- **Pool details endpoint** – liquidity, market cap, transaction counts

## Quick Start

```bash
# 1. Collect raw data (5 pages ≈ 50 pools, takes ~5 min)
python ml/collect.py --pages 5

# 2. Build ML-ready dataset
python ml/build_dataset.py

# 3. Train baseline model
python ml/train.py
```

## Full Collection

```bash
# 50 pages ≈ 500 pools, takes ~1 hour
python ml/collect.py --pages 50 --delay 2.0
python ml/build_dataset.py
python ml/train.py
```

## Scripts

| Script | Purpose |
|---|---|
| `ml/collect.py` | Fetch pool listings + OHLCV + details from GeckoTerminal |
| `ml/features.py` | Extract per-pool features at multiple detection points |
| `ml/build_dataset.py` | Combine all raw data into `dataset.parquet` / `dataset.csv` |
| `ml/train.py` | Train XGBoost classifier (target: `pumped_2x` at T=5min) |

## Features

Extracted at detection points T = 1, 3, 5, 10, 15 minutes after listing:

| Feature | Description |
|---|---|
| `price_change_T_pct` | % price change from open to minute T |
| `vol_per_min_T` | Average USD volume per minute in first T minutes |
| `price_velocity` | Rate of change over last 2 candles before T |
| `high_low_range_pct` | (high - low) / low across first T candles |
| `candles_up_pct` | Fraction of candles that closed above their open |
| `initial_liquidity_usd` | Pool reserve in USD at listing |
| `mcap_at_listing` | Fully diluted valuation at listing |
| `liq_to_mcap_ratio` | Liquidity / market cap |

## Labels

| Label | Definition |
|---|---|
| `pumped_2x` | Max gain within 1h >= 100% |
| `pumped_5x` | Max gain within 1h >= 400% |
| `pumped_10x` | Max gain within 2h >= 900% |
| `max_gain_1h_pct` | Highest % gain in first 60 minutes |
| `max_gain_2h_pct` | Highest % gain in first 120 minutes |
| `time_to_peak_min` | Minutes from listing to peak price (within 2h) |
| `survived_30min` | Pool still has candle data at minute 30 |

## Retraining

1. Collect more data: `python ml/collect.py --pages 100`
2. Rebuild dataset: `python ml/build_dataset.py`
3. Retrain: `python ml/train.py`

The collector skips already-downloaded pools, so repeated runs are incremental.
