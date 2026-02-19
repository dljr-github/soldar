# MemeTrans Dataset Notes

Paper: "MemeTrans: A Dataset for Detecting High-Risk Memecoin Launches on Solana"
Repo: https://github.com/git-disl/MemeTrans

## Dataset Overview

Main file: `data/feat_label.csv` (~78MB)
Raw data: `raw_data/memecoin.jsonl` + `memecoin_metadata.jsonl` + `sol_hourly.txt`
Full raw dataset (>100GB) not yet released.

## Label

- `label`: "high" (high-risk / rug) vs "low" (low-risk / legitimate)
- `min_ratio`: minimum price ratio observed (drawdown measure)
- `return_ratio`: overall return ratio
- `pred_proba`: model prediction probability

## Feature Groups

### Group 1 — Launch Metadata (6 features)
- `group1_price`: price at migration
- `group1_migrate_{year,month,day,hour,weekday}`: launch timing

### Group 2 — Early Holder Distribution (58 features)
- **Concentration**: `holder_gini`, `top{1,5,10,20,50,100}_pct`, `top10_to_top100`
- **Snipers**: `sniper_{0s,1s,5s,10s}_hold_pct`, `sniper_{0s,1s,5s,10s}_num/ratio`
- **Early holders**: `early_top{1,5,10,20}_hold_pct`, `_initial_hold_pct`, `_hold_ratio`
- **Dev wallet**: `dev_hold_pct`, `dev_initial_hold_pct`, `dev_hold_ratio`
- **PnL signals**: realized/unrealized pnl_mean for early_top5/10/20, snipers, top5/10/20/100, all

### Group 3 — Transaction Activity (18 features)
- `tx_num`, `tx_num_valid`, `trader_tup_num`, `time_span`, `time_span_valid`
- `tx_per_sec`, `tx_per_sec_valid`
- `wash_tx_num`, `wash_ratio` (wash trading detection)
- `transfer_tx_num`, `transfer_ratio`
- `trader_num`, `holder_num`
- `buy_num`, `sell_num`, `buy_user_num`, `sell_user_num`
- `buy_vol`, `sell_vol`, `avg_buy_vol`, `avg_sell_vol`, `sell_pressure`

### Group 4 — Post-Launch Holder Evolution (28 features)
- Same concentration metrics as Group 2 but measured later
- `cluster_num`, `cluster_holder_num/ratio`, `cluster_total/avg/max_pct` (coordinated wallets)
- Delta features: `top{1,5,10,20,50,100}_pct_delta` (change in holder concentration)

## Models Used

- Random Forest, XGBoost, LightGBM, Gradient Boosting, Logistic Regression, MLP
- Train/test split: 70/30 chronological
- StandardScaler normalization
- Filtering: `time_span_valid >= 60s` and `holder_num >= 100`

## Pipeline Components

- `data_pipeline/feat_gen.py`: Feature generation from raw transactions (UnionFind for wallet clustering, Gini coefficient)
- `data_pipeline/bundle_fundflow.py`, `bundle_jito.py`: Bundle/MEV analysis (stubs)
- `data_pipeline/tranx_parser.py`: Transaction parser (stub)
- `risk_prediction/ml_model_train.py`: Train + evaluate classifiers
- `risk_prediction/memecoin_selection.py`: Downstream token selection application

## Relevance to Our Screener

Key features we could integrate:
1. **Sniper detection** (group2 sniper features) — we don't track this yet
2. **Wash trading ratio** — would improve our legitimacy scoring
3. **Holder concentration deltas** (group4 vs group2) — early vs later holder shifts
4. **Wallet clustering** — detect coordinated buys from related wallets
5. **Dev wallet behavior** — dev hold ratio as a rug signal
6. **Sell pressure metric** — direct buy/sell volume ratio

The label definition (high-risk = rug within observation window) aligns with our
use case. The 100+ features could serve as a feature template for RL state space.

## Additional: Time Series Data (from feat_gen.py)

The feature generator also produces 10-second OHLCV candlestick bars (up to 360 bars
= 1 hour) stored in `feat_with_ts.pkl`. This time series data is ideal for:
- RL episode formatting with our SlippageAugmenter
- Price trajectory modeling for entry/exit timing
- Training sequence models (LSTM, Transformer) on intraday meme coin dynamics
