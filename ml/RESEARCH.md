# Research & Reference Datasets

External datasets and papers cloned for ML model development.

## Datasets

### MemeTrans/
**Paper:** "MemeTrans: A Dataset for Detecting High-Risk Memecoin Launches on Solana" (arXiv 2602.13480, Feb 2026)
**Size:** 41,470+ memecoins, 200M+ transactions
**Features:** 122 features across 4 groups (context, trading activity, holder concentration, time-series)
**Use:** Primary training dataset — Solana-native, most directly relevant
**Note:** Raw data (>100GB) not yet released; processed features + model code available

### bayi-pd/
**Paper:** "Sequence-Based Target Coin Prediction for Cryptocurrency Pump-and-Dump" (SIGMOD 2023)
**Repo:** github.com/Bayi-Hu/Pump-and-Dump-Detection-on-Cryptocurrency
**Use:** LSTM architecture reference for sequence modeling; dataset for CEX pump-dump events
**Key idea:** Predicts WHICH coins will be pumped before they are — closest to our use case

### derposoft-dl/
**Paper:** "Crypto Pump and Dump Detection via Deep Learning Techniques"
**Repo:** github.com/Derposoft/crypto_pump_and_dump_with_deep_learning
**Models:** CLSTM, AnomalyTransformer, TransformerTimeSeries
**Use:** Architecture comparison — multiple DL approaches to benchmark against

### sapienza-dataset/
**Paper:** "Pump and Dumps in the Bitcoin Era" (ICCCN 2020) — La Morgia, Mei et al.
**Repo:** github.com/SystemsLab-Sapienza/pump-and-dump-dataset
**Use:** Labeled features for CEX P&D events; feature engineering reference
**Note:** Same research group as MemeChain paper

### mehrnoom-social/
**Paper:** Social signals for P&D detection
**Repo:** github.com/Mehrnoom/Cryptocurrency-Pump-Dump
**Data:** Twitter sentiment timeseries + Telegram pump signal labels
**Use:** Social signal features if/when we add Twitter/Telegram monitoring

## ML Development Roadmap

1. **Phase 1 — Baseline** (now): XGBoost on MemeTrans features, train/val/test split
2. **Phase 2 — Sequence model**: LSTM encoder on minute-level OHLCV (bayi-pd architecture adapted for Solana/DEX data)
3. **Phase 3 — RL exit agent**: PPO via Stable-Baselines3, trained on historical replays with slippage augmentation (see ml/augment.py)
4. **Phase 4 — Social signals**: Add Twitter/Telegram features from mehrnoom-social approach
5. **Live trading**: Paper trading → validate → flip to live with small capital
