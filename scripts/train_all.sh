#!/bin/bash
set -e
source ~/miniconda3/etc/profile.d/conda.sh
conda activate meme-screener
cd ~/openclaw/projects/soldar

echo "=== Training XGBoost ==="
python ml/train_xgboost.py --train ml/data/train.parquet --val ml/data/val.parquet --test ml/data/test.parquet

echo "=== Training LSTM ==="
python ml/train_lstm.py --data-source sapienza

echo "=== Training RL Exit Agent ==="
python ml/rl_exit_agent.py --train

echo "=== All models trained ==="
