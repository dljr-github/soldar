#!/bin/bash
# Soldar — train all models
# Usage:
#   bash scripts/train.sh           # train all
#   bash scripts/train.sh --gpu     # GPU-accelerated (5090 etc.)
#   bash scripts/train.sh xgb       # single model
#   bash scripts/train.sh lstm      # single model
#   bash scripts/train.sh rl        # single model
set -e

source ~/miniconda3/etc/profile.d/conda.sh 2>/dev/null || true
conda activate soldar 2>/dev/null || conda activate meme-screener

cd "$(dirname "$0")/.."

GPU=false
TARGET="all"
for arg in "$@"; do
  [[ "$arg" == "--gpu" ]] && GPU=true
  [[ "$arg" == "xgb" || "$arg" == "lstm" || "$arg" == "rl" ]] && TARGET="$arg"
done

if $GPU; then
  export CUDA_VISIBLE_DEVICES=0
  # ROCm iGPU: override to nearest compiled arch (gfx1035 → gfx1030)
  export HSA_OVERRIDE_GFX_VERSION=10.3.0
  export ROCR_VISIBLE_DEVICES=0
  echo "🖥  GPU mode enabled (ROCm iGPU)"
  python -c "import torch; print(f'PyTorch CUDA: {torch.cuda.is_available()} | Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"CPU\"}')"
else
  # Hide GPU by default — CPU is faster for our model sizes
  export CUDA_VISIBLE_DEVICES=""
  export HIP_VISIBLE_DEVICES=""
fi

echo "🛰 Soldar — training models (target: $TARGET)"
mkdir -p ml/models ml/models/checkpoints

# Check data exists
if [[ ! -f ml/data/train.parquet ]]; then
  echo "❌ Training data not found. Run: bash scripts/download_data.sh"
  exit 1
fi

train_xgb() {
  echo ""
  echo "━━━ XGBoost + LightGBM ━━━"
  echo "▶ Training XGBoost baseline..."
  python ml/train_xgboost.py \
    --train ml/data/train.parquet \
    --val   ml/data/val.parquet \
    --test  ml/data/test.parquet

  echo "▶ Tuning XGBoost (Optuna, 30 trials)..."
  python ml/train_xgboost.py \
    --train ml/data/train.parquet \
    --val   ml/data/val.parquet \
    --test  ml/data/test.parquet \
    --tune

  echo "▶ Training LightGBM..."
  python ml/train_lightgbm.py \
    --train ml/data/train.parquet \
    --val   ml/data/val.parquet \
    --test  ml/data/test.parquet
}

train_lstm() {
  echo ""
  echo "━━━ LSTM (Sapienza sequences) ━━━"
  EXTRA_ARGS=""
  $GPU && EXTRA_ARGS="--device cuda"
  python ml/train_lstm.py --data-source sapienza $EXTRA_ARGS
}

train_rl() {
  echo ""
  echo "━━━ RL Exit Agent (PPO) ━━━"
  TS=500000
  $GPU && TS=1000000  # more timesteps on GPU since it's faster
  echo "▶ Training PPO exit agent ($TS timesteps)..."
  python ml/rl_exit_agent.py --train --timesteps $TS

  echo "▶ Evaluating against baselines..."
  python ml/rl_exit_agent.py --eval
}

run_backtest() {
  echo ""
  echo "━━━ Backtest on test set ━━━"
  python ml/backtest.py --save-report
}

case $TARGET in
  xgb)  train_xgb ;;
  lstm) train_lstm ;;
  rl)   train_rl ;;
  all)
    train_xgb
    train_lstm
    train_rl
    run_backtest
    ;;
esac

echo ""
echo "✅ Training complete. Models saved to ml/models/"
echo ""
ls -lh ml/models/*.pkl ml/models/*.pt ml/models/*.zip 2>/dev/null || true
echo ""
echo "View results: streamlit run dashboard.py --server.port 8502"
