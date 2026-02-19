#!/bin/bash
# Soldar — download and prepare training data
set -e

source ~/miniconda3/etc/profile.d/conda.sh 2>/dev/null || true
conda activate soldar 2>/dev/null || conda activate meme-screener

cd "$(dirname "$0")/.."
echo "🛰 Soldar — downloading training data"

mkdir -p ml/data ml/raw repos

# Clone research repos if not present
REPOS=(
  "https://github.com/git-disl/MemeTrans.git|repos/MemeTrans"
  "https://github.com/Bayi-Hu/Pump-and-Dump-Detection-on-Cryptocurrency.git|repos/bayi-pd"
  "https://github.com/Derposoft/crypto_pump_and_dump_with_deep_learning.git|repos/derposoft-dl"
  "https://github.com/SystemsLab-Sapienza/pump-and-dump-dataset.git|repos/sapienza-dataset"
  "https://github.com/Mehrnoom/Cryptocurrency-Pump-Dump.git|repos/mehrnoom-social"
)

for entry in "${REPOS[@]}"; do
  url="${entry%%|*}"
  dest="${entry##*|}"
  name=$(basename "$dest")
  if [[ -d "$dest/.git" ]]; then
    echo "✅ $name already cloned"
  else
    echo "📥 Cloning $name..."
    git clone "$url" "$dest" --depth 1 --quiet
    echo "✅ $name cloned"
  fi
done

# Build unified training dataset from research repos
echo ""
echo "🔧 Building training dataset..."
python ml/data_pipeline.py

echo ""
echo "✅ Data ready:"
ls -lh ml/data/*.parquet 2>/dev/null || echo "  Run: python ml/data_pipeline.py manually if needed"

# Optionally collect fresh historical data from GeckoTerminal
if [[ "$1" == "--collect-fresh" ]]; then
  echo ""
  echo "📡 Collecting fresh historical data from GeckoTerminal (this takes ~30min)..."
  python ml/collect.py --pages 200 --out ml/raw/ 2>&1 | tee ml/collect.log
fi
