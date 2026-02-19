#!/bin/bash
# Soldar — environment setup
# Usage: bash scripts/setup.sh [--gpu]
set -e

GPU=false
for arg in "$@"; do
  [[ "$arg" == "--gpu" ]] && GPU=true
done

CONDA_ENV="soldar"
echo "🛰 Setting up Soldar environment (env: $CONDA_ENV)"

# Detect conda
if ! command -v conda &>/dev/null; then
  # Try common locations
  for p in ~/miniconda3/etc/profile.d/conda.sh ~/anaconda3/etc/profile.d/conda.sh; do
    [[ -f "$p" ]] && source "$p" && break
  done
fi

conda --version || { echo "❌ conda not found. Install Miniconda first: https://docs.conda.io/en/latest/miniconda.html"; exit 1; }

# Create or update env
if conda env list | grep -q "^$CONDA_ENV "; then
  echo "✅ Conda env '$CONDA_ENV' already exists — updating"
  conda activate $CONDA_ENV
else
  echo "📦 Creating conda env '$CONDA_ENV' (Python 3.11)..."
  conda create -n $CONDA_ENV python=3.11 -y
  conda activate $CONDA_ENV
fi

# Core deps
pip install -q --upgrade pip

if $GPU; then
  echo "🖥  Installing GPU-optimized packages (CUDA 12.x)..."
  # PyTorch with CUDA
  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
  # XGBoost with GPU
  pip install xgboost[gpu]
  # LightGBM with GPU
  pip install lightgbm --install-option=--gpu 2>/dev/null || pip install lightgbm
else
  echo "💻 Installing CPU packages..."
  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
  pip install xgboost lightgbm
fi

# All other deps
pip install -q \
  requests aiohttp python-dotenv \
  pandas numpy scikit-learn \
  optuna shap pyarrow \
  imbalanced-learn \
  stable-baselines3[extra] gymnasium shimmy \
  streamlit streamlit-autorefresh plotly \
  tqdm solders \
  jupyter ipykernel

# Create .env from template if it doesn't exist
if [[ ! -f .env ]]; then
  cp .env.example .env 2>/dev/null || cat > .env << 'ENV'
# Soldar configuration
TELEGRAM_BOT_TOKEN=your_bot_token_here
TELEGRAM_CHAT_ID=your_chat_id_here

# Solana execution (for live trading)
WALLET_PRIVATE_KEY=your_base58_private_key_here
HELIUS_RPC_URL=https://mainnet.helius-rpc.com/?api-key=your_key_here

# Optional enrichment
BIRDEYE_API_KEY=your_birdeye_key_here
ENV
  echo "📝 Created .env template — fill in your credentials"
fi

# Create required dirs
mkdir -p ml/data ml/models ml/raw data

echo ""
echo "✅ Setup complete! Activate with: conda activate $CONDA_ENV"
echo ""
echo "Next steps:"
echo "  1. Edit .env with your credentials"
echo "  2. Download datasets: bash scripts/download_data.sh"
echo "  3. Train models:      bash scripts/train.sh"
echo "  4. Run screener:      python screener.py"
echo "  5. Run dashboard:     streamlit run dashboard.py --server.port 8502"
