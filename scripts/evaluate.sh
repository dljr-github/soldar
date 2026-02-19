#!/bin/bash
# Soldar — evaluate models on test set and generate reports
set -e

source ~/miniconda3/etc/profile.d/conda.sh 2>/dev/null || true
conda activate soldar 2>/dev/null || conda activate meme-screener

cd "$(dirname "$0")/.."
echo "🛰 Soldar — evaluation suite"

# Backtest all strategies on test set
echo ""
echo "━━━ Backtesting on held-out test set ━━━"
python ml/backtest.py --save-report

# RL agent eval vs baselines
if [[ -f ml/models/ppo_exit_agent.zip ]]; then
  echo ""
  echo "━━━ RL Agent vs Baselines ━━━"
  python ml/rl_exit_agent.py --eval
  echo ""
  echo "━━━ RL Demo Episodes ━━━"
  python ml/rl_exit_agent.py --demo --n 3
fi

# Print model metrics summary
echo ""
echo "━━━ Model Metrics Summary ━━━"
python -c "
import json, os, glob

model_dir = 'ml/models'
for f in sorted(glob.glob(f'{model_dir}/*_metrics.json')):
    name = os.path.basename(f).replace('_metrics.json','')
    with open(f) as fp:
        m = json.load(fp)
    clf = m.get('classification', m)
    auc = clf.get('roc_auc', 'N/A')
    prec = clf.get('precision', 'N/A')
    rec = clf.get('recall', 'N/A')
    f1 = clf.get('f1', 'N/A')
    print(f'{name:30s} AUC={auc:.4f}  Prec={prec:.4f}  Rec={rec:.4f}  F1={f1:.4f}' if isinstance(auc, float) else f'{name}: {m}')
"

echo ""
echo "✅ Full backtest report: ml/models/backtest_report.html"
echo "   Open in browser: open ml/models/backtest_report.html"
