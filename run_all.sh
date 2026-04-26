#!/usr/bin/env bash
# Run all 10 experiments sequentially. Tee'd to logs/run_all.log so you can tail it.
#
# Usage:
#   bash run_all.sh                  # all 10 in order
#   bash run_all.sh --smoke          # 1-batch smoke runs of every config
#   bash run_all.sh --only 04 07     # subset
#   bash run_all.sh --force          # re-run even if summary.json exists
#
# Activate your env BEFORE running:
#   conda activate psypectrum     OR     source .venv/bin/activate

set -euo pipefail
cd "$(dirname "$0")"

mkdir -p logs results checkpoints cache
ts=$(date +%Y%m%d_%H%M%S)
master_log="logs/run_all_${ts}.log"
echo "[run_all] master log: $master_log"
echo "[run_all] starting at $(date)"
echo "[run_all] python: $(which python)"
python -c "import torch; print('[run_all] torch', torch.__version__, 'cuda', torch.cuda.is_available())"

python -u run_experiments.py "$@" 2>&1 | tee "$master_log"
echo "[run_all] finished at $(date)"
