#!/bin/bash
#SBATCH --job-name=PsyPec
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=23:59:59
#SBATCH --partition=a100-gpu
#SBATCH --gres=gpu:1
#SBATCH --output=/work/users/d/y/dyy12/PsyPec/logs/%x-%j.out
#SBATCH --error=/work/users/d/y/dyy12/PsyPec/logs/%x-%j.err

set -euo pipefail

WORKDIR="/work/users/d/y/dyy12/PsyPec"
cd "$WORKDIR"
mkdir -p "$WORKDIR/logs"

# --- Conda env from environment.yml (name: psypectrum) ---
# One-time on the cluster (from this directory or path to the file):
#   conda env create -f "$WORKDIR/environment.yml"
# If conda is not on PATH, set CONDA_ROOT to your base prefix, e.g.:
#   export CONDA_ROOT="$HOME/miniconda3"
# Some sites use: module load conda   # (uncomment if Longleaf/docs say so)

if command -v conda >/dev/null 2>&1; then
  # shellcheck source=/dev/null
  source "$(conda info --base)/etc/profile.d/conda.sh"
else
  if [[ -z "${CONDA_ROOT:-}" ]]; then
    for _base in "$HOME/miniconda3" "$HOME/anaconda3" "$HOME/mambaforge" "$HOME/miniforge3"; do
      if [[ -f "$_base/etc/profile.d/conda.sh" ]]; then
        CONDA_ROOT="$_base"
        break
      fi
    done
  fi
  if [[ -z "${CONDA_ROOT:-}" ]] || [[ ! -f "$CONDA_ROOT/etc/profile.d/conda.sh" ]]; then
    echo "ERROR: conda not found. Install Miniconda/Mambaforge, or: export CONDA_ROOT=/path/to/base" >&2
    exit 1
  fi
  # shellcheck source=/dev/null
  source "$CONDA_ROOT/etc/profile.d/conda.sh"
fi

conda activate psypectrum
echo "Using Python: $(command -v python)"
python --version

# Do not load module python/3.12.4 here; the conda env supplies Python and avoids PATH clashes.

export HF_HOME="$WORKDIR/hf_cache"
export TRANSFORMERS_CACHE="$HF_HOME/transformers"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
mkdir -p "$TRANSFORMERS_CACHE" "$HF_DATASETS_CACHE"

# Emotion–color GNN–BERT (production_plan.md): BERT warm-up, then joint GNN + color; evaluate vs BERT-only.
BERT_MODEL="bert-base-uncased"
BATCH_WARMUP=16
BATCH_JOINT=16
LR_WARMUP=2e-5
LR_JOINT=1e-3
MAX_LEN=128
EPOCHS_WARMUP=3
EPOCHS_JOINT=3
LOG_PATH="$WORKDIR/logs/palette_gnn_${SLURM_JOB_ID}.jsonl"

python -u - <<PY
import json
import torch
from palette_model import run_full_pipeline

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

_, metrics = run_full_pipeline(
    bert_name="${BERT_MODEL}",
    batch_size_warmup=${BATCH_WARMUP},
    batch_size_joint=${BATCH_JOINT},
    lr_warmup=${LR_WARMUP},
    lr_joint=${LR_JOINT},
    max_length=${MAX_LEN},
    epochs_warmup=${EPOCHS_WARMUP},
    epochs_joint=${EPOCHS_JOINT},
    device=device,
    log_jsonl_path="${LOG_PATH}",
)
print("Done. Metrics:", json.dumps(metrics, indent=2))
PY
