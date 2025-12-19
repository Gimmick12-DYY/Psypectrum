#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=23:59:59
#SBATCH --partition=gpu            # TODO: set to the correct Longleaf GPU partition
#SBATCH --gres=gpu:1               # Adjust GPU count/type as needed
#SBATCH --output=/work/users/d/y/dyy12/PsyPec/logs/%x-%j.out
#SBATCH --error=/work/users/d/y/dyy12/PsyPec/logs/%x-%j.err
##SBATCH --account=YOUR_ALLOCATION  # TODO: set if required on Longleaf
##SBATCH --mail-type=END,FAIL

set -euo pipefail

WORKDIR="/work/users/d/y/dyy12/PsyPec"
cd "$WORKDIR"
mkdir -p "$WORKDIR/logs"

module load python/3.12.4
# Load CUDA/toolkit module if needed for the GPU partition.
# module load cuda/11.7
# source activate your_env

# Hugging Face caches in workdir to keep artifacts together.
export HF_HOME="$WORKDIR/hf_cache"
export TRANSFORMERS_CACHE="$HF_HOME/transformers"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
mkdir -p "$TRANSFORMERS_CACHE" "$HF_DATASETS_CACHE"

# ---- Training hyperparameters ----
BASE_MODEL="meta-llama/Llama-2-7b-hf"   # TODO: set the actual base model available to you
BATCH_SIZE=2
LR=5e-5
MAX_LEN=128
EPOCHS=3
COLOR_LOSS=0.1
LOG_PATH="$WORKDIR/logs/palette_colors_${SLURM_JOB_ID}.jsonl"

python -u - <<PY
import torch
from palette_model import train_goemotions_lora

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

train_goemotions_lora(
    base_model_name="${BASE_MODEL}",
    batch_size=${BATCH_SIZE},
    lr=${LR},
    max_length=${MAX_LEN},
    epochs=${EPOCHS},
    device=device,
    color_loss_weight=${COLOR_LOSS},
    log_jsonl_path="${LOG_PATH}",
    log_every=50,
)
PY
