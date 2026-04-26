#!/usr/bin/env bash
# Create and activate the project environment on a fresh server.
# Idempotent: re-running is safe.
#
# Usage:
#   bash setup_env.sh                # tries conda first, falls back to venv+pip
#   bash setup_env.sh --pip          # force venv+pip (skip conda)
#
# After this, activate with EITHER:
#   conda activate psypectrum
#   source .venv/bin/activate

set -euo pipefail

cd "$(dirname "$0")"
echo "[setup] working dir: $(pwd)"

USE_CONDA=1
if [[ "${1:-}" == "--pip" ]]; then
    USE_CONDA=0
fi

if [[ $USE_CONDA -eq 1 ]] && command -v conda >/dev/null 2>&1; then
    echo "[setup] using conda"
    if ! conda env list | grep -qE "^\s*psypectrum\s"; then
        echo "[setup] creating conda env 'psypectrum' from environment.yml"
        conda env create -f environment.yml
    else
        echo "[setup] env 'psypectrum' already exists"
        # shellcheck disable=SC1091
        source "$(conda info --base)/etc/profile.d/conda.sh"
        conda activate psypectrum
        pip install --upgrade -r requirements.txt
        conda deactivate
    fi
    echo
    echo "[setup] DONE. Activate with:"
    echo "  conda activate psypectrum"
else
    echo "[setup] using venv + pip"
    if [[ ! -d .venv ]]; then
        python3 -m venv .venv
    fi
    # shellcheck disable=SC1091
    source .venv/bin/activate
    pip install --upgrade pip
    pip install -r requirements.txt
    deactivate
    echo
    echo "[setup] DONE. Activate with:"
    echo "  source .venv/bin/activate"
fi

# Sanity check: NRC-VAD lexicon must exist for token-VAD runs.
if [[ ! -f NRC-VAD-Lexicon-v2.1.txt ]]; then
    echo
    echo "[WARNING] NRC-VAD-Lexicon-v2.1.txt is missing. Token-VAD runs will fail."
fi

# Sanity check: COLOR_MAP.txt
if [[ ! -f COLOR_MAP.txt ]]; then
    echo "[WARNING] COLOR_MAP.txt is missing. All non-baseline runs will fail."
fi

mkdir -p results logs checkpoints cache
echo "[setup] created results/ logs/ checkpoints/ cache/"
