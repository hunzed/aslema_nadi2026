#!/bin/bash -l
#SBATCH -J aslema_setup
#SBATCH -o ${PROJ}/logs/setup/setup_%j.out
#SBATCH -e ${PROJ}/logs/setup/setup_%j.err
#SBATCH -p cpu
#SBATCH -c 16
#SBATCH --mem=64GB
# ============================================================================
# Build the MAIN python environment (ms-swift + vLLM + evaluation deps).
#
#   source config.sh && bash scripts/setup/setup_env.sh      # or sbatch it
#
# This project needs TWO separate environments, because VoxCPM pulls a torch
# version that conflicts with the ms-swift/vLLM pins:
#
#   1. $ENV_NAME (conda)  - data prep, training, inference, evaluation,
#                           and the LLM stages of the augmentation pipeline.
#                           Built here.
#   2. a venv under data/tts_venv - VoxCPM speech synthesis only.
#                           Built by scripts/augmentation/tts/setup_voxcpm.sh
#
# Nothing else in the repo mixes them: every script states which one it needs.
# ============================================================================
set -e
: "${PROJ:?run 'source config.sh' first}"
: "${ENV_NAME:=aslema}"
: "${CONDA_SH:=$HOME/miniconda3/etc/profile.d/conda.sh}"

# keep conda's caches off the home quota on shared clusters
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$PROJ/tmp/xdg_cache}"
export CONDA_PKGS_DIRS="${CONDA_PKGS_DIRS:-$PROJ/tmp/conda_pkgs}"
mkdir -p "$XDG_CACHE_HOME" "$CONDA_PKGS_DIRS" "$PROJ/logs/setup"

[ -f "$CONDA_SH" ] || { echo "FATAL: no conda.sh at '$CONDA_SH' - set CONDA_SH in config.sh" >&2; exit 1; }
# shellcheck disable=SC1090
source "$CONDA_SH"

if ! conda env list | grep -qE "^${ENV_NAME}\s"; then
    # conda-forge + --override-channels avoids the defaults-channel ToS gate on
    # locked-down clusters; only python comes from conda, the rest from pip.
    conda create -n "$ENV_NAME" python=3.11 -y -c conda-forge --override-channels
fi
conda activate "$ENV_NAME"
[ "$CONDA_DEFAULT_ENV" = "$ENV_NAME" ] || { echo "FATAL: activate failed" >&2; exit 1; }

pip install --upgrade pip
pip install -r "$PROJ/requirements.txt"

python - <<'PY'
mods = ["torch", "transformers", "vllm", "swift", "librosa", "soundfile",
        "jiwer", "datasets", "pandas"]
import importlib
for m in mods:
    try:
        importlib.import_module(m); print(f"  OK      {m}")
    except Exception as e:
        print(f"  FAILED  {m}: {type(e).__name__}")
PY
echo "environment '$ENV_NAME' ready"
