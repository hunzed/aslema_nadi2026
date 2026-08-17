#!/bin/bash
# ============================================================================
# Single place to configure the whole repo. `source config.sh` before anything.
# ============================================================================
# Root of this checkout; every path below is derived from it.
export PROJ="${PROJ:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"

# --- compute -----------------------------------------------------------------
# Slurm account/QOS/partition. The scripts carry #SBATCH headers that read these;
# on a non-Slurm machine just run the underlying python/swift commands directly.
export SLURM_ACCOUNT="${SLURM_ACCOUNT:-your_account}"
export SLURM_QOS="${SLURM_QOS:-your_qos}"
export SLURM_PARTITION="${SLURM_PARTITION:-gpu}"

# --- python env --------------------------------------------------------------
# Conda env holding ms-swift + vLLM (see requirements.txt).
export ENV_NAME="${ENV_NAME:-aslema}"
export CONDA_SH="${CONDA_SH:-$HOME/miniconda3/etc/profile.d/conda.sh}"

# --- caches (keep off your home quota) ---------------------------------------
export HF_HOME="${HF_HOME:-$PROJ/hf_cache}"
export TMPDIR="${TMPDIR:-$PROJ/tmp}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-$PROJ/tmp/triton}"
mkdir -p "$HF_HOME" "$TMPDIR" "$TRITON_CACHE_DIR"

# --- LLM API (augmentation only) ---------------------------------------------
# Needed only by augmentation/: generation, judging, reference-ASR scoring.
# export GOOGLE_API_KEY=...
export GEN_MODEL="${GEN_MODEL:-gemini-3.6-flash}"          # bulk generation
export RARE_GEN_MODEL="${RARE_GEN_MODEL:-gemini-3.1-pro-preview}"  # rare intents
export JUDGE_MODELS="${JUDGE_MODELS:-gemini-3.1-pro-preview,gemini-3.6-flash,gemini-2.5-pro}"
export ASR_MODEL="${ASR_MODEL:-gemini-3.1-pro-preview}"    # reference selection
