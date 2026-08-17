#!/bin/bash -l
# ============================================================================
# Shared runtime setup, sourced by every job script in this repo.
# (#SBATCH headers cannot be sourced, so they stay in each job script.)
#
# Requires PROJ + ENV_NAME + CONDA_SH, all set by config.sh:
#     source config.sh && sbatch training/sft_lora.sh ...
# On a machine without Slurm, source this and run the swift/python command
# inside any job script directly.
# ============================================================================
: "${PROJ:?PROJ is not set - run 'source config.sh' first}"

# ---- caches (kept off the home quota) --------------------------------------
export HF_HOME="${HF_HOME:-$PROJ/hf_cache}"
export MODELSCOPE_CACHE="${MODELSCOPE_CACHE:-$HF_HOME/modelscope}"
export USE_HF="${USE_HF:-1}"          # ms-swift: resolve models from HF, not ModelScope
export TMPDIR="${TMPDIR:-$PROJ/tmp/job_${SLURM_JOB_ID:-manual}}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-$PROJ/tmp/triton_cache}"
mkdir -p "$TMPDIR" "$TRITON_CACHE_DIR" "$HF_HOME"

# ---- optional environment modules ------------------------------------------
# Only relevant on module-based HPC systems. Set NO_MODULES=1 to skip, and
# adjust the module names to your site. vLLM/triton JIT-compile kernels at
# runtime, so a working host gcc must be on PATH.
if [ -z "${NO_MODULES:-}" ] && command -v module >/dev/null 2>&1; then
    module purge 2>/dev/null || true
    module load ${CUDA_MODULE:-cuda12.6/toolkit} ${GCC_MODULE:-gcc11} 2>/dev/null \
        || echo "NOTE: could not load cuda/gcc modules - continuing with system defaults"
fi

# ---- conda ------------------------------------------------------------------
ENV_NAME="${ENV_NAME:-aslema}"
CONDA_SH="${CONDA_SH:-$HOME/miniconda3/etc/profile.d/conda.sh}"
if [ ! -f "$CONDA_SH" ]; then
    echo "FATAL: conda.sh not found at '$CONDA_SH'. Set CONDA_SH in config.sh." >&2
    exit 1
fi
# shellcheck disable=SC1090
source "$CONDA_SH"
conda activate "$ENV_NAME"
if [ "$CONDA_DEFAULT_ENV" != "$ENV_NAME" ]; then
    echo "FATAL: 'conda activate $ENV_NAME' did not take effect (got '$CONDA_DEFAULT_ENV')." >&2
    echo "Aborting before running against the wrong Python." >&2
    exit 1
fi

# ---- sanity: triton needs a working host compiler ---------------------------
gcc_sanity () {
    printf 'int main(){return 0;}' > "$TMPDIR/_cc.c" 2>/dev/null || return 0
    gcc -o "$TMPDIR/_cc.out" "$TMPDIR/_cc.c" 2>/dev/null \
        && echo "host gcc OK" || echo "WARNING: host gcc cannot compile - triton JIT may fail"
    rm -f "$TMPDIR/_cc.c" "$TMPDIR/_cc.out"
}
