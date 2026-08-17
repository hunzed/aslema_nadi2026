#!/bin/bash -l

#SBATCH -J infer_ft
#SBATCH -o ${PROJ}/logs/inference_ft/infer_ft_%j.out
#SBATCH -e ${PROJ}/logs/inference_ft/infer_ft_%j.err
#SBATCH -p gpu-H200
#SBATCH -q ${SLURM_QOS}
#SBATCH --gres=gpu:1
#SBATCH -A ${SLURM_ACCOUNT}
# A100 alt (override at submit; confirm your A100 partition name):
#   sbatch -p gpu-A100 -A A100_Intern -q A100_Intern <this script>
#SBATCH -c 32
#SBATCH --mem=100GB

# ============================================================================
# Generic inference for MERGED fine-tuned checkpoints - the post-SFT sibling
# of the per-model zero-shot scripts. One script for every model because the
# checkpoint path is only known after training; everything is parameterized
# at submit time instead of hardcoded per file.
#
# Required --export vars:
#   MODEL       merged checkpoint dir (from merge_lora.sh, i.e. .../checkpoint-XXXX-merged)
#   MODEL_NAME  result-file slug, e.g. qwen2_5_omni_7b_sft  (keep the _sft
#               suffix so zero-shot rows are never overwritten and the eval
#               scripts score it as a separate run)
#   FAMILY      qwen | gemma
#   SUBTASK     intent | slot_filling
# Optional: SPLIT (dev), MAX_NEW_TOKENS (64 intent / 256 slot),
#           EXTRA_ARGS (space-separated extra swift flags, e.g. "--model_type gemma4")
#
# Example:
#   sbatch --export=ALL,MODEL=$PROJ/models/qwen/qwen2_5_omni_7b_intent_lora/checkpoint-1200-merged,MODEL_NAME=qwen2_5_omni_7b_sft,FAMILY=qwen,SUBTASK=intent \
#       scripts/inference/infer_finetuned.sh
#
# For the 30B merged model add:  --mem=200GB  on the sbatch line.
# ============================================================================
set -e
# $0 is a Slurm spool path under sbatch, not this file's real location.
PROJ_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
source "${PROJ}/scripts/common.sh"
source "${PROJ}/scripts/inference/_swift_infer.sh"

for v in MODEL MODEL_NAME FAMILY SUBTASK; do
    if [ -z "${!v:-}" ]; then
        echo "FATAL: $v not set. See the usage example in this script's header." >&2
        exit 1
    fi
done
if [ ! -d "$MODEL" ]; then
    echo "FATAL: MODEL must be a local merged-checkpoint dir (got '$MODEL')." >&2
    echo "Run scripts/training/merge_lora.sh first - vLLM cannot serve a raw" >&2
    echo "LoRA adapter for Qwen3-Omni-MoE, and merged-dir serving is the one" >&2
    echo "path used for all families here." >&2
    exit 1
fi

# subtask-appropriate generation budget (same values as the zero-shot scripts)
if [ -z "${MAX_NEW_TOKENS:-}" ]; then
    if [ "$SUBTASK" = "intent" ]; then MAX_NEW_TOKENS=64; else MAX_NEW_TOKENS=256; fi
fi
# family knobs, mirroring the per-model zero-shot scripts
EXTRA_SWIFT_ARGS=()
if [ "$FAMILY" = "qwen" ]; then
    PRE_ENV="ENABLE_AUDIO_OUTPUT=0"        # thinker-only (Qwen-Omni-specific)
elif [ "$FAMILY" = "gemma" ]; then
    # the vLLM gemma4_mm list-vs-tensor audio crash applies to the merged
    # model exactly as it did zero-shot - keep the encoder unbatched
    EXTRA_SWIFT_ARGS+=(--vllm_max_num_seqs 1)
fi
# arrays can't cross sbatch --export; accept a plain string and split it
if [ -n "${EXTRA_ARGS:-}" ]; then
    read -r -a _extra <<< "$EXTRA_ARGS"
    EXTRA_SWIFT_ARGS+=("${_extra[@]}")
fi

run_infer
