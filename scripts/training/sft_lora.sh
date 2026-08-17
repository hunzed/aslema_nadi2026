#!/bin/bash -l
#SBATCH -J aslema_sft
#SBATCH -o ${PROJ}/logs/training/sft_%j.out
#SBATCH -e ${PROJ}/logs/training/sft_%j.err
#SBATCH -p ${SLURM_PARTITION}
#SBATCH -q ${SLURM_QOS}
#SBATCH -A ${SLURM_ACCOUNT}
#SBATCH --gres=gpu:1
#SBATCH -c 32
#SBATCH --mem=100GB
# ============================================================================
# Generic LoRA SFT entry point (ms-swift). One script for every backbone.
#
# Required --export vars:
#   MODEL       HF repo id or local dir (e.g. Qwen/Qwen3-Omni-30B-A3B-Instruct)
#   MODEL_NAME  output slug (e.g. qwen3_omni_30b_a3b)
#   FAMILY      qwen | gemma      (selects the prompt/jsonl variant)
#   SUBTASK     intent | slot_filling | joint
# Optional: NUM_EPOCHS (2), LR (1e-4), LORA_RANK (16), LORA_ALPHA (32),
#           BATCH (4), GRAD_ACCUM (2), RESUME_FROM, AUG_TAG (synth|mix)
#
# Example (the submitted system's recipe):
#   sbatch --mem=200GB --export=ALL,MODEL=Qwen/Qwen3-Omni-30B-A3B-Instruct,\
# MODEL_NAME=qwen3_omni_30b_a3b,FAMILY=qwen,SUBTASK=joint,NUM_EPOCHS=2,AUG_TAG=mix \
#       training/sft_lora.sh
#
# AUG_TAG trains on augmentation/data/swift/<subtask>/<family>/<tag>_sft.jsonl
# (built by bin/augmentation/build_manifests.py) instead of the real train split.
# ============================================================================
set -e
: "${PROJ:?source config.sh first}"
source "${PROJ}/scripts/common.sh"
source "${PROJ}/scripts/training/_swift_sft.sh"

for v in MODEL MODEL_NAME FAMILY SUBTASK; do
    [ -z "${!v:-}" ] && { echo "FATAL: $v not set (see header)" >&2; exit 1; }
done
: "${NUM_EPOCHS:=2}"

# family knobs
if [ "$FAMILY" = "qwen" ]; then
    PRE_ENV="ENABLE_AUDIO_OUTPUT=0"                    # thinker-only (Qwen-Omni)
elif [ "$FAMILY" = "gemma" ]; then
    # bare q/k/v/o names also match the vision/audio towers' unsupported
    # clippable linears -> scope LoRA by path instead
    : "${TARGET_REGEX:=^(?!.*(vision_tower|audio_tower)).*\.(q_proj|k_proj|v_proj|o_proj)$}"
fi
case "$MODEL" in *Qwen3-Omni*|*qwen3_omni*) EXTRA_SWIFT_ARGS=(--experts_impl grouped_mm);; esac

case "$SUBTASK" in
    intent|slot_filling) run_sft ;;
    joint)               run_sft_joint ;;
    *) echo "FATAL: SUBTASK must be intent | slot_filling | joint" >&2; exit 1 ;;
esac
