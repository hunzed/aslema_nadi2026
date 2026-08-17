#!/bin/bash -l

#SBATCH -J merge_lora
#SBATCH -o ${PROJ}/logs/training/merge_%j.out
#SBATCH -e ${PROJ}/logs/training/merge_%j.err
#SBATCH -p cpu-all
#SBATCH -c 16
#SBATCH --mem=200GB

# ============================================================================
# Merge a trained LoRA adapter into the base weights (W += (alpha/r)*B@A) and
# write a standalone merged checkpoint next to it: <CKPT>-merged/
#
# Why merge at all: vLLM REFUSES runtime LoRA on Qwen3-Omni-MoE ("does not
# support LoRA yet" - seen in the sibling reference project), and serving a
# merged checkpoint is the one path that works identically for every family
# here (qwen 7B, gemma-4, qwen3 30B MoE). So post-SFT inference always goes:
#   merge (this script, cpu)  ->  infer_finetuned.sh (GPU, merged dir as MODEL)
#
# CPU-only: merging is pure tensor arithmetic, no forward pass. 200GB RAM
# covers the 30B (~70GB bf16 weights held in memory + headroom); if your
# ms-swift version insists on a GPU for export, resubmit on the GPU partition:
#   sbatch -p gpu-H200 -A ${SLURM_ACCOUNT} -q ${SLURM_QOS} --gres=gpu:1 --export=ALL,CKPT=... scripts/training/merge_lora.sh
#
# Usage:
#   sbatch --export=ALL,CKPT=$PROJ/models/qwen/qwen2_5_omni_7b_intent_lora/checkpoint-XXXX \
#       scripts/training/merge_lora.sh
# ============================================================================
set -e
export NO_MODULES=1   # no GPU/triton here; don't require cuda/gcc modules
PROJ_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
source "${PROJ}/scripts/common.sh"

if [ -z "${CKPT:-}" ]; then
    echo "FATAL: CKPT not set. Submit with:" >&2
    echo "  sbatch --export=ALL,CKPT=\$PROJ/models/<family>/<run>_lora/checkpoint-XXXX scripts/training/merge_lora.sh" >&2
    exit 1
fi
if [ ! -d "$CKPT" ]; then
    echo "FATAL: checkpoint dir not found: $CKPT" >&2
    ls -d "$(dirname "$CKPT")"/checkpoint-* 2>/dev/null || true
    exit 1
fi

MERGED="${CKPT}-merged"
if ls "$MERGED"/*.safetensors >/dev/null 2>&1; then
    echo "Merged checkpoint already exists at $MERGED - skipping."
    exit 0
fi

echo "Node: $(hostname)"; date
echo "== merging $CKPT -> $MERGED =="

swift export \
    --adapters "$CKPT" \
    --merge_lora true \
    --use_hf true \
    --output_dir "$MERGED"

# sanity: a merged model must contain real weight shards
if ! ls "$MERGED"/*.safetensors >/dev/null 2>&1; then
    echo "FATAL: no *.safetensors in $MERGED after export - merge failed." >&2
    exit 1
fi
date
echo "Done. Serve it with:"
echo "  sbatch --export=ALL,MODEL=$MERGED,MODEL_NAME=<slug>_sft,FAMILY=<qwen|gemma>,SUBTASK=<intent|slot_filling> scripts/inference/infer_finetuned.sh"
