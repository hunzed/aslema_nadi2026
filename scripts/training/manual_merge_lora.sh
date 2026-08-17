#!/bin/bash -l

#SBATCH -J manual_merge
#SBATCH -o ${PROJ}/logs/training/manual_merge_%j.out
#SBATCH -e ${PROJ}/logs/training/manual_merge_%j.err
#SBATCH -p cpu-all
#SBATCH -c 16
#SBATCH --mem=200GB

# ============================================================================
# Manual LoRA merge, bypassing `swift export --merge_lora`/PeftModel entirely.
# `swift export --merge_lora true` (merge_lora.sh) crashes on
# qwen3_omni_30b_a3b specifically with:
#   TypeError: WeightConverter.__init__() got an unexpected keyword argument
#   'distributed_operation'
# (peft 0.19.1 / transformers>=5.8.0 API mismatch - see
# logs/setup/check_peft_tf_*.out for the confirmed root cause). This script
# does the identical arithmetic (W += (alpha/r)*B@A) directly on the base
# model's safetensors shards via scripts/training/manual_merge_lora.py - no
# PeftModel/transformers model class involved. CPU-only, pure tensor
# arithmetic, no GPU needed. Confirmed working 2026-07-23/28 on
# qwen3_omni_30b_a3b_{intent,slot_filling}_lora (288 LoRA deltas, 15 shards,
# ~4 min) and also qwen2.5-Omni-3B/7B (240/208 deltas, 3/5 shards).
#
# Use this ONLY for qwen3_omni_30b_a3b (MoE) checkpoints - merge_lora.sh is
# the default for every other family (gemma-4, qwen 3B/7B) and works fine
# there; this script also hard-requires a SHARDED base model
# (model.safetensors.index.json) and will refuse a single-shard
# model.safetensors base (e.g. gemma-4-E4B-it's local snapshot).
#
# Usage:
#   sbatch --export=ALL,CKPT=$PROJ/models/qwen/qwen3_omni_30b_a3b_intent_lora/checkpoint-XXXX \
#       scripts/training/manual_merge_lora.sh
# ============================================================================
set -e
export NO_MODULES=1   # no GPU/triton here; don't require cuda/gcc modules
PROJ_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
source "${PROJ}/scripts/common.sh"

if [ -z "${CKPT:-}" ]; then
    echo "FATAL: CKPT not set. Submit with:" >&2
    echo "  sbatch --export=ALL,CKPT=\$PROJ/models/<family>/<run>_lora/checkpoint-XXXX scripts/training/manual_merge_lora.sh" >&2
    exit 1
fi
if [ ! -f "$CKPT/adapter_config.json" ]; then
    echo "FATAL: $CKPT has no adapter_config.json - not a LoRA adapter dir?" >&2
    exit 1
fi

MERGED="${CKPT}-merged"
if ls "$MERGED"/*.safetensors >/dev/null 2>&1; then
    echo "Merged checkpoint already exists at $MERGED - skipping."
    exit 0
fi

BASE_MODEL="$(python -c "import json; print(json.load(open('$CKPT/adapter_config.json'))['base_model_name_or_path'])")"
echo "base model (from adapter_config.json): $BASE_MODEL"
if [ -d "$BASE_MODEL" ]; then
    RESOLVED_BASE="$BASE_MODEL"
else
    # base_model_name_or_path is usually already a resolved local HF-cache
    # snapshot dir (what ms-swift records), but fall back to resolving a real
    # repo id from the local cache only - never hit the network from cpu-all.
    RESOLVED_BASE="$(python -c "
from huggingface_hub import snapshot_download
print(snapshot_download('$BASE_MODEL', local_files_only=True))
")"
fi
echo "resolved local base model dir: $RESOLVED_BASE"
if [ ! -f "$RESOLVED_BASE/model.safetensors.index.json" ]; then
    echo "FATAL: $RESOLVED_BASE has no model.safetensors.index.json - single-shard base model?" >&2
    echo "This script requires a sharded base (see header note); it is not a safe fallback for those." >&2
    exit 1
fi

echo "Node: $(hostname)"; date
echo "== manually merging $CKPT -> $MERGED =="

python "${PROJ}/scripts/training/manual_merge_lora.py" \
    --base-dir "$RESOLVED_BASE" \
    --adapter-dir "$CKPT" \
    --output-dir "$MERGED"

if ! ls "$MERGED"/*.safetensors >/dev/null 2>&1; then
    echo "FATAL: no *.safetensors in $MERGED after merge." >&2
    exit 1
fi
date
echo "Done. Serve it with:"
echo "  sbatch --export=ALL,MODEL=$MERGED,MODEL_NAME=<slug>_sft,FAMILY=qwen,SUBTASK=<intent|slot_filling> scripts/inference/infer_finetuned.sh"
