#!/bin/bash -l
# ============================================================================
# Shared vLLM inference driver for all zero-shot runs. Sourced by the per-model
# scripts under qwen/ and gemma/. They set a handful of variables and call
# run_infer; everything common (path layout, resume-skip, the swift infer call)
# lives here so the qwen/gemma scripts stay a dozen readable lines each.
#
# Variables the caller sets before `run_infer`:
#   MODEL           HF repo id (e.g. Qwen/Qwen2.5-Omni-3B, google/gemma-4-E4B-it)
#   MODEL_NAME      short slug for output filenames (e.g. qwen2_5_omni_3b)
#   FAMILY          qwen | gemma  (selects the data/swift/<sub>/<FAMILY>/ inputs)
#   SUBTASK         intent | slot_filling
#   SPLIT           dev | test          (default: dev)
#   MAX_NEW_TOKENS  default 64 (intent) / 256 (slot) - caller sets
#   MAX_MODEL_LEN   default 4096
#   MAX_BATCH       default 16
#   MM_LIMIT        vllm_limit_mm_per_prompt JSON (default '{"audio": 1}')
#   PRE_ENV         extra env assignments prefixed to swift (e.g. ENABLE_AUDIO_OUTPUT=0)
#   EXTRA_SWIFT_ARGS  bash array of extra flags (e.g. --model_type ...)
#   INFER_BACKEND   vllm (default) | pt
#     gemma-4 caveat: vLLM's weight loader (default_loader.py:
#     track_weights_loading) does a strict "every param has a matching
#     checkpoint key" check that doesn't know about this model's tied k_norm
#     weights on the shared-KV layers (text_config.num_kv_shared_layers=18,
#     i.e. layers 24-41 of 42) -> `ValueError: Following weights were not
#     initialized from checkpoint: {...k_norm.weight}` at engine startup, for
#     BOTH the raw base checkpoint and any merged LoRA export (transformers'
#     own from_pretrained()/tie_weights() resolves these fine - this is a
#     vLLM-loader gap, not missing/corrupt data). INFER_BACKEND=pt routes
#     through ms-swift's native transformers backend instead, sidestepping
#     the strict loader entirely (slower, no vLLM batching, but correct).
# ============================================================================

run_infer () {
    : "${SPLIT:=dev}"
    : "${MAX_NEW_TOKENS:=64}"
    : "${MAX_MODEL_LEN:=4096}"
    : "${MAX_BATCH:=16}"
    : "${INFER_BACKEND:=vllm}"
    # NB: not `: "${MM_LIMIT:={\"audio\": 1}}"` - bash's ${VAR:=word} closes
    # `word` at the FIRST unescaped `}` (a bare literal `{` doesn't open a
    # nested level, only a nested `${` would), so that form assigns MM_LIMIT
    # the truncated '{"audio": 1' and silently drops the closing brace.
    if [ -z "${MM_LIMIT:-}" ]; then
        MM_LIMIT='{"audio": 1}'
    fi
    : "${PRE_ENV:=}"
    # NB: do NOT do `EXTRA_SWIFT_ARGS=("${EXTRA_SWIFT_ARGS[@]:-}")` here - bash's
    # `${arr[@]:-word}` treats an unset OR empty array as triggering the `:-`
    # fallback, so it turns a zero-element array into a one-element array
    # containing "" (the empty `word`). That phantom "" then gets passed to
    # `swift infer` as a real CLI token and crashes with
    # `ValueError: remaining_argv: ['']`. Without `set -u`, leaving
    # EXTRA_SWIFT_ARGS unset/empty already expands to zero words below - no
    # reinitialization needed.
    if [ -z "${EXTRA_SWIFT_ARGS+x}" ]; then
        EXTRA_SWIFT_ARGS=()
    fi

    local input="$PROJ/data/swift/$SUBTASK/$FAMILY/${SPLIT}_infer.jsonl"
    local outdir="$PROJ/outputs/inference/$SUBTASK/$FAMILY"
    local result="$outdir/${MODEL_NAME}_${SPLIT}.jsonl"
    mkdir -p "$outdir"

    echo "Node: $(hostname)"; nvidia-smi || true; date
    gcc_sanity || true
    echo "== $MODEL_NAME | $SUBTASK | $FAMILY | split=$SPLIT =="
    echo "   input : $input"
    echo "   result: $result"

    if [ ! -f "$input" ]; then
        echo "ERROR: input JSONL not found - run scripts/data_prep/prepare_slurptn.sh first." >&2
        exit 1
    fi
    if [ -f "$result" ]; then
        echo "Results already exist ($(wc -l < "$result") rows) - skipping."
        return 0
    fi

    BACKEND_ARGS=()
    if [ "$INFER_BACKEND" = "vllm" ]; then
        BACKEND_ARGS+=(
            --vllm_gpu_memory_utilization 0.9
            --vllm_max_model_len "$MAX_MODEL_LEN"
            --vllm_limit_mm_per_prompt "$MM_LIMIT"
        )
    fi

    # --remove_unused_columns false keeps our meta column (utterance id + gold)
    # so the evaluators can join by id, exactly as the Qwen3-Omni project relies on.
    env $PRE_ENV \
    swift infer \
        --model "$MODEL" \
        --use_hf true \
        --infer_backend "$INFER_BACKEND" \
        --torch_dtype bfloat16 \
        --val_dataset "$input" \
        --stream false \
        --max_batch_size "$MAX_BATCH" \
        "${BACKEND_ARGS[@]}" \
        --result_path "$result" \
        --logprobs false \
        --temperature 0 \
        --max_new_tokens "$MAX_NEW_TOKENS" \
        --remove_unused_columns false \
        --dataset_shuffle false \
        "${EXTRA_SWIFT_ARGS[@]}"

    date
    echo "Done. Output: $result  ($(wc -l < "$result") rows)"
}
