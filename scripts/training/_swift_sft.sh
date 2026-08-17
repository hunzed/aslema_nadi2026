#!/bin/bash -l
# ============================================================================
# Shared LoRA SFT driver for all fine-tuning runs. Sourced by the per-model
# scripts under qwen/ and gemma/. They set a handful of variables and call
# run_sft (single-subtask) or run_sft_joint (both subtasks at once, two-stage
# encoder freeze schedule - see that function's own docstring below);
# everything common (path layout, guards, the swift sft call) lives here -
# the sibling of scripts/inference/_swift_infer.sh.
#
# HYPERPARAMETERS: the flag set + defaults below mirror, 1:1, the PROVEN
# recipe from the sibling reference project's
#   Qwen3-Omni-30B-A3B-Instruct/scripts/train/train_qwen3omni_30b_lora.sh
# (itself based on ms-swift's official examples/models/qwen3_omni/zero3.sh),
# which trained successfully on this cluster. Deliberately NO extra tuning
# flags beyond that set - 1 epoch / lr 1e-4 / rank 16 / eff. batch 8 keeps
# runs short. The only additions are non-hyperparam infrastructure:
# --output_dir/--add_version false (predictable path for the resume-skip
# guard and merge_lora.sh) and --use_hf true (project convention).
#
# Variables the caller sets before `run_sft`:
#   MODEL            HF repo id (e.g. Qwen/Qwen2.5-Omni-7B)
#   MODEL_NAME       short slug for the output dir (e.g. qwen2_5_omni_7b)
#   FAMILY           qwen | gemma   (selects data/swift/<sub>/<FAMILY>/ inputs)
#   SUBTASK          intent | slot_filling
#   PRE_ENV          extra env assignments prefixed to swift (e.g. ENABLE_AUDIO_OUTPUT=0)
#   EXTRA_SWIFT_ARGS bash array of extra flags (e.g. --experts_impl grouped_mm)
# Tunables (all overridable via sbatch --export=ALL,NAME=value; defaults = the
# reference recipe's values):
#   NUM_EPOCHS=1  LR=1e-4  LORA_RANK=16  LORA_ALPHA=32
#   BATCH=4  GRAD_ACCUM=2  MAX_LENGTH=4096  SAVE_STEPS=50  EVAL_STEPS=50
#   TARGET_MODULES="q_proj k_proj v_proj o_proj"
#   TARGET_REGEX=<regex>            scope LoRA by path instead of bare name
#                                    (needed on gemma - see note below); when
#                                    set, takes priority over TARGET_MODULES
#   RESUME_FROM=<checkpoint dir>   resume an interrupted run
#   DEEPSPEED=zero3                multi-GPU sharding. NOT the default and not
#                                  recommended: see the multi-GPU guard below.
#   ALLOW_MULTI_GPU=1              explicit opt-in required to run on >1 GPU
#
# ms-swift 4.x notes baked in (see claude_notes / the Qwen3-Omni project):
#  * --tuner_type (renamed from 3.x --train_type);
#  * NEVER --target_modules all-linear on Qwen-Omni - peft 0.19 splits the
#    regex into single chars ("Target modules {'l','*',...} not found");
#    modules are listed explicitly instead;
#  * gemma-4 ONLY: bare --target_modules q_proj k_proj v_proj o_proj matches
#    by suffix anywhere in the model, including the vision tower's own
#    q/k/v/o_proj. Gemma-4's vision/audio towers wrap their linears in a
#    custom Gemma4ClippableLinear (config shows "use_clipped_linears": true
#    on vision_config/audio_config, absent from text_config) that PEFT's LoRA
#    injector doesn't recognize -> `ValueError: Target module
#    Gemma4ClippableLinear(...) is not supported` from get_peft_model, crashing
#    before the first training step (freeze_vit/freeze_aligner don't prevent
#    this - injection walks the whole model before freezing applies). Fix:
#    set TARGET_REGEX to exclude vision_tower/audio_tower, e.g.
#    '^(?!.*(vision_tower|audio_tower)).*\.(q_proj|k_proj|v_proj|o_proj)$'
#    (the gemma sft scripts default to this already);
#  * --attn_impl sdpa --padding_free false: flash-attn does not build on this
#    cluster (torch-cu13 wheels vs nvcc <= 12.6) - sdpa is fully supported,
#    just slightly slower. vLLM inference later is unaffected (own kernels);
#  * eval data: like the reference, --split_dataset_ratio 0.01 carves 1% off
#    the train jsonl for eval loss (cheap at --eval_steps 50). dev_sft.jsonl
#    exists but is NOT wired in as --val_dataset on purpose: evaluating the
#    full dev set every 50 steps would dominate the wall clock;
#  * training runs on the transformers backend, NOT vLLM - so the gemma-4
#    vLLM audio-batching crash (gemma4_mm.py list-vs-tensor) does NOT apply
#    here; it comes back at post-SFT inference (use infer_finetuned.sh which
#    keeps --vllm_max_num_seqs 1 for gemma).
# ============================================================================

# ----------------------------------------------------------------------------
# Multi-GPU guard. LoRA SFT in this project fits on ONE H200 for every model we
# run, up to and including the 30B MoE (frozen base + small adapter). Asking
# for 2 GPUs doubles the allocation on a shared cluster, buys no throughput,
# and on the 30B MoE actively SIGBUSes (job 354650: both torchrun ranks
# mmap-load the full ~60GB checkpoint before zero3 partitioning takes effect,
# 225GB host mem against a 200GB request on a ~201.5GB node). It failed
# intermittently rather than always, which is exactly why it kept getting
# re-adopted from the reference project and kept wasting two GPUs per run.
#
# So: >1 GPU is now an explicit, deliberate opt-in, never a default anyone can
# inherit by copying a script header.
#   sbatch --gres=gpu:2 --export=ALL,ALLOW_MULTI_GPU=1,DEEPSPEED=zero3,NPROC_PER_NODE=2 <script>
# ----------------------------------------------------------------------------
guard_single_gpu () {
    local n="${SLURM_GPUS_ON_NODE:-}"
    if [ -z "$n" ] && command -v nvidia-smi >/dev/null 2>&1; then
        n="$(nvidia-smi --list-gpus 2>/dev/null | wc -l)"
    fi
    [ -z "$n" ] && return 0                      # can't tell; don't block
    if [ "$n" -gt 1 ] && [ "${ALLOW_MULTI_GPU:-0}" != "1" ]; then
        echo "FATAL: this job was allocated $n GPUs but ALLOW_MULTI_GPU=1 was not set." >&2
        echo "       LoRA SFT here fits on a single H200 for every model including" >&2
        echo "       the 30B MoE; 2-GPU zero3 wastes an allocation and has SIGBUS'd" >&2
        echo "       on host memory (see the multi-GPU guard note in _swift_sft.sh)." >&2
        echo "       Resubmit with --gres=gpu:1, or opt in deliberately with" >&2
        echo "       --export=ALL,ALLOW_MULTI_GPU=1,DEEPSPEED=zero3,NPROC_PER_NODE=2" >&2
        exit 1
    fi
    if [ -n "${DEEPSPEED:-}" ] && [ "${ALLOW_MULTI_GPU:-0}" != "1" ]; then
        echo "FATAL: DEEPSPEED='$DEEPSPEED' set without ALLOW_MULTI_GPU=1." >&2
        echo "       deepspeed here only exists for the multi-GPU path; see above." >&2
        exit 1
    fi
}

run_sft () {
    guard_single_gpu
    : "${NUM_EPOCHS:=1}"
    : "${LR:=1e-4}"
    : "${LORA_RANK:=16}"
    : "${LORA_ALPHA:=32}"
    : "${BATCH:=4}"
    : "${GRAD_ACCUM:=2}"
    : "${MAX_LENGTH:=4096}"
    : "${SAVE_STEPS:=50}"
    : "${EVAL_STEPS:=50}"
    : "${TARGET_MODULES:=q_proj k_proj v_proj o_proj}"
    : "${TARGET_REGEX:=}"
    : "${PRE_ENV:=}"
    if [ -n "$TARGET_REGEX" ]; then
        EXTRA_SWIFT_ARGS_TARGET=(--target_regex "$TARGET_REGEX")
    else
        EXTRA_SWIFT_ARGS_TARGET=(--target_modules $TARGET_MODULES)
    fi
    # NB: same array footgun as _swift_infer.sh - never `${arr[@]:-}` (turns an
    # empty array into ("")); only initialize if truly unset.
    if [ -z "${EXTRA_SWIFT_ARGS+x}" ]; then
        EXTRA_SWIFT_ARGS=()
    fi
    if [ -n "${DEEPSPEED:-}" ]; then
        EXTRA_SWIFT_ARGS+=(--deepspeed "$DEEPSPEED")
    fi

    local train="$PROJ/data/swift/$SUBTASK/$FAMILY/train_sft.jsonl"
    local outdir="$PROJ/models/$FAMILY/${MODEL_NAME}_${SUBTASK}_lora"

    echo "Node: $(hostname)"; nvidia-smi || true; date
    gcc_sanity || true
    echo "== SFT $MODEL_NAME | $SUBTASK | $FAMILY =="
    echo "   train : $train"
    echo "   outdir: $outdir"

    if [ ! -f "$train" ]; then
        echo "ERROR: $train not found - re-run scripts/data_prep/prepare_slurptn.sh" >&2
        echo "(the SFT jsonls were added 2026-07-23; an older data-prep run only built *_infer.jsonl)" >&2
        exit 1
    fi
    # Resume-skip: a finished run leaves checkpoints in outdir. Refuse to
    # silently retrain over them; resume instead with RESUME_FROM=<ckpt>.
    if [ -z "${RESUME_FROM:-}" ] && ls "$outdir"/checkpoint-* >/dev/null 2>&1; then
        echo "Checkpoints already exist under $outdir - skipping."
        echo "To resume:  sbatch --export=ALL,RESUME_FROM=$outdir/checkpoint-XXXX <this script>"
        echo "To retrain: delete/move $outdir first."
        return 0
    fi
    if [ -n "${RESUME_FROM:-}" ]; then
        EXTRA_SWIFT_ARGS+=(--resume_from_checkpoint "$RESUME_FROM")
    fi
    mkdir -p "$outdir"

    env $PRE_ENV \
    swift sft \
        --model "$MODEL" \
        --use_hf true \
        --dataset "$train" \
        --split_dataset_ratio 0.01 \
        --load_from_cache_file true \
        --tuner_type lora \
        --torch_dtype bfloat16 \
        --num_train_epochs "$NUM_EPOCHS" \
        --per_device_train_batch_size "$BATCH" \
        --per_device_eval_batch_size "$BATCH" \
        --attn_impl sdpa \
        --learning_rate "$LR" \
        --lora_rank "$LORA_RANK" \
        --lora_alpha "$LORA_ALPHA" \
        "${EXTRA_SWIFT_ARGS_TARGET[@]}" \
        --freeze_vit true \
        --freeze_aligner true \
        --padding_free false \
        --gradient_accumulation_steps "$GRAD_ACCUM" \
        --gradient_checkpointing true \
        --eval_steps "$EVAL_STEPS" \
        --save_steps "$SAVE_STEPS" \
        --save_total_limit 2 \
        --logging_steps 5 \
        --max_length "$MAX_LENGTH" \
        --warmup_ratio 0.05 \
        --dataset_num_proc 4 \
        --dataloader_num_workers 4 \
        --output_dir "$outdir" \
        --add_version false \
        "${EXTRA_SWIFT_ARGS[@]}"

    date
    echo "Done. Adapters under: $outdir"
    echo "Next: sbatch --export=ALL,CKPT=$outdir/checkpoint-<best> scripts/training/merge_lora.sh"
}

# ============================================================================
# run_sft_joint: single-stage LoRA SFT on BOTH subtasks (intent + slot_filling)
# at once, per model. Same tunables as run_sft (LR/LORA_RANK/BATCH/NUM_EPOCHS/
# etc., default NUM_EPOCHS=1, plain). Caller sets MODEL/MODEL_NAME/FAMILY/
# PRE_ENV/EXTRA_SWIFT_ARGS same as for run_sft; SUBTASK is only used for
# logging (data comes from both per-subtask train_sft.jsonl files, passed
# together via ms-swift's multi-value --dataset).
#
# --freeze_vit true --freeze_aligner true throughout - the omni encoder/
# aligner is never touched, matching every single-task script's default; only
# the LoRA-tuned LLM trains. --save_strategy epoch (rather than step-based
# --save_steps) so a checkpoint lands at the end of every epoch.
#
# Output layout (no new data/ directories - reuses the existing per-subtask
# train_sft.jsonl files):
#   $PROJ/models/$FAMILY/${MODEL_NAME}_joint_lora/     <- checkpoints, same
#                                                          layout as run_sft
# ============================================================================
run_sft_joint () {
    guard_single_gpu
    : "${NUM_EPOCHS:=1}"
    : "${LR:=1e-4}"
    : "${LORA_RANK:=16}"
    : "${LORA_ALPHA:=32}"
    : "${BATCH:=4}"
    : "${GRAD_ACCUM:=2}"
    : "${MAX_LENGTH:=4096}"
    : "${EVAL_STEPS:=50}"
    : "${TARGET_MODULES:=q_proj k_proj v_proj o_proj}"
    : "${TARGET_REGEX:=}"
    : "${PRE_ENV:=}"
    if [ -n "$TARGET_REGEX" ]; then
        EXTRA_SWIFT_ARGS_TARGET=(--target_regex "$TARGET_REGEX")
    else
        EXTRA_SWIFT_ARGS_TARGET=(--target_modules $TARGET_MODULES)
    fi
    if [ -z "${EXTRA_SWIFT_ARGS+x}" ]; then
        EXTRA_SWIFT_ARGS=()
    fi
    if [ -n "${DEEPSPEED:-}" ]; then
        EXTRA_SWIFT_ARGS+=(--deepspeed "$DEEPSPEED")
    fi

    local intent_train="$PROJ/data/swift/intent/$FAMILY/train_sft.jsonl"
    local slot_train="$PROJ/data/swift/slot_filling/$FAMILY/train_sft.jsonl"
    local outdir="$PROJ/models/$FAMILY/${MODEL_NAME}_joint_lora"

    echo "Node: $(hostname)"; nvidia-smi || true; date
    gcc_sanity || true
    echo "== Joint SFT (intent + slot_filling) $MODEL_NAME | $FAMILY =="
    echo "   intent train : $intent_train"
    echo "   slot   train : $slot_train"
    echo "   outdir: $outdir"

    for f in "$intent_train" "$slot_train"; do
        if [ ! -f "$f" ]; then
            echo "ERROR: $f not found - re-run scripts/data_prep/prepare_slurptn.sh" >&2
            exit 1
        fi
    done

    # Resume-skip: same convention as run_sft (any existing checkpoint short-
    # circuits; delete/move outdir first to force a retrain).
    if [ -z "${RESUME_FROM:-}" ] && ls "$outdir"/checkpoint-* >/dev/null 2>&1; then
        echo "Checkpoints already exist under $outdir - skipping."
        echo "To resume:  sbatch --export=ALL,RESUME_FROM=$outdir/checkpoint-XXXX <this script>"
        echo "To retrain: delete/move $outdir first."
        return 0
    fi
    if [ -n "${RESUME_FROM:-}" ]; then
        EXTRA_SWIFT_ARGS+=(--resume_from_checkpoint "$RESUME_FROM")
    fi
    mkdir -p "$outdir"

    env $PRE_ENV \
    swift sft \
        --model "$MODEL" \
        --use_hf true \
        --dataset "$intent_train" "$slot_train" \
        --split_dataset_ratio 0.01 \
        --load_from_cache_file true \
        --tuner_type lora \
        --torch_dtype bfloat16 \
        --num_train_epochs "$NUM_EPOCHS" \
        --per_device_train_batch_size "$BATCH" \
        --per_device_eval_batch_size "$BATCH" \
        --attn_impl sdpa \
        --learning_rate "$LR" \
        --lora_rank "$LORA_RANK" \
        --lora_alpha "$LORA_ALPHA" \
        "${EXTRA_SWIFT_ARGS_TARGET[@]}" \
        --freeze_vit true \
        --freeze_aligner true \
        --padding_free false \
        --gradient_accumulation_steps "$GRAD_ACCUM" \
        --gradient_checkpointing true \
        --eval_steps "$EVAL_STEPS" \
        --save_strategy epoch \
        --save_total_limit 2 \
        --logging_steps 5 \
        --max_length "$MAX_LENGTH" \
        --warmup_ratio 0.05 \
        --dataset_num_proc 4 \
        --dataloader_num_workers 4 \
        --output_dir "$outdir" \
        --add_version false \
        "${EXTRA_SWIFT_ARGS[@]}"

    date
    echo "Done. Adapters under: $outdir"
    echo "Next: sbatch --export=ALL,CKPT=$outdir/checkpoint-<best> scripts/training/merge_lora.sh"
}
