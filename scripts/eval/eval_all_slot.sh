#!/bin/bash -l

#SBATCH -J nadi5_eval_slot
#SBATCH -o ${PROJ}/logs/slot_filling/eval_%j.out
#SBATCH -e ${PROJ}/logs/slot_filling/eval_%j.err
#SBATCH -p cpu-all
#SBATCH -c 4
#SBATCH --mem=16GB

# Score every Subtask 5.2 (Slot Filling) result file under
# outputs/inference/slot_filling/{qwen,gemma,whisper}/ with the official SLURP-TN
# WER/CER/CoER/CVER scorer, writing predictions.jsonl + metrics.json under
# results/slot_filling/<run>/. CPU-only. Default split = dev; override SPLIT=test.
set -e
export NO_MODULES=1
# $0 is a Slurm spool path under sbatch, not this file's real location.
PROJ_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
source "${PROJ}/scripts/common.sh"

SPLIT="${SPLIT:-dev}"
VENDOR="$PROJ/data/vendor/SLURP-TN-baselines"
REF="$PROJ/data/slurptn/data/manifests/$([ "$SPLIT" = test ] && echo test || echo validation).csv"

shopt -s nullglob
for family in qwen gemma whisper; do
    for result in "$PROJ"/outputs/inference/slot_filling/$family/*_${SPLIT}.jsonl; do
        run="$(basename "${result%.jsonl}")"
        echo "==== scoring $family / $run ===="
        python "$PROJ/eval/eval_slot_filling.py" "$result" \
            --vendor "$VENDOR" \
            --reference "$REF" \
            --out-dir "$PROJ/results/slot_filling/$run"
    done
done
echo "Slot-filling evaluation complete. Metrics under $PROJ/results/slot_filling/"
