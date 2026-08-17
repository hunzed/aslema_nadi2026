#!/bin/bash -l

#SBATCH -J nadi5_eval_intent
#SBATCH -o ${PROJ}/logs/intent/eval_%j.out
#SBATCH -e ${PROJ}/logs/intent/eval_%j.err
#SBATCH -p cpu-all
#SBATCH -c 4
#SBATCH --mem=16GB

# Score every Subtask 5.1 (Intent) result file present under
# outputs/inference/intent/{qwen,gemma,whisper}/ and write per-run metrics + the
# shared {"id","hyp","ref"} classifications file under results/intent/<run>/.
# CPU-only (no modules/GPU). Default split = dev; override with SPLIT=test.
set -e
export NO_MODULES=1
# $0 is a Slurm spool path under sbatch, not this file's real location.
PROJ_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
source "${PROJ}/scripts/common.sh"

SPLIT="${SPLIT:-dev}"
LABELS="$PROJ/data/slurptn/intent_labels.txt"
REF="$PROJ/data/slurptn/data/manifests/$([ "$SPLIT" = test ] && echo test || echo validation).csv"

shopt -s nullglob
for family in qwen gemma whisper; do
    for result in "$PROJ"/outputs/inference/intent/$family/*_${SPLIT}.jsonl; do
        run="$(basename "${result%.jsonl}")"     # e.g. qwen2_5_omni_3b_dev
        echo "==== scoring $family / $run ===="
        python "$PROJ/eval/eval_intent.py" "$result" \
            --labels "$LABELS" \
            --reference "$REF" \
            --out-dir "$PROJ/results/intent/$run"
    done
done
echo "Intent evaluation complete. Metrics under $PROJ/results/intent/"
