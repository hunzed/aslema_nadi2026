#!/bin/bash -l

#SBATCH -J nadi5_dataprep
#SBATCH -o ${PROJ}/logs/data_prep/prep_%j.out
#SBATCH -e ${PROJ}/logs/data_prep/prep_%j.err
#SBATCH -p cpu-all
#SBATCH -c 16
#SBATCH --mem=64GB

# ============================================================================
# One-shot data prep for both subtasks (CPU, no GPU).
#   1. clone the official SLURP-TN baseline repo under bin/_vendor (for its
#      data_prep.py + evaluation/scoring.py - same two pieces the notebooks use);
#   2. run data_prep.py -> downloads Elyadata/SLURP-TN, exports audio, writes
#      data/slurptn/data/manifests/{train,validation,test}.csv;
#   3. extract the intent + slot label inventories from all splits;
#   4. build the ms-swift inference JSONLs for BOTH families x BOTH subtasks x
#      the dev (validation) and test splits;
#   5. build the ms-swift SFT JSONLs (with the gold assistant/target turn) from
#      the train split (+ a dev_sft.jsonl; NOT wired into training - the SFT
#      driver mirrors the reference recipe's --split_dataset_ratio 0.01 - but
#      kept for ad-hoc eval-loss checks on real dev).
# Idempotent: re-running skips the clone if present and overwrites the jsonls -
# safe to re-run on a server that already has the inference jsonls just to add
# the SFT ones.
# ============================================================================
set -e
export NO_MODULES=1   # data prep is CPU-only; don't require cuda/gcc modules
# $0 is a Slurm spool path under sbatch, not this file's real location.
PROJ_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
source "${PROJ}/scripts/common.sh"

cd "$PROJ"
VENDOR="$PROJ/data/vendor/SLURP-TN-baselines"
DATA="$PROJ/data/slurptn"
MANIFESTS="$DATA/data/manifests"

echo "Node: $(hostname)"; date

# 1. baseline repo (data_prep.py + evaluation/scoring.py)
if [ ! -d "$VENDOR/.git" ]; then
    mkdir -p "$PROJ/data/vendor"
    git clone https://github.com/elyadata/SLURP-TN-baselines.git "$VENDOR"
fi

# 2. build the SpeechBrain manifests + export audio (bracket->inline conversion)
if [ ! -f "$MANIFESTS/validation.csv" ]; then
    ( cd "$VENDOR" && python -u data_prep.py --data_folder "$DATA/data" )
else
    echo "Manifests already present at $MANIFESTS - skipping data_prep.py"
fi

# 3. label inventories (both subtasks read these)
python "$PROJ/data_prep/extract_labels.py" \
    --manifest-dir "$MANIFESTS" --out-dir "$DATA"

# 4. ms-swift inference JSONLs: family x subtask x split
build () {  # $1=script $2=labels $3=subtask-dir
    local script="$1" labels="$2" sub="$3"
    for family in qwen gemma; do
        # dev = validation split (has gold refs -> scorable)
        python "$PROJ/data_prep/$script" \
            --manifest "$MANIFESTS/validation.csv" --labels "$labels" \
            --family "$family" --split dev \
            --out "$PROJ/data/swift/$sub/$family/dev_infer.jsonl"
        # test = held-out evaluation split (submission; refs may be absent)
        python "$PROJ/data_prep/$script" \
            --manifest "$MANIFESTS/test.csv" --labels "$labels" \
            --family "$family" --split test \
            --out "$PROJ/data/swift/$sub/$family/test_infer.jsonl"
        # SFT jsonls (gold assistant turn appended): train split for training,
        # dev (validation) kept for ad-hoc eval-loss checks (not used by default).
        python "$PROJ/data_prep/$script" \
            --manifest "$MANIFESTS/train.csv" --labels "$labels" \
            --family "$family" --split train --with-target \
            --out "$PROJ/data/swift/$sub/$family/train_sft.jsonl"
        python "$PROJ/data_prep/$script" \
            --manifest "$MANIFESTS/validation.csv" --labels "$labels" \
            --family "$family" --split dev --with-target \
            --out "$PROJ/data/swift/$sub/$family/dev_sft.jsonl"
    done
}

build to_swift_intent.py       "$DATA/intent_labels.txt" intent
build to_swift_slot_filling.py "$DATA/slot_labels.txt"   slot_filling

date
echo "Data prep complete."
echo "  manifests : $MANIFESTS"
echo "  swift data: $PROJ/data/swift/{intent,slot_filling}/{qwen,gemma}/{dev,test}_infer.jsonl"
echo "  sft data  : $PROJ/data/swift/{intent,slot_filling}/{qwen,gemma}/{train,dev}_sft.jsonl"
