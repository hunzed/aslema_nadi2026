#!/bin/bash -l

#SBATCH -J nadi5_voxcpm_tts
#SBATCH -o ${PROJ}/augmentation/logs/tts_%j.out
#SBATCH -e ${PROJ}/augmentation/logs/tts_%j.err
#SBATCH -p gpu-H200
#SBATCH -q ${SLURM_QOS}
#SBATCH --gres=gpu:1
#SBATCH -A ${SLURM_ACCOUNT}
#SBATCH -c 16
#SBATCH --mem=64GB

# VoxCPM2 synthesis of the tts_queue (resume-safe). Knobs via --export:
#   LORA=1                 use the Derja fine-tune adapter
#   QUEUE=<file>           work list under augmentation/data/ (default tts_queue.jsonl)
#   WAV_DIR=wav_ft         output dir under augmentation/data/ (default wav)
#   SHARD=0 NSHARDS=2      split the queue across parallel jobs
#   LIMIT=5                smoke run
# Dual-variant 2+2 GPU launch: bash augmentation/tts/submit_all_tts.sh
set -e
NADI=${PROJ}
export HF_HOME="$NADI/custom_gcp/hf_cache"       # writable; model ~5GB on first run
export TMPDIR="$NADI/augmentation/tts/tmp"; mkdir -p "$TMPDIR"

echo "Node: $(hostname)"; nvidia-smi | head -12; date
"$NADI/augmentation/tts/venv/bin/python" "$NADI/augmentation/tts/synthesize.py" \
    ${LIMIT:+--limit $LIMIT}
date
