#!/bin/bash -l

#SBATCH -J nadi5_voxcpm_ft
#SBATCH -o ${PROJ}/bin/augmentation/logs/voxcpm_ft_%j.out
#SBATCH -e ${PROJ}/bin/augmentation/logs/voxcpm_ft_%j.err
#SBATCH -p gpu-H200
#SBATCH -q ${SLURM_QOS}
#SBATCH --gres=gpu:2
#SBATCH -A ${SLURM_ACCOUNT}
#SBATCH -c 32
#SBATCH --mem=120GB

# ============================================================================
# LoRA fine-tune VoxCPM2 on the SLURP-TN TRAIN split (~5h Tunisian speech)
# to pull its Arabic phonology toward Derja. Official recipe:
# torchrun scripts/train_voxcpm_finetune.py with the voxcpm_v2 LoRA yaml,
# manifests = {"audio","text"} jsonl. 2x H200 (LoRA ~20GB + DDP overhead).
#
#   sbatch augmentation/tts/finetune/sbatch_finetune.sh
#
# Steps: clone repo (if missing) -> build manifests -> patch the official
# LoRA yaml (paths + iters) -> torchrun x2. Checkpoints:
#   augmentation/tts/finetune/checkpoints/slurptn_lora/step_*/ (+ latest symlink)
# Inference with the adapter afterwards (hot-swap):
#   repo scripts/test_voxcpm_lora_infer.py --lora_ckpt .../latest
# ============================================================================
set -e
NADI=${PROJ}
TTS="$NADI/augmentation/tts"
FT="$TTS/finetune"
VPY="$TTS/venv/bin/python"
export HF_HOME="$NADI/custom_gcp/hf_cache"
export TMPDIR="$TTS/tmp"; mkdir -p "$TMPDIR"

echo "Node: $(hostname)"; nvidia-smi | head -12; date

# 1. repo (training scripts/configs are not in the pip package)
if [ ! -d "$FT/VoxCPM" ]; then
    git clone --depth 1 https://github.com/OpenBMB/VoxCPM.git "$FT/VoxCPM"
fi

# 2. manifests (train-only, idempotent)
"$VPY" "$FT/prepare_manifest.py"

# 3. resolve the local VoxCPM2 snapshot (downloaded by earlier inference runs)
SNAP=$(ls -d "$HF_HOME"/hub/models--openbmb--VoxCPM2/snapshots/*/ 2>/dev/null | head -1)
if [ -z "$SNAP" ]; then
    echo "VoxCPM2 weights not in $HF_HOME yet - downloading..."
    "$VPY" -c "from huggingface_hub import snapshot_download; print(snapshot_download('openbmb/VoxCPM2'))"
    SNAP=$(ls -d "$HF_HOME"/hub/models--openbmb--VoxCPM2/snapshots/*/ | head -1)
fi
echo "pretrained: $SNAP"

# 4. patch the official LoRA yaml with our paths/schedule
SRC=$(ls "$FT"/VoxCPM/conf/voxcpm_v2/*lora*.yaml | head -1)
CFG="$FT/slurptn_lora.yaml"
SNAP="$SNAP" SRC="$SRC" CFG="$CFG" FT="$FT" "$VPY" - <<'EOF'
import os, yaml
src, cfg, ft, snap = os.environ['SRC'], os.environ['CFG'], os.environ['FT'], os.environ['SNAP']
c = yaml.safe_load(open(src))
def set_deep(d, key, val):   # set key wherever it already exists in the tree
    hit = False
    def rec(x):
        nonlocal hit
        if isinstance(x, dict):
            for k in list(x):
                if k == key: x[k] = val; hit = True
                else: rec(x[k])
    rec(d); return hit
for k, v in {
    "pretrained_path": snap.rstrip("/"),
    "train_manifest": f"{ft}/slurptn_tts_train.jsonl",
    "val_manifest":   f"{ft}/slurptn_tts_val.jsonl",
    # 2627 rows / (2 GPUs * batch 2 * accum 8) ~ 82 steps/epoch -> 2 epochs
    "num_iters": 170,
    "warmup_steps": 10,
}.items():
    if not set_deep(c, k, v):
        c[k] = v
# checkpoints/logs under our tree (the repo config uses save_path/tensorboard)
for k in ("save_path", "save_dir", "output_dir", "exp_dir", "checkpoint_dir"):
    set_deep(c, k, f"{ft}/checkpoints/slurptn_lora")
set_deep(c, "tensorboard", f"{ft}/checkpoints/slurptn_lora/tb")
yaml.safe_dump(c, open(cfg, "w"), allow_unicode=True, sort_keys=False)
print("patched config ->", cfg)
print({k: c.get(k) for k in ("pretrained_path","train_manifest","num_iters")})
EOF

# 5. train (2 GPUs)
cd "$FT/VoxCPM"
PYTHONPATH="$FT/VoxCPM" "$TTS/venv/bin/torchrun" --nproc_per_node=2 \
    scripts/train_voxcpm_finetune.py --config_path "$CFG"

date
echo "Done. Adapter: $FT/checkpoints/slurptn_lora/ (latest symlink)"
