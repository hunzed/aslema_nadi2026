#!/bin/bash
# Create an isolated venv for VoxCPM2 (torch>=2.5 would fight the nadi_task5
# env's pins, so it gets its own). Run once, anywhere with internet:
#   bash augmentation/tts/setup_voxcpm.sh
set -e
NADI=${PROJ}
TTS="$NADI/augmentation/tts"
export TMPDIR="$TTS/tmp"; mkdir -p "$TMPDIR"
export PIP_CACHE_DIR="$TTS/pip_cache"

if [ ! -x "$TTS/venv/bin/python" ]; then
    "$NADI/conda/envs/nadi_task5/bin/python3.11" -m venv "$TTS/venv"
fi
"$TTS/venv/bin/pip" install --upgrade pip
# voxcpm pulls torch/torchaudio; librosa+soundfile for 48k->16k resampling
"$TTS/venv/bin/pip" install voxcpm librosa soundfile

"$TTS/venv/bin/python" - <<'EOF'
import torch, voxcpm, librosa, soundfile
print("voxcpm OK | torch", torch.__version__, "| cuda avail:", torch.cuda.is_available())
EOF
echo "venv ready: $TTS/venv (cuda:False here is fine - GPU comes from the sbatch node)"
