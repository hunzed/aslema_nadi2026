#!/bin/bash
# Submit the full dual-variant synthesis: 12,138 texts x {base, fine-tuned}
# = ~24k clips, 2 GPUs per variant (4 jobs total, ~1.5-2.5 h each).
#   bash augmentation/tts/submit_all_tts.sh
#   DRY_RUN=1 bash augmentation/tts/submit_all_tts.sh
set -e
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

for shard in 0 1; do
    for variant in base ft; do
        if [ "$variant" = ft ]; then exp="LORA=1,WAV_DIR=wav_ft"; else exp="LORA=0,WAV_DIR=wav"; fi
        cmd=(sbatch --export=ALL,$exp,SHARD=$shard,NSHARDS=2 "$HERE/sbatch_tts.sh")
        echo "+ ${cmd[*]}"
        [ -z "${DRY_RUN:-}" ] && "${cmd[@]}"
    done
done
echo "4 jobs submitted. Watch: squeue -u \$USER | logs: augmentation/logs/tts_*.out"
echo "Outputs: augmentation/data/wav/ (base) and augmentation/data/wav_ft/ (fine-tuned)"
