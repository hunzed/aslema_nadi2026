#!/usr/bin/env python3
"""Official NADI-2026 shared-task-5 test set -> SLURP-TN-style manifest CSV.

Source: UBC-NLP/nadi26_subtask_5_SLU_test (HF datasets-server confirms one
config "default", one split "clean", 989 rows, features {audio, id}, no gold
labels). The same 989 audio/id pairs serve BOTH subtasks (5.1 intent, 5.2 slot
filling) - there is no per-subtask config, unlike SLURP-TN's own held-out
`test` split which this project already treats as unlabeled/submission-only.

Writes a manifest with the columns bin/data_prep/to_swift_{intent,slot_filling}.py
already expect (ID, wav [, intent | tun_slu_annotation]); leaving the gold
columns absent makes those scripts fall through their existing
`row.get(...) or ""` path, producing an inference-only jsonl exactly like the
existing test split.

Usage:
    python bin/data_prep/prepare_official_test.py \
        --out-dir data/slurptn/data/wav/official_test \
        --manifest data/slurptn/data/manifests/official_test.csv
"""
import argparse
import csv
import io
from pathlib import Path

import soundfile as sf
from datasets import Audio, load_dataset


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="UBC-NLP/nadi26_subtask_5_SLU_test")
    ap.add_argument("--split", default="clean")
    ap.add_argument("--out-dir", required=True, help="dir to write extracted wav files")
    ap.add_argument("--manifest", required=True, help="output manifest CSV path")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = Path(args.manifest)
    manifest.parent.mkdir(parents=True, exist_ok=True)

    # decode=False keeps the audio column as raw bytes so we can decode it
    # with soundfile ourselves - avoids datasets' default torchcodec decode
    # path, which needs ffmpeg shared libs (libavutil.so.*) not present here.
    ds = load_dataset(args.dataset, split=args.split)
    ds = ds.cast_column("audio", Audio(decode=False))

    n = 0
    with open(manifest, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["ID", "wav"])
        for row in ds:
            audio_bytes = row["audio"]["bytes"]
            data, samplerate = sf.read(io.BytesIO(audio_bytes))
            wav_path = out_dir / f"{row['id']}.wav"
            sf.write(wav_path, data, samplerate)
            w.writerow([row["id"], str(wav_path)])
            n += 1

    print(f"Wrote {n} wav files -> {out_dir}")
    print(f"Wrote manifest ({n} rows, no gold labels) -> {manifest}")


if __name__ == "__main__":
    main()
