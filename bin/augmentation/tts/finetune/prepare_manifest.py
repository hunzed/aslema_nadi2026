#!/usr/bin/env python3
"""Build VoxCPM2 fine-tuning manifests from the SLURP-TN TRAIN split only.

Format per the official walkthrough: {"audio": "/path.wav", "text": "..."}
Val = last 50 train rows (carved off train - dev stays untouched).
"""
import os
import csv
import json
from pathlib import Path

NADI = Path(os.environ["PROJ"])
OUT = NADI / "augmentation/tts/finetune"
OUT.mkdir(parents=True, exist_ok=True)

rows = []
with open(NADI / "data/slurptn/data/manifests/train.csv", encoding="utf-8") as f:
    for r in csv.DictReader(f):
        if Path(r["wav"]).exists() and r["tun_transcription"].strip():
            rows.append({"audio": r["wav"],
                         "text": r["tun_transcription"].strip()})

train, val = rows[:-50], rows[-50:]
for name, part in (("slurptn_tts_train.jsonl", train),
                   ("slurptn_tts_val.jsonl", val)):
    with open(OUT / name, "w", encoding="utf-8") as f:
        for r in part:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"{OUT / name}: {len(part)} rows")
