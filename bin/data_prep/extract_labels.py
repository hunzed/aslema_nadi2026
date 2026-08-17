#!/usr/bin/env python3
"""Derive the intent + slot label inventories from the SLURP-TN manifests.

The reference notebooks build both inventories from ALL THREE splits (train,
validation, test) so that every label the evaluation might contain has a slot
in the prompt - two intent labels occur only in val/test, so a train-only
inventory would be incomplete. We do the same here.

Reads:  <manifest_dir>/{train,validation,test}.csv
Writes: <out_dir>/intent_labels.txt   (one label per line, sorted)
        <out_dir>/slot_labels.txt      (one label per line, sorted)

Usage:
    python extract_labels.py --manifest-dir data/slurptn/data/manifests \
                             --out-dir data/slurptn
"""
import argparse
import csv
import re
from pathlib import Path

SLOT_RE = re.compile(r"<([^<>/\s]+)>")


def read_column(manifest_dir, column):
    values = []
    for split in ("train", "validation", "test"):
        path = Path(manifest_dir) / f"{split}.csv"
        if not path.exists():
            print(f"  (skip missing manifest {path})")
            continue
        with open(path, encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                v = (row.get(column) or "").strip()
                if v:
                    values.append(v)
    return values


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    intents = sorted({v for v in read_column(args.manifest_dir, "intent")})
    annotations = read_column(args.manifest_dir, "tun_slu_annotation")
    slots = sorted({m for ann in annotations for m in SLOT_RE.findall(ann)})

    (out_dir / "intent_labels.txt").write_text("\n".join(intents) + "\n", encoding="utf-8")
    (out_dir / "slot_labels.txt").write_text("\n".join(slots) + "\n", encoding="utf-8")

    print(f"Wrote {len(intents)} intent labels -> {out_dir / 'intent_labels.txt'}")
    print(f"Wrote {len(slots)} slot labels   -> {out_dir / 'slot_labels.txt'}")


if __name__ == "__main__":
    main()
