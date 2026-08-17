#!/usr/bin/env python3
"""Package a run's predictions into a NADI-2026 submission zip.

  * Subtask 5.1: {"id": "<Segment_ID>", "hyp": "<predicted_intent>"}
  * Subtask 5.2: {"id": ..., "hyp": ...}   (annotated transcription)
Held-out submissions drop the "ref" field, keeping id + hyp only.

Usage:
    python make_submission.py --subtask 5.1 --team myteam \
        --predictions results/intent/qwen2_5_omni_3b_test/classifications.jsonl \
        --out results/intent/nadi2026_subtask_5.1_myteam.zip
"""
import argparse
import json
import zipfile
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subtask", required=True, choices=["5.1", "5.2"])
    ap.add_argument("--team", required=True)
    ap.add_argument("--predictions", required=True,
                    help="classifications.jsonl (5.1) or predictions.jsonl (5.2)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(args.predictions, encoding="utf-8") if l.strip()]
    packed = [{"id": r.get("id"), "hyp": r.get("hyp")} for r in rows]

    staged = Path(args.out).with_suffix(".predictions.jsonl")
    with open(staged, "w", encoding="utf-8") as w:
        for r in packed:
            w.write(json.dumps(r, ensure_ascii=False) + "\n")

    with zipfile.ZipFile(args.out, "w", zipfile.ZIP_DEFLATED) as z:
        z.write(staged, arcname="predictions.jsonl")
    staged.unlink()

    print(f"Wrote {len(packed)} predictions -> {args.out}")
    print(f"(expected name pattern: nadi2026_subtask_{args.subtask}_{args.team}.zip)")


if __name__ == "__main__":
    main()
