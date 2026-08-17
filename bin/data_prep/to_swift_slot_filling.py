#!/usr/bin/env python3
"""SLURP-TN manifest -> ms-swift JSONL for Subtask 5.2 (Slot Filling).

Two modes:
  * inference (default): rows WITHOUT an assistant/target turn. The gold
    annotation is kept in `meta.ref` so bin/eval/eval_slot_filling.py can build
    the shared {"id","ref","hyp"} file the SLURP-TN scorer expects, joined by id.
  * --with-target (SFT): rows WITH a trailing assistant turn holding the gold
    inline slot-annotated transcription (`tun_slu_annotation`) verbatim, i.e.
    the ms-swift training format. Rows with no gold annotation are skipped.

Per-family difference (why data/swift/slot_filling is split qwen/ vs gemma/):
only the `<audio>` placeholder position in the user turn - same rationale as
to_swift_intent.py.

Usage:
    python to_swift_slot_filling.py --manifest data/slurptn/data/manifests/validation.csv \
        --labels data/slurptn/slot_labels.txt --family qwen \
        --split dev --out data/swift/slot_filling/qwen/dev_infer.jsonl
"""
import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # for prompts/
from prompts.slot_filling_prompts import build_system_prompt, load_labels, USER_INSTRUCTION


def build_user_content(family):
    # The reference notebook uses a bare "<audio>" user turn for Qwen. Gemma-4
    # wants audio after text, so we add the short instruction line for it (and,
    # for symmetry, keep it available to qwen too but audio-first).
    if family == "gemma":
        return f"{USER_INSTRUCTION}\n<audio>"
    return "<audio>"  # qwen: matches the reference notebook exactly


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--family", choices=["qwen", "gemma"], required=True)
    ap.add_argument("--split", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--with-target", action="store_true",
                    help="append the gold assistant turn (ms-swift SFT format); "
                         "skips rows with no gold annotation")
    args = ap.parse_args()

    system_prompt = build_system_prompt(load_labels(args.labels))
    user_content = build_user_content(args.family)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    n = 0
    skipped = 0
    with open(args.manifest, encoding="utf-8") as fh, open(out, "w", encoding="utf-8") as w:
        for row in csv.DictReader(fh):
            ref = (row.get("tun_slu_annotation") or "").strip()
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ]
            if args.with_target:
                if not ref:
                    skipped += 1
                    continue
                # target = the gold annotated transcription verbatim - the same
                # single line the zero-shot prompt asks for, so both output
                # styles go through the identical scorer path.
                messages.append({"role": "assistant", "content": ref})
            rec = {
                "messages": messages,
                "audios": [row["wav"]],
                "meta": {
                    "id": row["ID"],
                    "subtask": "5.2",
                    "family": args.family,
                    "split": args.split,
                    "ref": ref,
                },
            }
            w.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n += 1

    kind = "SFT" if args.with_target else "inference"
    print(f"Wrote {n} slot-filling {kind} rows ({args.family}/{args.split}) -> {out}"
          + (f" (skipped {skipped} rows without gold annotation)" if skipped else ""))


if __name__ == "__main__":
    main()
