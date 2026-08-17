#!/usr/bin/env python3
"""SLURP-TN manifest -> ms-swift JSONL for Subtask 5.1 (Intent).

Two modes:
  * inference (default): rows WITHOUT an assistant/target turn. The gold
    intent is kept in `meta.intent` so bin/eval/eval_intent.py can join scores
    back by utterance id (ms-swift preserves `meta` when infer runs with
    --remove_unused_columns false, exactly as the Qwen3-Omni project relies on).
  * --with-target (SFT): rows WITH a trailing assistant turn holding the gold
    answer in the exact output format the prompt requests
    ('{"intent": "<label>"}'), i.e. the ms-swift training format. Rows whose
    manifest has no gold intent are skipped (can't train on them).

Per-family difference (the reason data/swift/intent is split into qwen/ and
gemma/): the `<audio>` placeholder position in the user turn.
  * qwen : audio first, then instruction   (Qwen2.5-Omni is order-insensitive;
           matches the reference notebook's "<audio>" leading user content)
  * gemma: instruction first, then audio    (Gemma-4 model card: place audio
           content AFTER text)

Usage:
    python to_swift_intent.py --manifest data/slurptn/data/manifests/validation.csv \
        --labels data/slurptn/intent_labels.txt --family qwen \
        --split dev --out data/swift/intent/qwen/dev_infer.jsonl
"""
import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))  # for prompt_builders/
from prompt_builders.intent_prompts import build_system_prompt, load_labels, USER_INSTRUCTION


def build_user_content(family):
    if family == "gemma":
        return f"{USER_INSTRUCTION}\n<audio>"
    return f"<audio>\n{USER_INSTRUCTION}"  # qwen (default)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--family", choices=["qwen", "gemma"], required=True)
    ap.add_argument("--split", required=True, help="tag stored in meta (e.g. dev, test)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--with-target", action="store_true",
                    help="append the gold assistant turn (ms-swift SFT format); "
                         "skips rows with no gold intent")
    args = ap.parse_args()

    system_prompt = build_system_prompt(load_labels(args.labels))
    user_content = build_user_content(args.family)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    n = 0
    skipped = 0
    with open(args.manifest, encoding="utf-8") as fh, open(out, "w", encoding="utf-8") as w:
        for row in csv.DictReader(fh):
            gold = (row.get("intent") or "").strip()
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ]
            if args.with_target:
                if not gold:
                    skipped += 1
                    continue
                # target = exactly the single-line JSON the prompt requests, so
                # training and zero-shot outputs stay format-compatible and the
                # same eval_intent.py parser scores both.
                messages.append({"role": "assistant",
                                 "content": json.dumps({"intent": gold}, ensure_ascii=False)})
            rec = {
                "messages": messages,
                "audios": [row["wav"]],
                "meta": {
                    "id": row["ID"],
                    "subtask": "5.1",
                    "family": args.family,
                    "split": args.split,
                    # gold label kept for dev scoring; blank/absent for held-out test
                    "intent": gold,
                },
            }
            w.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n += 1

    kind = "SFT" if args.with_target else "inference"
    print(f"Wrote {n} intent {kind} rows ({args.family}/{args.split}) -> {out}"
          + (f" (skipped {skipped} rows without gold intent)" if skipped else ""))


if __name__ == "__main__":
    main()
