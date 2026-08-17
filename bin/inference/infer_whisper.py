#!/usr/bin/env python3
"""Generate predictions from a fine-tuned Whisper checkpoint and write them in
the same `{"response": ..., "meta": {...}}` shape ms-swift's `swift infer`
result files use - so eval_intent.py / eval_slot_filling.py score them with
NO changes.

Reads directly from the SLURP-TN manifest CSV (same one to_swift_*.py reads),
not the data/swift/*.jsonl files - Whisper doesn't consume the chat-formatted
prompts those hold, only audio path + gold label/annotation for scoring.

Usage:
    python bin/train/infer_whisper.py --model_dir $PROJ/models/whisper/whisper_small_intent \\
        --manifest $PROJ/data/slurptn/data/manifests/validation.csv --subtask intent \\
        --split dev --out $PROJ/outputs/inference/intent/whisper/whisper_small_dev.jsonl

    # joint model: force the matching "[INTENT] "/"[SLOT] " decoder prefix (see
    # whisper_common.py) so one fine-tuned checkpoint can serve either subtask -
    # run this twice, once per subtask, same as any other model in this project.
    python bin/train/infer_whisper.py --model_dir $PROJ/models/whisper/whisper_small_joint \\
        --manifest $PROJ/data/slurptn/data/manifests/validation.csv --subtask slot_filling \\
        --split dev --joint --out $PROJ/outputs/inference/slot_filling/whisper/whisper_small_joint_dev.jsonl
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import torch
from transformers import WhisperForConditionalGeneration, WhisperProcessor

sys.path.insert(0, str(Path(__file__).resolve().parent))
from whisper_common import LANGUAGE, MARKERS, TASK, load_audio


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--subtask", required=True, choices=["intent", "slot_filling"])
    ap.add_argument("--split", required=True, help="tag stored in meta (e.g. dev, test)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--joint", action="store_true",
                     help="model was trained by train_whisper.py --subtask joint - force the "
                          "matching [INTENT]/[SLOT] prefix via forced_decoder_ids")
    ap.add_argument("--max_new_tokens", type=int, default=None,
                     help="default 64 for intent, 256 for slot_filling (same as the ms-swift scripts)")
    ap.add_argument("--num_beams", type=int, default=8,
                     help="beam search width (default 8, matches the working SLURP-TN Whisper "
                          "baseline recipe's test_beam_size; use 1 for greedy)")
    ap.add_argument("--batch_size", type=int, default=8)
    args = ap.parse_args()

    max_new_tokens = args.max_new_tokens or (64 if args.subtask == "intent" else 256)

    out_path = Path(args.out)
    if out_path.exists():
        print(f"Results already exist at {out_path} - skipping (resume-skip convention).")
        return
    out_path.parent.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = WhisperProcessor.from_pretrained(args.model_dir, language=LANGUAGE, task=TASK)
    model = WhisperForConditionalGeneration.from_pretrained(args.model_dir).to(device)
    model.eval()
    # Whisper's default suppress_tokens blocks "non-speech" punctuation (incl.
    # { " : < >) so plain ASR won't hallucinate symbols - but our subtask targets
    # ARE built from those symbols (JSON intent labels, <slot> tags). Left in
    # place, generation can never produce a valid label/tag no matter how well
    # the model was trained.
    model.generation_config.suppress_tokens = []

    # `forced_decoder_ids` is no longer read by generate() as of this project's
    # installed transformers (5.12.1) - it's not even in the generate() signature,
    # so passing it is silently absorbed and ignored, and the [INTENT]/[SLOT]
    # marker was never actually being forced (the joint model was free-generating
    # and defaulting to whichever marker it prefers regardless of --subtask).
    # decoder_input_ids is the supported replacement: build the full initial
    # decoder sequence (bos + lang/task/notimestamps + marker) ourselves and hand
    # it to generate() as the literal prompt to continue from.
    prompt_ids = [model.generation_config.decoder_start_token_id] + \
        [tok for _, tok in processor.get_decoder_prompt_ids(language=LANGUAGE, task=TASK)]
    if args.joint:
        marker = MARKERS[args.subtask]
        marker_ids = processor.tokenizer(marker, add_special_tokens=False).input_ids
        prompt_ids = prompt_ids + marker_ids

    with open(args.manifest, encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))

    results = []
    for i in range(0, len(rows), args.batch_size):
        batch = rows[i:i + args.batch_size]
        audios = [load_audio(r["wav"]) for r in batch]
        inputs = processor.feature_extractor(audios, sampling_rate=16000, return_tensors="pt")
        input_features = inputs.input_features.to(device)
        decoder_input_ids = torch.tensor([prompt_ids] * len(batch), dtype=torch.long, device=device)
        with torch.no_grad():
            generated = model.generate(
                input_features,
                decoder_input_ids=decoder_input_ids,
                max_new_tokens=max_new_tokens,
                num_beams=args.num_beams,
            )
        texts = processor.tokenizer.batch_decode(generated, skip_special_tokens=True)
        for row, text in zip(batch, texts):
            text = text.strip()
            if args.joint and text.startswith(MARKERS[args.subtask].strip()):
                text = text[len(MARKERS[args.subtask]):].strip()
            meta = {
                "id": row["ID"],
                "subtask": "5.1" if args.subtask == "intent" else "5.2",
                "family": "whisper",
                "split": args.split,
            }
            if args.subtask == "intent":
                meta["intent"] = (row.get("intent") or "").strip()
            else:
                meta["ref"] = (row.get("tun_slu_annotation") or "").strip()
            results.append({"response": text, "meta": meta})
        print(f"  {min(i + args.batch_size, len(rows))}/{len(rows)}", end="\r")

    with open(out_path, "w", encoding="utf-8") as w:
        for r in results:
            w.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"\nWrote {len(results)} rows -> {out_path}")


if __name__ == "__main__":
    main()
