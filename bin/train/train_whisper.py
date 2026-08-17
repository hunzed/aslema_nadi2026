#!/usr/bin/env python3
"""Fine-tune openai/whisper-small on SLURP-TN for NADI-2026 Subtask 5.1
(Intent) and/or 5.2 (Slot Filling) - FULL fine-tune via HF `transformers`
Seq2SeqTrainer (user-confirmed choice; whisper-small is ~244M params, cheap
enough that LoRA buys little and Whisper+peft is a rougher-edged combo).

Not an ms-swift model: Whisper has no chat template, so it can't go through
`swift sft` like the Qwen-Omni/gemma-4 scripts under scripts/training/{qwen,
gemma}/ - this is its own training path, sourced from the SLURP-TN manifest
CSV directly (same ID/wav/intent/tun_slu_annotation columns to_swift_*.py
reads) rather than the data/swift/*.jsonl chat-formatted files.

Three modes (--subtask):
  * intent       : target = '{"intent": "<label>"}'  (same string the
                   ms-swift zero-shot/SFT prompts request, so eval_intent.py
                   parses it unchanged)
  * slot_filling : target = the gold tun_slu_annotation verbatim
  * joint        : both subtasks at once - each utterance appears TWICE (once
                   per subtask, same audio) with a literal "[INTENT] "/
                   "[SLOT] " prefix baked into the target text (see
                   whisper_common.py) so the model has a signal for which task
                   is being asked. infer_whisper.py --joint forces the
                   matching prefix via forced_decoder_ids at generation time.
                   HIGHEST-RISK part of this pair of scripts - the marker
                   mechanism is built from standard/documented Whisper APIs
                   (plain text label, forced_decoder_ids) but is not something
                   this project has run before; sanity-check with a short
                   (few-hundred-step) run and a manual generate() call before
                   committing to a full training run.

Usage:
    python bin/train/train_whisper.py --subtask intent \\
        --manifest $PROJ/data/slurptn/data/manifests/train.csv \\
        --output_dir $PROJ/models/whisper/whisper_small_intent
"""
import argparse
import csv
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Union

import torch
from datasets import Dataset
from transformers import (
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    WhisperForConditionalGeneration,
    WhisperProcessor,
)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from whisper_common import LANGUAGE, TASK, build_target, collect_slot_labels, load_audio


@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    """Standard HF Whisper fine-tuning collator: pads input_features and
    labels separately (they have unrelated shapes), masks label padding with
    -100, and strips a duplicate leading BOS (the model re-adds one itself via
    decoder_start_token_id when it shifts `labels` into `decoder_input_ids`)."""
    processor: Any

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        input_features = [{"input_features": f["input_features"]} for f in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        label_features = [{"input_ids": f["labels"]} for f in features]
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels
        return batch


def load_rows(manifest):
    with open(manifest, encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def build_dataset(manifest, subtask):
    rows = load_rows(manifest)
    joint = subtask == "joint"
    subtasks = ["intent", "slot_filling"] if joint else [subtask]
    ids, paths, texts = [], [], []
    skipped = 0
    for row in rows:
        for st in subtasks:
            target = build_target(st, row, joint=joint)
            if target is None:
                skipped += 1
                continue
            ids.append(row["ID"])
            paths.append(row["wav"])
            texts.append(target)
    print(f"Built {len(ids)} examples from {manifest} (subtask={subtask}, "
          f"skipped {skipped} rows without gold label)")
    return Dataset.from_dict({"id": ids, "audio": paths, "text": texts})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subtask", required=True, choices=["intent", "slot_filling", "joint"])
    ap.add_argument("--manifest", required=True,
                     help="train manifest CSV, e.g. $PROJ/data/slurptn/data/manifests/train.csv")
    ap.add_argument("--model_name_or_path", default="openai/whisper-small")
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--num_train_epochs", type=float, default=2.0)
    ap.add_argument("--learning_rate", type=float, default=1e-5)
    ap.add_argument("--per_device_train_batch_size", type=int, default=16)
    ap.add_argument("--per_device_eval_batch_size", type=int, default=16)
    ap.add_argument("--gradient_accumulation_steps", type=int, default=1)
    ap.add_argument("--warmup_ratio", type=float, default=0.05)
    ap.add_argument("--eval_steps", type=int, default=200)
    ap.add_argument("--save_steps", type=int, default=200)
    ap.add_argument("--logging_steps", type=int, default=25)
    ap.add_argument("--max_label_length", type=int, default=225,
                     help="drop training examples whose tokenized target exceeds this length")
    ap.add_argument("--eval_ratio", type=float, default=0.01,
                     help="held-out slice of train for eval loss only, mirrors the ms-swift "
                          "driver's --split_dataset_ratio (not the real scored dev set)")
    ap.add_argument("--dataset_num_proc", type=int, default=1,
                     help="audio decoding under multiprocessing can be flaky on some setups; "
                          "default 1 is the safe choice, raise if your env handles it fine")
    ap.add_argument("--resume_from_checkpoint", default=None)
    args = ap.parse_args()

    outdir = Path(args.output_dir)
    if args.resume_from_checkpoint is None and any(outdir.glob("checkpoint-*")):
        print(f"Checkpoints already exist under {outdir} - skipping (resume-skip convention).")
        print(f"To resume: --resume_from_checkpoint {outdir}/checkpoint-XXXX")
        print(f"To retrain: delete/move {outdir} first.")
        return
    outdir.mkdir(parents=True, exist_ok=True)

    processor = WhisperProcessor.from_pretrained(args.model_name_or_path, language=LANGUAGE, task=TASK)

    # Register slot-label tags (<time>, <person>, ...) as whole tokens before
    # any tokenization happens below, so they can't be split into ordinary
    # BPE fragments - see whisper_common.collect_slot_labels docstring.
    slot_labels = []
    if args.subtask in ("slot_filling", "joint"):
        slot_labels = collect_slot_labels(args.manifest)
        added = processor.tokenizer.add_tokens(slot_labels)
        print(f"Added {added} slot-label tokens to the tokenizer: {slot_labels}")

    full = build_dataset(args.manifest, args.subtask)
    split = full.train_test_split(test_size=args.eval_ratio, seed=42)
    train_ds, eval_ds = split["train"], split["test"]

    def prepare(batch):
        array = load_audio(batch["audio"])
        batch["input_features"] = processor.feature_extractor(
            array, sampling_rate=16000
        ).input_features[0]
        batch["labels"] = processor.tokenizer(batch["text"]).input_ids
        return batch

    # Map BEFORE the model is loaded: datasets.map() always forks a worker
    # process internally (even at num_proc=1), and forking after a full torch
    # model is loaded can leave the child inheriting a locked MKL/OpenMP mutex
    # from the parent's threadpool with no thread left to release it - a
    # silent hang, not a crash. The feature extractor/tokenizer alone don't
    # trigger that threadpool init, so mapping first sidesteps it.
    train_ds = train_ds.map(prepare, remove_columns=train_ds.column_names, num_proc=args.dataset_num_proc)
    eval_ds = eval_ds.map(prepare, remove_columns=eval_ds.column_names, num_proc=args.dataset_num_proc)
    train_ds = train_ds.filter(lambda x: len(x["labels"]) <= args.max_label_length)
    eval_ds = eval_ds.filter(lambda x: len(x["labels"]) <= args.max_label_length)

    model = WhisperForConditionalGeneration.from_pretrained(args.model_name_or_path)
    if slot_labels:
        model.resize_token_embeddings(len(processor.tokenizer))
    model.generation_config.language = LANGUAGE
    model.generation_config.task = TASK
    model.generation_config.forced_decoder_ids = None  # superseded by language/task above
    # Targets here are JSON intent labels / <slot>-tagged text, not plain speech -
    # clear Whisper's default "non-speech" punctuation blocklist (blocks { " : < >
    # among others) so the saved checkpoint's generation_config doesn't carry a
    # suppress list that makes those symbols impossible to generate later.
    model.generation_config.suppress_tokens = []
    model.config.use_cache = False  # required alongside gradient_checkpointing

    collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)

    training_args = Seq2SeqTrainingArguments(
        output_dir=str(outdir),
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        num_train_epochs=args.num_train_epochs,
        gradient_checkpointing=True,
        bf16=torch.cuda.is_available(),
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_steps=args.save_steps,
        save_total_limit=2,
        logging_steps=args.logging_steps,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        predict_with_generate=False,  # loss-only eval during training (matches the ms-swift driver's convention)
        report_to=[],
        dataloader_num_workers=4,
    )

    trainer = Seq2SeqTrainer(
        args=training_args,
        model=model,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=collator,
        processing_class=processor,
    )

    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)

    # Full fine-tune (no adapter to merge) - save the best model straight into
    # output_dir root so infer_whisper.py can point at it directly.
    # load_best_model_at_end=True already swapped trainer.model to the best
    # checkpoint by eval_loss, so this save is of that checkpoint, not the
    # last step. processor.save_pretrained persists the resized tokenizer
    # (slot-label tokens added above) alongside it.
    trainer.save_model(str(outdir))
    processor.save_pretrained(str(outdir))
    print(f"Saved fine-tuned model + processor to {outdir}")


if __name__ == "__main__":
    main()
