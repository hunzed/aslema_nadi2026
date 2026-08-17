#!/usr/bin/env python3
"""Shared constants/helpers for the Whisper fine-tuning scripts
(train_whisper.py, infer_whisper.py).

Whisper is a plain audio-encoder -> text-decoder ASR model with no chat
template/system-prompt mechanism (unlike the ms-swift Qwen-Omni/gemma-4 path),
so subtask targets are built directly from the SLURP-TN manifest CSV columns
(the same ID/wav/intent/tun_slu_annotation columns to_swift_intent.py and
to_swift_slot_filling.py read) instead of the data/swift/*.jsonl files.
"""
import csv
import json
import re
from pathlib import Path

import librosa
import soundfile as sf

LANGUAGE = "ar"    # Whisper's closest built-in language code; no dedicated Tunisian-dialect token
TASK = "transcribe"


def load_audio(path, target_sr=16000):
    """Read a wav file and resample to target_sr, bypassing datasets' Audio()
    feature - that path decodes via torchcodec, which dlopens FFmpeg shared
    libs (libavutil.so.*) that aren't installed on the cluster. soundfile +
    librosa have no such runtime dependency."""
    audio, sr = sf.read(path, dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != target_sr:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
    return audio

# Joint (both-subtasks-at-once) training needs a signal for which task is being
# asked of a given utterance - Whisper has no text-instruction channel like the
# LLM chat prompts do, so a literal marker is baked into the target text at
# train time (plain text, tokenized like any other target - no special-token
# vocab surgery) and forced as a decoder prefix (forced_decoder_ids) at
# inference time. Both paths tokenize this identically, so there is no
# separate/untested code path between training and inference.
MARKERS = {"intent": "[INTENT] ", "slot_filling": "[SLOT] "}


def build_target(subtask, row, joint=False):
    """subtask: 'intent' | 'slot_filling'. row: a manifest CSV DictReader row.
    Returns None if the row has no gold label for this subtask (caller skips it) -
    mirrors to_swift_{intent,slot_filling}.py's --with-target skip behavior."""
    if subtask == "intent":
        gold = (row.get("intent") or "").strip()
        if not gold:
            return None
        # same single-line JSON the ms-swift prompts request, so predictions
        # from either path are scored by the identical eval_intent.py parser.
        text = json.dumps({"intent": gold}, ensure_ascii=False)
    elif subtask == "slot_filling":
        gold = (row.get("tun_slu_annotation") or "").strip()
        if not gold:
            return None
        text = gold
    else:
        raise ValueError(f"unknown subtask: {subtask}")
    return (MARKERS[subtask] + text) if joint else text


SLOT_LABEL_PATTERN = re.compile(r"<([^<>/\s]+)>")


def collect_slot_labels(train_manifest):
    """Scan train.csv and its sibling validation.csv/test.csv (same manifest
    directory, standard SLURP-TN layout) for slot-label tags (<time>,
    <person>, ...). The SLURP-TN SpeechBrain baseline registers these as
    dedicated tokenizer tokens before training so a tag like <email_address>
    is one token instead of several ordinary BPE fragments; train_whisper.py
    does the same for --subtask slot_filling/joint."""
    manifest_dir = Path(train_manifest).parent
    labels = set()
    for name in ("train.csv", "validation.csv", "test.csv"):
        path = manifest_dir / name
        if not path.exists():
            continue
        with open(path, encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                labels.update(SLOT_LABEL_PATTERN.findall(row.get("tun_slu_annotation") or ""))
    return sorted(labels)
