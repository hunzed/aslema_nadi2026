#!/usr/bin/env python3
"""Gemini-ASR WER scorer - generic: score any (id, wav, gold_text) set.

Transcribes each clip with the best available Gemini model (default
gemini-3.1-pro-preview), applies the project's Arabic ASR normalization to
BOTH hypothesis and gold, and records per-item WER. Used for:
  1. reference selection  (score the train clips; keep the clearest)
  2. synthetic-audio filtering later (score VoxCPM output vs synth gold)

Normalization (spec by user, 2026-07-26):
  - remove Arabic diacritics
  - keep Arabic letters, Latin letters, digits, @, %, whitespace
  - keep . , only between two digits; keep ' only between two non-spaces
  - all other chars -> space
  - Alef variants -> bare Alef
  - collapse whitespace, lowercase Latin, strip

Usage:
  python gemini_asr_wer.py --train-refs              # score train candidates
  python gemini_asr_wer.py --manifest x.csv --out y.jsonl   # any manifest
Resume-safe (appends, skips scored ids). WORKERS env for parallelism.
"""
import argparse
import csv
import json
import os
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

NADI = Path(os.environ["PROJ"])
sys.path.insert(0, str(NADI / "augmentation"))
from gemini_client import client, generate_json  # noqa: E402  (client reuse)

ASR_MODEL = os.environ.get("ASR_MODEL", "gemini-3.1-pro-preview")
DIACRITICS = re.compile(r"[ً-ْٰ]")
ALEF = re.compile(r"[أإآٱ]")
ARABIC = r"ء-ي٠-٩پچڤگ"

PROMPT = ("Transcribe this Tunisian Arabic (Derja) audio VERBATIM. Arabic "
          "script for Arabic words; keep French/English code-switched words "
          "in Latin script exactly as spoken. No diacritics, no punctuation, "
          "no explanations - output ONLY the raw transcription text.")


def normalize(s):
    s = DIACRITICS.sub("", s)
    s = ALEF.sub("ا", s)
    out = []
    for i, ch in enumerate(s):
        if re.match(rf"[{ARABIC}a-zA-Z0-9@%\s]", ch):
            out.append(ch)
        elif ch in ".," and i > 0 and i + 1 < len(s) \
                and s[i - 1].isdigit() and s[i + 1].isdigit():
            out.append(ch)
        elif ch == "'" and i > 0 and i + 1 < len(s) \
                and not s[i - 1].isspace() and not s[i + 1].isspace():
            out.append(ch)
        else:
            out.append(" ")
    return re.sub(r"\s+", " ", "".join(out)).lower().strip()


def wer(ref, hyp):
    r, h = ref.split(), hyp.split()
    d = list(range(len(h) + 1))
    for i in range(1, len(r) + 1):
        prev, d[0] = d[0], i
        for j in range(1, len(h) + 1):
            cur = min(d[j] + 1, d[j - 1] + 1, prev + (r[i - 1] != h[j - 1]))
            prev, d[j] = d[j], cur
    return d[len(h)] / max(1, len(r))


def transcribe(wav_path):
    from google.genai import types
    audio = Path(wav_path).read_bytes()
    resp = client().models.generate_content(
        model=ASR_MODEL,
        contents=[PROMPT, types.Part.from_bytes(data=audio,
                                                mime_type="audio/wav")],
        config=types.GenerateContentConfig(temperature=0),
    )
    return (resp.text or "").strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-refs", action="store_true",
                    help="score the 2.5-10s train candidates for ref selection")
    ap.add_argument("--manifest", help="csv with ID,wav,tun_transcription cols")
    ap.add_argument("--out", help="output scores jsonl")
    ap.add_argument("--workers", type=int,
                    default=int(os.environ.get("WORKERS", 16)))
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    if args.train_refs:
        args.manifest = str(NADI / "data/slurptn/data/manifests/train.csv")
        args.out = str(NADI / "augmentation/tts/ref_asr_scores.jsonl")

    items = []
    with open(args.manifest, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            wav = r.get("wav")
            if not wav or not Path(wav).exists():
                continue
            if args.train_refs:
                try:
                    if not (2.5 <= float(r["duration"]) <= 10.0):
                        continue
                except (KeyError, ValueError):
                    pass
            items.append({"id": r["ID"], "wav": wav,
                          "gold": r["tun_transcription"].strip()})
    done = set()
    out = Path(args.out)
    if out.exists():
        for l in open(out, encoding="utf-8"):
            done.add(json.loads(l)["id"])
    todo = [it for it in items if it["id"] not in done]
    if args.limit:
        todo = todo[:args.limit]
    print(f"model={ASR_MODEL} workers={args.workers} | "
          f"items={len(items)} todo={len(todo)}")

    def score(it):
        import time
        for attempt in range(4):
            try:
                hyp = transcribe(it["wav"])
                gold_n, hyp_n = normalize(it["gold"]), normalize(hyp)
                return {**it, "hyp": hyp,
                        "wer": round(wer(gold_n, hyp_n), 4)}
            except Exception as e:
                time.sleep(min(2 ** attempt * 2, 30))
                err = str(e)
        return {**it, "hyp": None, "wer": None, "error": err[:200]}

    n = 0
    with open(out, "a", encoding="utf-8") as f, \
            ThreadPoolExecutor(max_workers=args.workers) as ex:
        for res in ex.map(score, todo):
            f.write(json.dumps(res, ensure_ascii=False) + "\n")
            f.flush()
            n += 1
            if n % 50 == 0 or n == len(todo):
                print(f"  {n}/{len(todo)}")
    print(f"done -> {out}")


if __name__ == "__main__":
    main()
