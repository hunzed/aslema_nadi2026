#!/usr/bin/env python3
"""Stage 5 - assemble final manifests + swift SFT jsonls.

Takes the surviving synthetic items (final_keep_ids.txt if the acoustic
filter ran, else everything in gen_kept.jsonl that has a wav), fills real
durations from the wav headers, and emits:

  augmentation/data/manifests/synth.csv                     synthetic only
  augmentation/data/manifests/train_plus_synth.csv          real train + synth
  augmentation/data/manifests/train_dev_plus_synth.csv      + dev (FINAL submission runs only)
  augmentation/data/swift/<subtask>/qwen/{synth,mix,mix_dev}_sft.jsonl   via the
      project's own to_swift converters (--with-target), so prompts/format
      are byte-identical to every previous training run.

--max-ratio caps synthetic at N x real train size (default 2.0) by uniform
per-intent downsampling of the surplus (rare intents are never downsampled).
"""
import os
import argparse
import csv
import json
import random
import struct
import subprocess
from collections import Counter, defaultdict
from pathlib import Path

DATA = Path(os.environ.get("AUG_DATA",
                          Path(os.environ["PROJ"]) / "data" / "augmentation"))
PROJ = Path(os.environ["PROJ"])
PY = PROJ / "conda/envs/nadi_task5/bin/python"
FIELDS = ["ID", "duration", "wav", "tun_transcription",
          "tun_slu_annotation", "intent"]


def wav_duration(path):
    try:
        with open(path, "rb") as f:
            hdr = f.read(44)
        if hdr[:4] != b"RIFF":
            return 0.0
        rate = struct.unpack("<I", hdr[24:28])[0]
        bytes_per_sec = struct.unpack("<I", hdr[28:32])[0] or rate * 2
        size = Path(path).stat().st_size - 44
        return round(size / bytes_per_sec, 3)
    except Exception:
        return 0.0


def read_manifest(path):
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_manifest(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    print(f"  {path}  ({len(rows)} rows)")


def to_swift(manifest, subtask, tag):
    script = ("to_swift_intent.py" if subtask == "intent"
              else "to_swift_slot_filling.py")
    labels = ("intent_labels.txt" if subtask == "intent" else "slot_labels.txt")
    out = DATA / "swift" / subtask / "qwen" / f"{tag}_sft.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [str(PY), str(PROJ / "data_prep" / script),
         "--manifest", str(manifest),
         "--labels", str(PROJ / "data/slurptn" / labels),
         "--family", "qwen", "--split", tag, "--out", str(out),
         "--with-target"], check=True)
    print(f"  swift: {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-ratio", type=float, default=2.0,
                    help="cap synth at this multiple of real train rows")
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--skip-missing-wav", action="store_true", default=True)
    args = ap.parse_args()
    rng = random.Random(args.seed)

    items = [json.loads(l) for l in open(DATA / "gen_kept.jsonl", encoding="utf-8")]

    # one row per AUDIO RENDITION: base run -> wav/<id>.wav (ID=<id>),
    # LoRA run -> wav_ft/<id>.wav (ID=<id>_ft). Same text/labels twice =
    # doubled acoustic diversity per utterance.
    def audio_ok(path, n_chars):
        """Code-level sanity gate, thresholds calibrated on a 3k-sample
        analysis of the actual output distribution (2026-07-26):
          healthy: activity p5=0.17 p50=0.43 | sec/char p50=0.107 p99=0.30
          broken:  activity <=0.05 ("long but silent") or sec/char >=0.47
                   ("babble/looping, way too long for the text")
        """
        try:
            import numpy as np
            import wave
            with wave.open(str(path)) as w:
                sr, n = w.getframerate(), w.getnframes()
                x = np.frombuffer(w.readframes(n),
                                  dtype="<i2").astype("float32") / 32768
            dur = n / sr
            rms = float(np.sqrt((x ** 2).mean() + 1e-12))
            clip = float((np.abs(x) > 0.999).mean())
            fr = max(1, int(0.025 * sr))
            m = len(x) // fr
            act = float((np.sqrt((x[:m * fr].reshape(m, fr) ** 2)
                                 .mean(1) + 1e-12) > 0.02).mean())
            spc = dur / max(1, n_chars)
            return (0.4 <= dur <= 30 and rms >= 0.01 and clip < 0.05
                    and act >= 0.10 and 0.04 <= spc <= 0.40)
        except Exception:
            return False

    rows, dropped = [], 0
    for it in items:
        for wdir, suffix in ((DATA / "wav", ""), (DATA / "wav_ft", "_ft")):
            wav = wdir / f"{it['id']}.wav"
            if not wav.exists():
                continue
            if not audio_ok(wav, len(it["text"])):
                dropped += 1
                continue
            rows.append({"ID": it["id"] + suffix,
                         "duration": wav_duration(wav),
                         "wav": str(wav),
                         "tun_transcription": it["text"],
                         "tun_slu_annotation": it["annotated"],
                         "intent": it["intent"]})

    print(f"audio sanity gate dropped {dropped} wavs "
          f"(silent/short/clipped); {len(rows)} renditions kept")

    # audio-level keep list (from the Gemini-ASR output filter); ids may be
    # plain or _ft-suffixed
    keep_file = DATA / "final_keep_ids.txt"
    if keep_file.exists():
        keep = set(keep_file.read_text().split())
        before = len(rows)
        rows = [r for r in rows if r["ID"] in keep]
        print(f"acoustic-filtered rows: {before} -> {len(rows)}")

    train = read_manifest(PROJ / "data/slurptn/data/manifests/train.csv")
    dev = read_manifest(PROJ / "data/slurptn/data/manifests/validation.csv")

    # ---- ratio cap with rare-intent immunity -------------------------------
    cap = int(len(train) * args.max_ratio)
    if len(rows) > cap:
        tc = Counter(r["intent"] for r in train)
        rare = {i for i, c in tc.items() if c < 10} | \
               {r["intent"] for r in rows if tc.get(r["intent"], 0) == 0}
        keep_rows = [r for r in rows if r["intent"] in rare]
        rest = [r for r in rows if r["intent"] not in rare]
        rng.shuffle(rest)
        keep_rows += rest[:max(0, cap - len(keep_rows))]
        print(f"ratio cap {args.max_ratio}x: {len(rows)} -> {len(keep_rows)} "
              f"(rare intents kept whole)")
        rows = keep_rows

    print("synthetic per intent (lowest 8):",
          dict(sorted(Counter(r['intent'] for r in rows).items(),
                      key=lambda kv: kv[1])[:8]))

    mdir = DATA / "manifests"
    write_manifest(mdir / "synth.csv", rows)
    mixed = train + rows
    rng.shuffle(mixed)
    write_manifest(mdir / "train_plus_synth.csv", mixed)
    mixed_dev = train + dev + rows
    rng.shuffle(mixed_dev)
    write_manifest(mdir / "train_dev_plus_synth.csv", mixed_dev)

    for sub in ("intent", "slot_filling"):
        to_swift(mdir / "synth.csv", sub, "synth")
        to_swift(mdir / "train_plus_synth.csv", sub, "mix")
        to_swift(mdir / "train_dev_plus_synth.csv", sub, "mix_dev")

    print("""
Training handoff (shadow tree, proven recipe - example for the slot winner):
  sbatch --mem=200GB --export=ALL,CUSTOM_MODEL=stock_qwen3omni30b,SUBTASK=slot_filling,\\
TRAIN_JSONL=augmentation/data/swift/slot_filling/qwen/mix_sft.jsonl,MODEL_NAME_SUFFIX=_aug \\
    custom_gcp/scripts/sft_custom.sh        # (see augmentation/README.md - TRAIN_JSONL
                                            #  support requires the one-line driver
                                            #  override documented there)
""")


if __name__ == "__main__":
    main()
