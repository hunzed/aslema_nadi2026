#!/usr/bin/env python3
"""VoxCPM2 voice-cloned synthesis of the judged synthetic text.

For every {id, text} in augmentation/data/tts_queue.jsonl:
  1. pick a real SLURP-TN TRAIN utterance as the cloning prompt -
     deterministic per id (stable across resumes), duration-filtered 2.5-10 s,
     so the ~real speakers' voices/channel conditions are inherited
     ("ultimate cloning": prompt_wav_path + prompt_text = its Tunisian
     transcript);
  2. model.generate(...) -> 48 kHz float array;
  3. resample to 16 kHz mono (the SLU pipeline's format) ->
     augmentation/data/wav/<id>.wav

Resume-safe (skips existing wavs). Knobs (env or CLI):
  CFG_VALUE=2.0          guidance; higher = closer to text, less natural
  STEPS=16               CFM steps (default 10 upstream; 16 = quality tilt)
  NORMALIZE=0            VoxCPM text normalization OFF by default here -
                         Derja + French/English code-switching is exactly the
                         kind of text a normalizer mangles. A/B it if unsure.
  STYLE_HINT=""          optional "(...)" voice-design prefix - NOT
                         recommended together with prompt cloning
Usage:
  python synthesize.py --limit 5          # smoke: 5 items
  python synthesize.py                    # everything
"""
import argparse
import csv
import hashlib
import json
import os
import sys
from pathlib import Path

NADI = Path(os.environ["PROJ"])
DATA = NADI / "augmentation/data"


def load_refs(min_s=2.5, max_s=10.0):
    # curated pool (curate_refs.py output) wins when present: random
    # references gave only 2/8 acceptable clips in the listening test -
    # cloning inherits reference noise/mumble, so quality-score the pool.
    pool_file = NADI / "augmentation/tts/ref_pool.txt"
    if pool_file.exists():
        refs = []
        for line in open(pool_file, encoding="utf-8"):
            wav, text = line.rstrip("\n").split("\t", 1)
            if Path(wav).exists():
                refs.append((wav, text))
        if refs:
            print(f"using CURATED reference pool: {len(refs)} refs")
            return refs
    refs = []
    with open(NADI / "data/slurptn/data/manifests/train.csv", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            try:
                d = float(r["duration"])
            except ValueError:
                continue
            if min_s <= d <= max_s and Path(r["wav"]).exists():
                refs.append((r["wav"], r["tun_transcription"].strip()))
    if not refs:
        sys.exit("FATAL: no usable reference wavs found in the train manifest")
    return refs


def pick_ref(item_id, refs):
    h = int(hashlib.sha1(item_id.encode()).hexdigest(), 16)
    return refs[h % len(refs)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    # upstream defaults, unforced (user call 2026-07-26): cfg 2.0, 10 steps,
    # normalize on, no style injection. NORMALIZE=0 exists as an escape hatch
    # if the samples reveal the normalizer mangling code-switched Derja.
    ap.add_argument("--cfg", type=float,
                    default=float(os.environ.get("CFG_VALUE", 2.0)))
    ap.add_argument("--steps", type=int,
                    default=int(os.environ.get("STEPS", 10)))
    ap.add_argument("--normalize", action="store_true",
                    default=os.environ.get("NORMALIZE", "1") == "1")
    ap.add_argument("--style-hint", default=os.environ.get("STYLE_HINT", ""))
    args = ap.parse_args()

    # QUEUE selects the work list (default = the original synthetic queue);
    # the en2tn run passes QUEUE=tts_queue_en2tn.jsonl WAV_DIR=wav_en2tn so it
    # can never touch the 24k-clip main run's outputs.
    queue = DATA / os.environ.get("QUEUE", "tts_queue.jsonl")
    items = [json.loads(l) for l in open(queue, encoding="utf-8")]
    print(f"queue: {queue}")
    # sharding for multi-GPU runs: SHARD=0 NSHARDS=2 -> even items, etc.
    shard = int(os.environ.get("SHARD", 0))
    nshards = int(os.environ.get("NSHARDS", 1))
    if nshards > 1:
        items = items[shard::nshards]
        print(f"shard {shard}/{nshards}: {len(items)} items")
    # WAV_DIR separates model variants (wav = base, wav_ft = LoRA)
    out_dir = DATA / os.environ.get("WAV_DIR", "wav")
    out_dir.mkdir(parents=True, exist_ok=True)
    todo = [it for it in items if not (out_dir / f"{it['id']}.wav").exists()]
    if args.limit:
        todo = todo[:args.limit]
    print(f"items={len(items)} todo={len(todo)} | cfg={args.cfg} "
          f"steps={args.steps} normalize={args.normalize}")
    if not todo:
        return

    refs = load_refs()
    print(f"reference pool: {len(refs)} train utterances (2.5-10 s)")

    import librosa
    import numpy as np
    import soundfile as sf
    from voxcpm import VoxCPM
    # load_denoiser only matters for noisy reference audio; SLURP-TN refs are
    # dataset-quality, so skip it for a faster load (DENOISE=1 re-enables)
    kw = {}
    if os.environ.get("LORA", "0") == "1":   # use the Derja fine-tune adapter
        from voxcpm.model.voxcpm import LoRAConfig
        ckpt = NADI / "augmentation/tts/finetune/checkpoints/slurptn_lora/latest"
        cfg = json.loads((ckpt / "lora_config.json").read_text())["lora_config"]
        kw = dict(lora_config=LoRAConfig(**cfg), lora_weights_path=str(ckpt))
        print(f"using LoRA adapter: {ckpt}")
    model = VoxCPM.from_pretrained(
        "openbmb/VoxCPM2",
        load_denoiser=os.environ.get("DENOISE", "0") == "1", **kw)
    sr_out = model.tts_model.sample_rate

    ok = err = 0
    for i, it in enumerate(todo):
        ref_wav, ref_text = pick_ref(it["id"], refs)
        text = (args.style_hint + it["text"]) if args.style_hint else it["text"]
        try:
            # silence-retry guard: VoxCPM occasionally emits a near-empty
            # waveform (seen once in the A/B samples: 7.7s at RMS 0.0005).
            # Retry with a fresh seed, then a different reference.
            wav16 = None
            for attempt in range(3):
                gen_kw = dict(
                    text=text,
                    prompt_wav_path=ref_wav, prompt_text=ref_text,
                    reference_wav_path=ref_wav,
                    cfg_value=args.cfg, inference_timesteps=args.steps,
                    normalize=args.normalize)
                # NB: generate() has no seed kwarg in this voxcpm version;
                # CFM sampling is stochastic per call, so a plain re-call is
                # already a fresh draw.
                if attempt == 2:   # last resort: different reference
                    h2 = int(hashlib.sha1(
                        (it["id"] + "#retry").encode()).hexdigest(), 16)
                    ref_wav, ref_text = refs[h2 % len(refs)]
                    gen_kw.update(prompt_wav_path=ref_wav,
                                  prompt_text=ref_text,
                                  reference_wav_path=ref_wav)
                wav = np.asarray(model.generate(**gen_kw),
                                 dtype="float32").squeeze()
                w16 = librosa.resample(wav, orig_sr=sr_out, target_sr=16000)
                rms = float(np.sqrt((w16 ** 2).mean() + 1e-12))
                if rms >= 0.01 and len(w16) >= 0.4 * 16000:
                    wav16 = w16
                    break
                print(f"  [{it['id']}] silent/short output "
                      f"(rms={rms:.4f}), retry {attempt + 1}")
            if wav16 is None:
                raise RuntimeError("silent output after 3 attempts")
            sf.write(out_dir / f"{it['id']}.wav", wav16, 16000, subtype="PCM_16")
            ok += 1
        except Exception as e:
            err += 1
            print(f"  [{it['id']}] FAILED: {str(e)[:120]}")
        if (i + 1) % 25 == 0 or i + 1 == len(todo):
            print(f"  {i + 1}/{len(todo)} (ok {ok}, err {err})")
    print(f"done: {ok} synthesized, {err} failed -> {out_dir}")


if __name__ == "__main__":
    main()
