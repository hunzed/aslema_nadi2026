#!/usr/bin/env python3
"""Build ref_pool.txt from Gemini-ASR WER scores (replaces the SNR heuristic).

Keeps references whose normalized ASR WER <= --max-wer (default 0.15), or
--top N by WER if you prefer a fixed size. The two ear-verified good refs
(prod2/prod8) are always included.
"""
import os
import argparse
import json
from pathlib import Path

NADI = Path(os.environ["PROJ"])
KNOWN_GOOD = {"slurptn_train_002621", "slurptn_train_000349"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-wer", type=float, default=0.15)
    ap.add_argument("--top", type=int, default=0,
                    help="if >0: take best N by WER instead of thresholding")
    args = ap.parse_args()

    scores = {}
    for l in open(NADI / "data" / "augmentation" / "tts" / "ref_asr_scores.jsonl", encoding="utf-8"):
        r = json.loads(l)
        if r.get("wer") is not None:
            scores[r["id"]] = r          # last write wins on resume dups

    ranked = sorted(scores.values(), key=lambda r: r["wer"])
    if args.top:
        pool = ranked[:args.top]
    else:
        pool = [r for r in ranked if r["wer"] <= args.max_wer]
    ids = {r["id"] for r in pool}
    for r in ranked:
        if r["id"] in KNOWN_GOOD and r["id"] not in ids:
            pool.append(r)

    out = NADI / "data" / "augmentation" / "tts" / "ref_pool.txt"
    with open(out, "w", encoding="utf-8") as f:
        for r in pool:
            f.write(f"{r['wav']}\t{r['gold']}\n")
    import statistics
    wers = [r["wer"] for r in ranked]
    print(f"scored {len(ranked)} | pool {len(pool)} "
          f"(criterion: {'top ' + str(args.top) if args.top else 'wer<=' + str(args.max_wer)})")
    print(f"WER distribution: min {min(wers):.2f} median "
          f"{statistics.median(wers):.2f} mean {statistics.mean(wers):.2f}")
    print(f"known-good WERs: "
          f"{ {k: scores[k]['wer'] for k in KNOWN_GOOD if k in scores} }")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
