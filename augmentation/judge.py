#!/usr/bin/env python3
"""Stage 3 - LLM judge panel (second, independent model vote).

The generator (GEN_MODEL, flash-class) produced the items; a STRONGER,
different model (JUDGE_MODEL, pro-class) now audits them in batches of 10 on
four axes and may REPAIR minor annotation slips:

  derja        1-5  authentic spoken Tunisian Derja? (MSA/other dialect -> low)
  natural      1-5  plausible voice-assistant utterance?
  intent_match bool text really expresses the claimed intent?
  slots_ok     bool every slot span/label correct AND no missed spans?
  fixed_annotated  optional corrected annotation when the only problem is
                   markup (re-validated downstream before acceptance)

keep-rule: derja>=4 and natural>=3 and intent_match and slots_ok(-or-fixed).
--votes 2 runs two independent panels (different sampling temps) and requires
unanimity - use for the final high-stakes run.

Outputs: gen_judged.jsonl (all items + verdicts), gen_kept.jsonl (survivors,
with fixes applied), tts_queue.jsonl ({id, text} for VoxCPM voice cloning).
Resume-safe via judged-id tracking.
"""
import argparse
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from gemini_client import JUDGE_MODEL, JUDGE_MODELS, generate_json  # noqa: E402
from validate import parse_annotation, norm                         # noqa: E402

DATA = Path(__file__).resolve().parent.parent / "data"
PROJ = Path(os.environ["PROJ"])

RUBRIC = """You are a strict native Tunisian Arabic (Derja) reviewer auditing synthetic SLU training data (SLURP-TN style).

For EACH item judge:
- derja (1-5): 5 = unmistakably spoken Tunisian Derja (Tunisian function words, natural French/English code-switching fine); 1 = MSA or another dialect.
- natural (1-5): does it sound like something a person says to a voice assistant?
- intent_match (bool): does the text genuinely express the stated intent label?
- slots_ok (bool): in `annotated`, is every marked span a correct value for its slot label, AND is no obvious slot value left unmarked? Format: `<label> value >`.
- If slots_ok is false ONLY because of fixable markup (wrong boundary, missed span, wrong label), provide `fixed_annotated` with the minimal correction. Never rewrite the words themselves.

Return ONLY a JSON array, one object per item, same order:
{"id": "...", "derja": 1-5, "natural": 1-5, "intent_match": true/false, "slots_ok": true/false, "fixed_annotated": "..." (optional)}"""


def judge_batch(items, model, temperature=0.2):
    payload = [{"id": it["id"], "intent": it["intent"],
                "annotated": it["annotated"]} for it in items]
    prompt = (RUBRIC + "\n\nITEMS:\n"
              + json.dumps(payload, ensure_ascii=False, indent=1))
    res = generate_json(model, prompt, temperature=temperature)
    if isinstance(res, dict):
        res = [res]
    return {r.get("id"): r for r in res if isinstance(r, dict)}


def passes(v):
    try:
        return (int(v.get("derja", 0)) >= 4 and int(v.get("natural", 0)) >= 3
                and bool(v.get("intent_match")) and bool(v.get("slots_ok")))
    except (TypeError, ValueError):
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, default=10)
    ap.add_argument("--models", default=None,
                    help="comma-separated annotator models (default: single "
                         "JUDGE_MODEL; use 'panel' for the 3-model majority "
                         "panel from JUDGE_MODELS)")
    ap.add_argument("--limit", type=int, default=0, help="debug: judge only N items")
    ap.add_argument("--workers", type=int,
                    default=int(os.environ.get("WORKERS", 6)),
                    help="concurrent item-batches; each also fans out its "
                         "annotators, so peak calls = workers x annotators")
    args = ap.parse_args()

    if args.models == "panel":
        annotators = JUDGE_MODELS
    elif args.models:
        annotators = [m.strip() for m in args.models.split(",") if m.strip()]
    else:
        annotators = [JUDGE_MODEL]
    majority = len(annotators) // 2 + 1

    slot_inv = {l.strip() for l in open(PROJ / "data/slurptn/slot_labels.txt") if l.strip()}
    items = [json.loads(l) for l in open(DATA / "gen_valid.jsonl", encoding="utf-8")]
    if args.limit:
        items = items[:args.limit]

    judged_path = DATA / "gen_judged.jsonl"
    done = set()
    if judged_path.exists():
        for l in open(judged_path, encoding="utf-8"):
            done.add(json.loads(l)["id"])
    todo = [it for it in items if it["id"] not in done]
    print(f"annotators={annotators} (majority {majority}/{len(annotators)}) | "
          f"workers={args.workers} | items={len(items)} todo={len(todo)}")

    chunks = [todo[i:i + args.batch] for i in range(0, len(todo), args.batch)]

    def judge_chunk(chunk):
        verdicts = {}   # model -> {id: verdict}
        with ThreadPoolExecutor(max_workers=len(annotators)) as ex2:
            futs = {m: ex2.submit(judge_batch, chunk, m) for m in annotators}
            for m, f in futs.items():
                try:
                    verdicts[m] = f.result()
                except Exception as e:
                    print(f"  annotator {m} failed on a batch: {e}")
                    verdicts[m] = {}
        return chunk, verdicts

    n_done = 0
    with open(judged_path, "a", encoding="utf-8") as fout, \
            ThreadPoolExecutor(max_workers=args.workers) as ex:
        for chunk, verdicts in ex.map(judge_chunk, chunks):
            for it in chunk:
                per = {m: verdicts[m].get(it["id"], {}) for m in annotators}
                votes = sum(passes(v) for v in per.values())
                ok = votes >= majority
                # repair path: strongest annotator (first) may fix markup; the
                # fix must re-validate, and it flips only THAT annotator's vote
                r1 = per[annotators[0]]
                fixed = r1.get("fixed_annotated")
                if not passes(r1) and fixed and votes + 1 >= majority:
                    parsed, err = parse_annotation(norm(str(fixed)), slot_inv)
                    if not err and int(r1.get("derja", 0)) >= 4 \
                            and int(r1.get("natural", 0)) >= 3 \
                            and bool(r1.get("intent_match")):
                        slots, text = parsed
                        it["annotated"] = norm(str(fixed))
                        it["text"] = text
                        it["slots"] = [{"label": l, "value": v} for l, v in slots]
                        it["meta"]["judge_fixed"] = True
                        ok = votes + 1 >= majority
                it["judge"] = {"votes": votes, "of": len(annotators),
                               "keep": ok, "per_annotator": per}
                fout.write(json.dumps(it, ensure_ascii=False) + "\n")
            fout.flush()
            n_done += len(chunk)
            print(f"  judged {n_done}/{len(todo)}")

    all_judged = [json.loads(l) for l in open(judged_path, encoding="utf-8")]
    seen, kept = set(), []
    for it in all_judged:
        if it["id"] in seen:
            continue
        seen.add(it["id"])
        if it.get("judge", {}).get("keep"):
            kept.append(it)
    with open(DATA / "gen_kept.jsonl", "w", encoding="utf-8") as f:
        for it in kept:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")
    with open(DATA / "tts_queue.jsonl", "w", encoding="utf-8") as f:
        for it in kept:
            f.write(json.dumps({"id": it["id"], "text": it["text"]},
                               ensure_ascii=False) + "\n")
    print(f"kept {len(kept)}/{len(seen)} -> gen_kept.jsonl + tts_queue.jsonl")
    print("Next: run VoxCPM voice cloning over tts_queue.jsonl -> "
          "augmentation/data/wav/<id>.wav (16 kHz mono)")


if __name__ == "__main__":
    main()
