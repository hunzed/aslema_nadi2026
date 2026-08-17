#!/usr/bin/env python3
"""Stage 1 - few-shot synthetic generation of Tunisian Derja SLU data.

Three complementary modes per intent (diversity beats volume):
  coverage : fresh utterances for an intent, slot combos sampled from the
             gold combo distribution + occasional plausible-novel combos,
             slot VALUES explicitly required to be new (unseen-value
             robustness -> CVER). This is the main mode; quotas from stage 0
             are inverse to train coverage, so absent/rare intents (Emails,
             set, addcontact, joke, ...) get the most mass.
  paraphrase: dialectal re-phrasings of real train rows that keep slot labels
             AND values fixed (lexical/word-order variety for same semantics).
  distractor: utterances for the intent containing NO slots at all, phrased so
             they superficially resemble slot-bearing ones (teaches precision;
             only for intents whose gold combos include the empty combo).

Few-shot: K exemplars per call (default 6) = up to 4 same-intent seeds
(rotated) + 2 fixed cross-intent anchors for format. Output is a strict JSON
array; every item carries provenance meta. Resume-safe: completed batch ids
are tracked in gen_done.txt and skipped.

Usage:
  python generate.py                # full run per quotas.json
  python generate.py --intents Emails set --batch 20 --shots 6
"""
import argparse
import json
import os
import random
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from gemini_client import GEN_MODEL, generate_json  # noqa: E402

DATA = Path(__file__).resolve().parent.parent / "data"

SYSTEM = """You are a native Tunisian Arabic (Derja) speaker creating training data for a Tunisian voice assistant (SLURP-TN style).

HARD RULES:
1. Language: authentic spoken TUNISIAN Derja in Arabic script. NEVER Modern Standard Arabic, never Egyptian/Levantine/Gulf. Natural Tunisian function words (بلاهي، برشا، تو، فما، شنوة، علاش، نحب، باش...) and natural French/English code-switching exactly like the examples (des alarmes, email, week-end...).
2. Style: short spoken commands/questions to a voice assistant, 3~15 words, first person, casual. No punctuation-heavy written style.
3. Annotation format: wrap each slot value as `<slot_label> value >` inline, e.g.:
   بلاهي فيقني <time> السبعة و النصف متاع الصباح >
   A space after `<slot_label>` and a space before the closing `>`. Slot labels come ONLY from the provided slot inventory. Nested slots are forbidden.
4. `text` must equal `annotated` with the markup removed (same words, same order).
5. Diversity: every utterance must differ substantially from the examples and from each other (different wording, structure, slot values). Invent NEW realistic slot values (Tunisian names, places, times, artists...) - do not copy values from the examples.

Return ONLY a JSON array, each element:
{"text": "...", "annotated": "...", "intent": "...", "slots": [{"label": "...", "value": "..."}]}"""

ANCHORS = [
    {"text": "بلاهي فيقني السبعة و النصف متاع الصباح",
     "annotated": "بلاهي فيقني <time> السبعة و النصف متاع الصباح >",
     "intent": "alarm_set", "slots": [{"label": "time", "value": "السبعة و النصف متاع الصباح"}]},
    {"text": "فماشي des alarmes مبرمجين شنوما هوما",
     "annotated": "فماشي des alarmes مبرمجين شنوما هوما",
     "intent": "alarm_query", "slots": []},
]


def build_prompt(mode, intent, pool, k_shots, n_items, rng):
    seeds = pool["seeds"].get(intent, [])
    shots = rng.sample(seeds, min(len(seeds), max(0, k_shots - 2)))
    shot_objs = ANCHORS + [
        {"text": s["text"], "annotated": s["annotated"], "intent": intent,
         "slots": [{"label": l, "value": v} for l, v in s["slots"]]}
        for s in shots]

    slot_inv = ", ".join(pool["slots"])
    combos = pool["slot_combos"].get(intent) or [[]]
    lines = [SYSTEM, "",
             f"SLOT INVENTORY (the only legal labels): {slot_inv}", "",
             "EXAMPLES (gold data, imitate the style and format):",
             json.dumps(shot_objs, ensure_ascii=False, indent=1), ""]

    if mode == "paraphrase":
        src = rng.choice(seeds)
        lines += [
            f'TASK: produce {n_items} DIFFERENT Tunisian Derja paraphrases of this utterance. '
            f'Keep the SAME intent ("{intent}"), the SAME slot labels and the SAME slot values '
            f'(verbatim), but change the wording, word order and phrasing:',
            json.dumps({"annotated": src["annotated"]}, ensure_ascii=False)]
    elif mode == "distractor":
        lines += [
            f'TASK: produce {n_items} NEW Tunisian Derja utterances with intent "{intent}" '
            f'that contain NO slot values at all ("slots": [], "annotated" == "text"), but are '
            f'phrased naturally, similar in flavor to slot-bearing requests.']
    else:  # coverage
        picked = [rng.choice(combos) for _ in range(n_items)]
        lines += [
            f'TASK: produce {n_items} NEW Tunisian Derja utterances with intent "{intent}".',
            'For item i use exactly the slot-label set given at position i below '
            '(empty list = no slots). Invent fresh, realistic slot values:',
            json.dumps(picked, ensure_ascii=False)]
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, default=20, help="items per API call")
    ap.add_argument("--shots", type=int, default=6, help="few-shot exemplars per call")
    ap.add_argument("--intents", nargs="*", help="restrict to these intents")
    ap.add_argument("--paraphrase-per-intent", type=int, default=60)
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--temperature", type=float, default=1.1)
    ap.add_argument("--max-batches", type=int, default=0,
                    help="stop after N batches (0 = all; use for smoke tests)")
    ap.add_argument("--workers", type=int,
                    default=int(os.environ.get("WORKERS", 6)),
                    help="concurrent API calls (default 6 or $WORKERS)")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    pool = json.loads((DATA / "seed_pool.json").read_text())
    quotas = json.loads((DATA / "quotas.json").read_text())
    out_path = DATA / "gen_raw.jsonl"
    done_path = DATA / "gen_done.txt"
    done = set(done_path.read_text().split()) if done_path.exists() else set()

    intents = args.intents or pool["intents"]
    plan = []  # (batch_id, mode, intent, n)
    for intent in intents:
        q = quotas.get(intent, 100)
        if q <= 0:
            continue   # artifact labels with no seeds: never generated
        n_cov = q
        for b in range(0, n_cov, args.batch):
            plan.append((f"cov:{intent}:{b}", "coverage", intent,
                         min(args.batch, n_cov - b)))
        if pool["seeds"].get(intent):
            n_par = min(args.paraphrase_per_intent, max(q // 2, 20))
            for b in range(0, n_par, args.batch):
                plan.append((f"par:{intent}:{b}", "paraphrase", intent,
                             min(args.batch, n_par - b)))
        if [] in [list(c) for c in pool["slot_combos"].get(intent, [])] or \
                not pool["slot_combos"].get(intent):
            for b in range(0, max(args.batch, q // 6), args.batch):
                plan.append((f"dis:{intent}:{b}", "distractor", intent,
                             args.batch))

    todo = [p for p in plan if p[0] not in done]
    if args.max_batches:
        todo = todo[:args.max_batches]
    print(f"model={GEN_MODEL} | workers={args.workers} | "
          f"batches total={len(plan)} todo={len(todo)}")

    def run_one(task):
        bid, mode, intent, n = task
        # per-batch RNG: deterministic AND thread-safe
        trng = random.Random(f"{args.seed}:{bid}")
        prompt = build_prompt(mode, intent, pool, args.shots, n, trng)
        try:
            items = generate_json(GEN_MODEL, prompt,
                                  temperature=args.temperature)
        except Exception as e:
            return bid, None, str(e)
        return bid, items if isinstance(items, list) else [items], None

    n_items = 0
    with open(out_path, "a", encoding="utf-8") as fout, \
            open(done_path, "a") as fdone, \
            ThreadPoolExecutor(max_workers=args.workers) as ex:
        for bi, (bid, items, err) in enumerate(ex.map(run_one, todo)):
            mode, intent = bid.split(":")[0], bid.split(":")[1]
            if err is not None:
                print(f"  [{bid}] FAILED permanently: {err}")
                continue
            kept = 0
            for j, it in enumerate(items):
                if not isinstance(it, dict) or "annotated" not in it:
                    continue
                it.setdefault("intent", intent)
                it["meta"] = {"mode": mode, "batch": bid, "model": GEN_MODEL}
                it["id"] = f"synth_{bid.replace(':', '_')}_{j:03d}"
                fout.write(json.dumps(it, ensure_ascii=False) + "\n")
                kept += 1
            fout.flush()
            fdone.write(bid + "\n")
            fdone.flush()
            n_items += kept
            print(f"  [{bi + 1}/{len(todo)}] {bid}: +{kept} (total {n_items})")
    print(f"done -> {out_path}")


if __name__ == "__main__":
    main()
