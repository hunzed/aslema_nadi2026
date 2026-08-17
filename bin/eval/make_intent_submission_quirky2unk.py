#!/usr/bin/env python3
"""NADI-2026 Subtask 5.1 -- package a run under the "punt -> abstain" rule.

    python bin/eval/make_intent_submission_quirky2unk.py --raw <preds.jsonl> --out <zip>

Same idea as make_intent_submission_floor.py, but parameterised so any run can
be packaged. The floor script stays hard-wired on purpose (it reproduces the
submitted 66.78 artifact byte-for-byte); this one is the reusable sibling.

--------------------------------------------------------------------------
THE RULE
--------------------------------------------------------------------------
Take the model's own predicted intent, then:

  general_quirky                        -> unknown   (blanket, no exceptions)
  any intent outside the 6 TRAINED      -> unknown   (unseen at train time,
  scenarios (event_query, calendar_add,              so the task rules score it
  calendar_remove, ...)                              as `unknown`)
  unparseable response                  -> unknown
  otherwise                             -> keep, unchanged

NO prediction is ever replaced by a DIFFERENT intent -- every change is a label
being replaced by an abstention. The one exception is CANON below, which
expands SLURP-TN's bare action fragments (`query`, `addcontact`, `set`, ...)
into proper `scenario_action` labels. Those fragments are annotation noise in
data/slurptn/intent_labels.txt; left raw they are not valid SLURP-60 labels and
score wrong by construction. `query` -> `alarm_query` is arbitrary (bare `query`
is the action of TEN SLURP intents) and is kept only for consistency with the
submitted floor artifact; --no-canon sends the fragments to `unknown` instead.
"""
import argparse
import json
import re
import zipfile
from collections import Counter
from pathlib import Path

# The 6 scenarios SLURP-TN train actually covers (read off train.csv).
TRAINED_SCENARIOS = {"alarm", "email", "general", "news", "takeaway", "weather"}

# Original SLURP inventory: 18 scenarios x actions = 60 intents
# (Bastianelli et al., EMNLP 2020; same taxonomy as Amazon MASSIVE).
SLURP_60 = {
    "alarm_query", "alarm_remove", "alarm_set",
    "audio_volume_down", "audio_volume_mute", "audio_volume_other", "audio_volume_up",
    "calendar_query", "calendar_remove", "calendar_set",
    "cooking_query", "cooking_recipe",
    "datetime_convert", "datetime_query",
    "email_addcontact", "email_query", "email_querycontact", "email_sendemail",
    "general_greet", "general_joke", "general_quirky",
    "iot_cleaning", "iot_coffee", "iot_hue_lightchange", "iot_hue_lightdim",
    "iot_hue_lightoff", "iot_hue_lighton", "iot_hue_lightup",
    "iot_wemo_off", "iot_wemo_on",
    "lists_createoradd", "lists_query", "lists_remove",
    "music_dislikeness", "music_likeness", "music_query", "music_settings",
    "news_query",
    "play_audiobook", "play_game", "play_music", "play_podcasts", "play_radio",
    "qa_currency", "qa_definition", "qa_factoid", "qa_maths", "qa_stock",
    "recommendation_events", "recommendation_locations", "recommendation_movies",
    "social_post", "social_query",
    "takeaway_order", "takeaway_query",
    "transport_query", "transport_taxi", "transport_ticket", "transport_traffic",
    "weather_query",
}
assert len(SLURP_60) == 60

# SLURP-TN annotation-noise fragments -> canonical form.
CANON = {
    "Emails": "email_query", "sendemail": "email_sendemail",
    "addcontact": "email_addcontact", "querycontact": "email_querycontact",
    "greet": "general_greet", "joke": "general_joke", "quirky": "general_quirky",
    "set": "alarm_set", "query": "alarm_query", "remove": "alarm_remove",
}


def parse_intent(resp):
    """raw model response -> the intent string it named, or None if unusable."""
    resp = (resp or "").strip()
    intent = None
    try:
        v = json.loads(resp)
        if isinstance(v, dict):
            intent = v.get("intent")
    except json.JSONDecodeError:
        m = re.search(r'"intent"\s*:\s*"([^"]+)"', resp)
        if m:
            intent = m.group(1)
    return intent.strip() if isinstance(intent, str) else None


# --literal: the only two edits, nothing else touched.
TO_UNKNOWN = {"general_quirky", "event_query", "calendar_add", "calendar_remove"}


def run_literal(args):
    rows, changed = [], Counter()
    for line in open(args.raw, encoding="utf-8"):
        r = json.loads(line)
        intent = parse_intent(r.get("response"))
        hyp = "unknown" if intent in TO_UNKNOWN else intent
        if hyp != intent:
            changed[f"{intent} -> unknown"] += 1
        rows.append({"id": r["meta"]["id"], "hyp": hyp})

    rows.sort(key=lambda r: int(re.match(r"(\d+)", r["id"]).group(1)))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    payload = "\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n"
    with zipfile.ZipFile(args.out, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("predictions.jsonl", payload)

    print(f"source : {args.raw}")
    print(f"rows   : {len(rows)}   changed: {sum(changed.values())}")
    for k, v in changed.most_common():
        print(f"  {k:44s} {v:4d}")
    print("\nfinal label distribution:")
    for k, v in Counter(r["hyp"] for r in rows).most_common():
        print(f"  {str(k):24s} {v:4d}")
    print(f"\nzip: {args.out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True, type=Path,
                    help="model predictions jsonl (response + meta.id)")
    ap.add_argument("--out", required=True, type=Path, help="submission .zip")
    ap.add_argument("--no-canon", action="store_true",
                    help="send bare action fragments to `unknown` instead of "
                         "expanding them (see CANON)")
    ap.add_argument("--literal", action="store_true",
                    help="ONLY change general_quirky and the listed out-of-scope "
                         "intents to `unknown`; pass every other prediction "
                         "through exactly as the model emitted it (no CANON)")
    args = ap.parse_args()

    if args.literal:
        return run_literal(args)

    if not args.raw.exists():
        raise SystemExit(f"missing raw predictions: {args.raw}")

    rows, counts, changed = [], Counter(), Counter()
    for line in open(args.raw, encoding="utf-8"):
        r = json.loads(line)
        raw_intent = parse_intent(r.get("response"))
        intent = raw_intent
        if intent is not None and not args.no_canon:
            intent = CANON.get(intent, intent)

        if intent is None:
            hyp, rule = "unknown", "unparseable -> unknown"
        elif intent == "general_quirky":
            hyp, rule = "unknown", "general_quirky -> unknown"
        elif intent not in SLURP_60 or intent.split("_", 1)[0] not in TRAINED_SCENARIOS:
            hyp, rule = "unknown", "out-of-scope -> unknown"
        else:
            hyp, rule = intent, "keep"

        counts[rule] += 1
        if hyp != raw_intent:
            changed[f"{raw_intent} -> {hyp}"] += 1
        rows.append({"id": r["meta"]["id"], "hyp": hyp})

    # numeric sort: ids are "<N>-nadi_test-clean", so lexicographic is wrong
    rows.sort(key=lambda r: int(re.match(r"(\d+)", r["id"]).group(1)))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    payload = "\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n"
    with zipfile.ZipFile(args.out, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("predictions.jsonl", payload)

    print(f"source : {args.raw}")
    print(f"rows   : {len(rows)}")
    for k, v in sorted(counts.items()):
        print(f"  {k:28s} {v:4d}  ({v / len(rows):5.1%})")
    print("\nchanges applied:")
    for k, v in changed.most_common():
        print(f"  {k:44s} {v:4d}")
    print("\nfinal label distribution:")
    for k, v in Counter(r["hyp"] for r in rows).most_common():
        print(f"  {k:24s} {v:4d}")
    print(f"\nzip: {args.out}")


if __name__ == "__main__":
    main()
