#!/usr/bin/env python3
"""Rewrite the official_test infer jsonl with the OPEN (SLURP-60 + unknown) label
inventory instead of the released 23-label one.

Why: on the official test set the current models put 562/989 (56.8%) of
utterances into `general_quirky` and 986/989 inside the released 23 labels --
i.e. they never name an unseen intent and never say `unknown`. Per the
Codabench rules an unseen-intent utterance is correct as `unknown` OR as its
exact SLURP label, and `general_quirky` is simply wrong. The models can't emit
those answers because the prompt forbids them, so this is fixed at the prompt,
not by more training.

Writes data/swift/intent/<family>/official_test_open_infer.jsonl, a new SPLIT
name so it never collides with the existing runs or the resume-skip guard.

Usage:
  python bin/data_prep/make_open_inventory_infer.py            # both families
  python bin/data_prep/make_open_inventory_infer.py --family qwen
"""
import os
import argparse
import json
from pathlib import Path

PROJ = Path(os.environ["PROJ"])

# Original SLURP inventory: 18 scenarios x actions = 60 intents
# (Bastianelli et al., EMNLP 2020; same taxonomy as Amazon MASSIVE).
# Kept byte-identical to custom_gcp/scripts/package_intent_test_v2.py so the
# packaging step and the prompt can never drift apart.
SLURP_60 = [
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
]
assert len(SLURP_60) == 60 and len(set(SLURP_60)) == 60

# The 6 scenarios that SLURP-TN training data actually covers. Naming them
# explicitly is what stops the model reflexively snapping every out-of-domain
# utterance back onto a familiar in-domain label.
TRAINED_SCENARIOS = "alarm, email, general, news, takeaway, weather"

SYSTEM = (
    "You are a spoken language understanding system for Tunisian Arabic "
    "(Tunisian dialect; code-switching with French/English words is common).\n\n"
    "Task: listen to the audio utterance and classify the speaker's INTENT. "
    "Choose exactly one label from the fixed inventory below - do not invent "
    "new labels, do not translate, do not explain.\n\n"
    f"Valid intent labels ({len(SLURP_60)}): " + ", ".join(SLURP_60) + "\n\n"
    "If the utterance does not fit ANY label above, answer exactly: unknown\n\n"
    "IMPORTANT: this test set covers the FULL inventory above, which is much "
    "broader than the six scenarios (" + TRAINED_SCENARIOS + ") you may be "
    "most familiar with. Many utterances are about music, calendars, lists, "
    "IoT/smart-home devices, transport, cooking, social media, general "
    "question-answering or audio volume. Classify what you actually hear.\n"
    "Do NOT use general_quirky as a catch-all: reserve it for genuinely "
    "nonsensical or unanswerable chit-chat. If an utterance has a clear topic "
    "that is not in the list, answer unknown instead of general_quirky.\n\n"
    'OUTPUT FORMAT (valid single-line JSON, no markdown or extra text): '
    '{"intent": "<one label copied exactly from the list above, or unknown>"}'
)

USER = (
    "Listen to the utterance and identify its intent.\n"
    'Respond ONLY with a single-line JSON object: {"intent": "<label>"}'
)


def convert(family: str, split: str) -> None:
    src = PROJ / f"data/swift/intent/{family}/{split}_infer.jsonl"
    dst = PROJ / f"data/swift/intent/{family}/{split}_open_infer.jsonl"
    if not src.exists():
        raise SystemExit(f"missing {src} - run prepare_slurptn.sh first")

    n = 0
    with open(src, encoding="utf-8") as fin, open(dst, "w", encoding="utf-8") as fout:
        for line in fin:
            r = json.loads(line)
            for m in r["messages"]:
                if m["role"] == "system":
                    m["content"] = SYSTEM
                elif m["role"] == "user":
                    # keep the family-specific <audio> placeholder position
                    # (gemma wants it after the text, qwen before) -- only the
                    # instruction text around it is replaced.
                    m["content"] = (
                        f"{USER}\n<audio>" if m["content"].rstrip().endswith("<audio>")
                        else f"<audio>\n{USER}"
                    )
            r.setdefault("meta", {})["split"] = f"{split}_open"
            # NB: do NOT add a `dataset` key here. It appears in swift's *result*
            # jsonls, not its inputs -- swift's loader injects its own `dataset`
            # routing tag and a pre-existing one crashes _check_column_names with
            # "The table can't have duplicated columns but columns ['dataset']
            # are duplicated."
            r.pop("dataset", None)
            fout.write(json.dumps(r, ensure_ascii=False) + "\n")
            n += 1
    print(f"{family}/{split}: {n} rows -> {dst}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", choices=["qwen", "gemma"], default=None)
    # dev is all in-domain, so it is the regression check: the open prompt must
    # not cost in-domain accuracy relative to the released-23 run.
    ap.add_argument("--split", choices=["official_test", "dev"], default=None)
    a = ap.parse_args()
    for fam in ([a.family] if a.family else ["qwen", "gemma"]):
        for sp in ([a.split] if a.split else ["official_test", "dev"]):
            convert(fam, sp)
