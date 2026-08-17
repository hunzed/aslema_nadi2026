#!/usr/bin/env python3
"""Stage 2 - deterministic validation of generated items (free, no API).

Checks, in order (first failure wins, recorded in rejects file):
  format     annotation grammar parses: `<label> value >` tokens, balanced,
             labels in inventory, values non-empty, no nesting
  text_match text == annotated stripped of markup (whitespace-normalized)
  intent     intent label in inventory
  arabic     Arabic-script ratio over letters >= 0.5 (code-switch tolerated)
  length     2..25 whitespace tokens
  dup        near-duplicate (char-3gram Jaccard >= 0.85) of a gold row or an
             earlier accepted synthetic row of the same intent

Outputs: gen_valid.jsonl, gen_rejects.jsonl (with reasons), stats to stdout.
"""
import os
import collections
import csv
import json
import re
import sys
from pathlib import Path

DATA = Path(__file__).resolve().parent.parent / "data"
PROJ = Path(os.environ["PROJ"])

OPEN_RE = re.compile(r"<([a-z_]+)>")
AR_RE = re.compile(r"[؀-ۿ]")
LETTER_RE = re.compile(r"[^\W\d_]", re.UNICODE)


def parse_annotation(annotated, slot_inv):
    """Return (slots, plain_text) or (None, reason)."""
    slots, plain, i = [], [], 0
    while i < len(annotated):
        m = OPEN_RE.search(annotated, i)
        if not m:
            tail = annotated[i:]
            if ">" in tail:
                return None, "stray_close"
            plain.append(tail)
            break
        head = annotated[i:m.start()]
        if ">" in head:
            return None, "stray_close"
        plain.append(head)
        if m.group(1) not in slot_inv:
            return None, f"bad_label:{m.group(1)}"
        close = annotated.find(">", m.end())
        if close == -1:
            return None, "unclosed"
        value = annotated[m.end():close].strip()
        if OPEN_RE.search(value):
            return None, "nested"
        if not value:
            return None, "empty_value"
        slots.append((m.group(1), value))
        plain.append(" " + value + " ")
        i = close + 1
    text = re.sub(r"\s+", " ", "".join(plain)).strip()
    return (slots, text), None


def norm(s):
    return re.sub(r"\s+", " ", s.strip())


def trigrams(s):
    s = re.sub(r"\s+", "", s)
    return {s[i:i + 3] for i in range(max(1, len(s) - 2))}


def jaccard(a, b):
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def main():
    slot_inv = {l.strip() for l in open(PROJ / "data/slurptn/slot_labels.txt") if l.strip()}
    intent_inv = {l.strip() for l in open(PROJ / "data/slurptn/intent_labels.txt") if l.strip()}

    gold_grams = collections.defaultdict(list)  # intent -> [trigram sets]
    for name in ("train", "validation"):
        with open(PROJ / f"data/slurptn/data/manifests/{name}.csv", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                gold_grams[r["intent"]].append(trigrams(r["tun_transcription"]))

    src = DATA / "gen_raw.jsonl"
    kept, rejects, seen_ids = [], [], set()
    accepted_grams = collections.defaultdict(list)
    reasons = collections.Counter()

    for line in open(src, encoding="utf-8"):
        it = json.loads(line)
        if it.get("id") in seen_ids:
            continue
        seen_ids.add(it.get("id"))

        def rej(why):
            reasons[why] += 1
            it["reject_reason"] = why
            rejects.append(it)

        annotated = norm(str(it.get("annotated", "")))
        text = norm(str(it.get("text", "")))
        intent = str(it.get("intent", ""))
        parsed, err = parse_annotation(annotated, slot_inv)
        if err:
            rej(f"format:{err}"); continue
        slots, stripped = parsed
        if not text:
            text = stripped
        if stripped != text:
            rej("text_match"); continue
        if intent not in intent_inv:
            rej("intent"); continue
        letters = LETTER_RE.findall(text)
        if letters and len(AR_RE.findall(text)) / len(letters) < 0.5:
            rej("arabic_ratio"); continue
        ntok = len(text.split())
        if not (2 <= ntok <= 25):
            rej("length"); continue
        g = trigrams(text)
        pool = gold_grams.get(intent, []) + accepted_grams[intent]
        if any(jaccard(g, h) >= 0.85 for h in pool):
            rej("near_dup"); continue

        it["annotated"] = annotated
        it["text"] = text
        it["slots"] = [{"label": l, "value": v} for l, v in slots]
        accepted_grams[intent].append(g)
        kept.append(it)

    with open(DATA / "gen_valid.jsonl", "w", encoding="utf-8") as f:
        for it in kept:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")
    with open(DATA / "gen_rejects.jsonl", "w", encoding="utf-8") as f:
        for it in rejects:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")

    n = len(kept) + len(rejects)
    print(f"in={n} kept={len(kept)} rejected={len(rejects)}")
    for why, c in reasons.most_common():
        print(f"  {why:24s} {c}")
    per_int = collections.Counter(it["intent"] for it in kept)
    print("kept per intent (lowest 8):",
          dict(sorted(per_int.items(), key=lambda kv: kv[1])[:8]))


if __name__ == "__main__":
    main()
