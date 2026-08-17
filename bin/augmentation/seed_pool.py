#!/usr/bin/env python3
"""Stage 0 - build the seed pool, coverage report, and generation quotas.

Reads the SLURP-TN train (+ validation) manifests and emits:
  augmentation/data/seed_pool.json   per-intent real exemplars + per-slot value banks
  augmentation/data/quotas.json      how many synthetic utterances to generate per intent
  (stdout)                       coverage report - the gaps synthetic data must fill

Dev usage policy: dev examples are included as seeds ONLY for intents with
<3 train examples (incl. the two train-absent labels Emails/set). Those rows
are tagged "from_dev": true so downstream can account for the mild
dev-contamination (dev metrics on those intents become slightly optimistic).
"""
import os
import argparse
import collections
import csv
import json
import re
from pathlib import Path

PROJ = Path(os.environ["PROJ"])
SLOT_RE = re.compile(r"<([a-z_]+)>")


def parse_slots(annotated):
    """`... <label> value > ...` -> list of (label, value). Robust enough for
    gold data (well-formed by construction)."""
    out = []
    i = 0
    while True:
        m = SLOT_RE.search(annotated, i)
        if not m:
            break
        close = annotated.find(">", m.end())
        # value runs from after the open tag to the next standalone '>'
        end = annotated.find(" >", m.end())
        if end == -1:
            end = len(annotated)
        out.append((m.group(1), annotated[m.end():end].strip()))
        i = end + 1
    return out


def load_manifest(path, origin):
    rows = []
    with open(path, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            rows.append({
                "id": r["ID"],
                "text": r["tun_transcription"].strip(),
                "annotated": r["tun_slu_annotation"].strip(),
                "intent": r["intent"].strip(),
                "slots": parse_slots(r["tun_slu_annotation"]),
                "from_dev": origin == "dev",
            })
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target-total", type=int, default=12000,
                    help="approx pre-filter synthetic size to plan quotas for")
    ap.add_argument("--out-dir", default=str(PROJ / "data" / "augmentation"))
    ap.add_argument("--allow-dev-seeds", action="store_true",
                    help="OPT-IN: let the two zero-train intents (Emails, set) "
                         "borrow style seeds from dev. DEFAULT IS TRAIN-ONLY - "
                         "dev is never touched by generation.")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    intents = [l.strip() for l in open(PROJ / "data/slurptn/intent_labels.txt") if l.strip()]
    slots = [l.strip() for l in open(PROJ / "data/slurptn/slot_labels.txt") if l.strip()]

    train = load_manifest(PROJ / "data/slurptn/data/manifests/train.csv", "train")
    dev = load_manifest(PROJ / "data/slurptn/data/manifests/validation.csv", "dev")

    by_intent = collections.defaultdict(list)
    for r in train:
        by_intent[r["intent"]].append(r)
    tcount = {i: len(by_intent.get(i, [])) for i in intents}

    # DEFAULT: TRAIN-ONLY. Dev is loaded solely for the near-dup filter and
    # coverage stats - generation never sees it. The zero-train intents
    # (Emails, set) generate from the label name + cross-intent anchors.
    # --allow-dev-seeds is the explicit opt-in to borrow dev style seeds for
    # those two intents only (validate.py still near-dup-rejects vs dev).
    dev_seeded = []
    if args.allow_dev_seeds:
        for r in dev:
            if tcount.get(r["intent"], 0) == 0:
                by_intent[r["intent"]].append(r)
                dev_seeded.append(r["intent"])

    # slot value banks + per-intent slot-combo stats (from train+dev gold)
    slot_values = collections.defaultdict(set)
    combos = collections.defaultdict(collections.Counter)
    for r in train + dev:
        for lab, val in r["slots"]:
            slot_values[lab].add(val)
        combos[r["intent"]][tuple(sorted(l for l, _ in r["slots"]))] += 1

    # ---- quotas: inverse-coverage tiers, with ARTIFACT-LABEL policy --------
    # SLURP-TN's label set contains degenerate action-only variants of the
    # real scenario_action labels (set~alarm_set, Emails~email_sendemail,
    # sendemail~email_sendemail, greet~general_greet, joke~general_joke,
    # quirky~general_quirky, query~takeaway/*_query, querycontact/addcontact~
    # email_*) - verified by reading the actual rows. There is NO learnable
    # semantic distinction, only annotation noise, so:
    #   * zero-train artifact labels (Emails, set): quota 0 - generating from
    #     the label name would mass-produce mislabeled near-duplicates of the
    #     big legitimate classes and hurt them;
    #   * artifact labels WITH train seeds: capped small and anchored on
    #     their real examples (paraphrase-heavy), never name-based flooding.
    ARTIFACT_BARE = {"Emails", "set", "addcontact", "greet", "joke",
                     "querycontact", "sendemail", "quirky", "query"}
    quotas = {}
    for i in intents:
        n = tcount[i]
        if i in ARTIFACT_BARE:
            quotas[i] = 0 if n == 0 else min(120, 20 * max(1, n))
        elif n < 40:
            quotas[i] = 400
        elif n < 150:
            quotas[i] = 250
        else:
            quotas[i] = 150
    # scale only the REAL labels to hit target_total; artifact caps are hard
    art_sum = sum(v for k, v in quotas.items() if k in ARTIFACT_BARE)
    real_sum = sum(v for k, v in quotas.items() if k not in ARTIFACT_BARE)
    scale = max(0.1, (args.target_total - art_sum) / max(1, real_sum))
    quotas = {k: (v if k in ARTIFACT_BARE else max(40, int(v * scale)))
              for k, v in quotas.items()}

    pool = {
        "intents": intents,
        "slots": slots,
        "train_counts": tcount,
        "seeds": {i: [
            {"text": r["text"], "annotated": r["annotated"],
             "slots": r["slots"], "from_dev": r["from_dev"], "id": r["id"]}
            for r in by_intent.get(i, [])] for i in intents},
        "slot_values": {k: sorted(v) for k, v in slot_values.items()},
        "slot_combos": {i: [list(c) for c, _ in combos[i].most_common(8)]
                        for i in intents},
    }
    (out_dir / "seed_pool.json").write_text(
        json.dumps(pool, ensure_ascii=False, indent=1))
    (out_dir / "quotas.json").write_text(json.dumps(quotas, indent=1))

    print(f"train rows {len(train)} | dev rows {len(dev)}")
    print(f"intents absent from train : {[i for i in intents if tcount[i] == 0]}")
    print(f"intents <10 train rows    : "
          f"{ {i: tcount[i] for i in intents if 0 < tcount[i] < 10} }")
    print(f"dev-seeded intents (mild dev contamination, tracked): "
          f"{sorted(set(dev_seeded))}")
    print(f"slot labels: {len(slots)}; value-bank sizes: "
          f"{ {k: len(v) for k, v in sorted(slot_values.items())[:6]} } ...")
    print(f"planned pre-filter synthetic total: {sum(quotas.values())}")
    print(f"wrote {out_dir}/seed_pool.json and quotas.json")


if __name__ == "__main__":
    main()
