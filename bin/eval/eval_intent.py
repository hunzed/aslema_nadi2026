#!/usr/bin/env python3
"""Score a swift-infer result file for Subtask 5.1 (Intent Recognition).

Each swift result row has `response` (model output, expected {"intent": "<label>"})
and, because infer ran with --remove_unused_columns false, the `meta` column we
attached in to_swift_intent.py (holding the utterance id and the gold intent).

Steps:
  1. parse the predicted label out of `response` (JSON first, then a
     case-insensitive exact/nearest match against the valid inventory);
  2. join to the gold label via meta.intent (or a --reference manifest by id);
  3. report accuracy + macro-F1 + weighted-F1 and write:
       <out_dir>/classifications.jsonl  ({"id","hyp","ref"} - shared schema)
       <out_dir>/metrics.json

Usage:
    python eval_intent.py <swift_result.jsonl> --labels data/slurptn/intent_labels.txt \
        --out-dir results/intent/qwen2_5_omni_3b_dev [--reference <manifest.csv>]
"""
import argparse
import csv
import difflib
import json
import re
from collections import defaultdict
from pathlib import Path


def load_labels(path):
    return sorted({l.strip() for l in Path(path).read_text(encoding="utf-8").splitlines()
                   if l.strip() and not l.startswith("#")})


def parse_intent(text, labels):
    if not text:
        return None
    lower_map = {l.lower(): l for l in labels}
    m = re.search(r'"intent"\s*:\s*"([^"]+)"', text)
    if m:
        cand = m.group(1).strip()
        if cand.lower() in lower_map:
            return lower_map[cand.lower()]
        near = difflib.get_close_matches(cand.lower(), lower_map.keys(), n=1, cutoff=0.85)
        if near:
            return lower_map[near[0]]
        # off-taxonomy but still a real prediction - keep it as a string
        # rather than discarding it. Downstream scorers (local + CodaBench)
        # require hyp to be a string; a dropped/None hyp either silently
        # excludes the row from scoring or (CodaBench) crashes the whole
        # submission on the first such row, so a wrong-but-present string
        # is strictly better than null.
        return cand
    # fallback: first valid label that appears as a whole word in the output
    for l in labels:
        if re.search(rf'\b{re.escape(l)}\b', text, flags=re.IGNORECASE):
            return l
    return None


def load_gold_from_manifest(path):
    gold = {}
    with open(path, encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            gold[row["ID"]] = (row.get("intent") or "").strip()
    return gold


def macro_weighted_f1(pairs, labels):
    """pairs: list of (gold, pred). Returns (accuracy, macro_f1, weighted_f1)."""
    tp = defaultdict(int); fp = defaultdict(int); fn = defaultdict(int); support = defaultdict(int)
    correct = 0
    for gold, pred in pairs:
        support[gold] += 1
        if pred == gold:
            correct += 1
            tp[gold] += 1
        else:
            fn[gold] += 1
            if pred is not None:
                fp[pred] += 1
    f1s = {}
    for l in set(list(support) + list(tp) + list(fp)):
        prec = tp[l] / (tp[l] + fp[l]) if (tp[l] + fp[l]) else 0.0
        rec = tp[l] / (tp[l] + fn[l]) if (tp[l] + fn[l]) else 0.0
        f1s[l] = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    present = [l for l in support]  # labels with at least one gold example
    macro = sum(f1s[l] for l in present) / len(present) if present else 0.0
    total = sum(support.values())
    weighted = sum(f1s[l] * support[l] for l in present) / total if total else 0.0
    acc = correct / total if total else 0.0
    return acc, macro, weighted


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("result")
    ap.add_argument("--labels", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--reference", help="manifest CSV to source gold labels if meta.intent is blank")
    args = ap.parse_args()

    labels = load_labels(args.labels)
    gold_by_id = load_gold_from_manifest(args.reference) if args.reference else {}

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    classifications = []
    pairs = []
    unparsed = 0

    for line in open(args.result, encoding="utf-8"):
        if not line.strip():
            continue
        row = json.loads(line)
        meta = row.get("meta") or {}
        uid = meta.get("id")
        gold = (meta.get("intent") or "").strip() or gold_by_id.get(uid, "")
        pred = parse_intent(row.get("response", ""), labels)
        if pred is None:
            unparsed += 1
        # never write a null hyp - CodaBench's scorer requires a string and
        # aborts the whole submission on the first non-string record.
        hyp = pred if pred is not None else (row.get("response") or "").strip()
        classifications.append({"id": uid, "hyp": hyp, "ref": gold or None})
        if gold:
            pairs.append((gold, pred))

    (out_dir / "classifications.jsonl").write_text(
        "\n".join(json.dumps(c, ensure_ascii=False) for c in classifications) + "\n",
        encoding="utf-8")

    metrics = {"total": len(classifications), "scored": len(pairs), "unparsed": unparsed}
    if pairs:
        acc, macro, weighted = macro_weighted_f1(pairs, labels)
        metrics.update({"accuracy": round(acc, 4),
                        "macro_f1": round(macro, 4),
                        "weighted_f1": round(weighted, 4)})
    else:
        metrics["note"] = "no gold labels available (held-out test?) - predictions only"

    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))
    print(f"Wrote {out_dir/'classifications.jsonl'} and {out_dir/'metrics.json'}")


if __name__ == "__main__":
    main()
