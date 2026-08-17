#!/usr/bin/env python3
"""Score a swift-infer result file for Subtask 5.2 (Slot Filling).

Builds the SLURP-TN shared schema ({"id","ref","hyp"}) from the swift result
rows (response = model output; meta.ref = gold annotation carried through from
to_swift_slot_filling.py) and then computes WER / CER / CoER / CVER with the
official baseline scorer (evaluation/scoring.py from elyadata/SLURP-TN-baselines,
cloned under bin/_vendor by the data-prep step).

Usage:
    python eval_slot_filling.py <swift_result.jsonl> \
        --out-dir results/slot_filling/qwen2_5_omni_3b_dev \
        [--vendor bin/_vendor/SLURP-TN-baselines] [--reference <manifest.csv>]
"""
import argparse
import csv
import json
import sys
from pathlib import Path

DEFAULT_VENDOR = Path(__file__).resolve().parents[1] / "_vendor" / "SLURP-TN-baselines"


def load_gold_from_manifest(path):
    gold = {}
    with open(path, encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            gold[row["ID"]] = (row.get("tun_slu_annotation") or "").strip()
    return gold


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("result")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--vendor", default=str(DEFAULT_VENDOR),
                    help="path to the cloned SLURP-TN-baselines repo (for evaluation/scoring.py)")
    ap.add_argument("--reference", help="manifest CSV to source gold refs if meta.ref is blank")
    args = ap.parse_args()

    gold_by_id = load_gold_from_manifest(args.reference) if args.reference else {}

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = out_dir / "predictions.jsonl"

    rows = []
    for line in open(args.result, encoding="utf-8"):
        if not line.strip():
            continue
        r = json.loads(line)
        meta = r.get("meta") or {}
        uid = meta.get("id")
        ref = (meta.get("ref") or "").strip() or gold_by_id.get(uid, "")
        hyp = (r.get("response") or "").strip()
        rows.append({"id": uid, "ref": ref, "hyp": hyp})

    with open(predictions_path, "w", encoding="utf-8") as w:
        for r in rows:
            w.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"Wrote {len(rows)} predictions -> {predictions_path}")

    have_refs = [r for r in rows if r["ref"]]
    if not have_refs:
        print("No gold references present (held-out test?) - skipping metric computation.")
        (out_dir / "metrics.json").write_text(
            json.dumps({"total": len(rows), "scored": 0,
                        "note": "predictions only, no references"}, indent=2),
            encoding="utf-8")
        return

    # Score with the official baseline evaluator.
    sys.path.insert(0, args.vendor)
    try:
        from evaluation.scoring import compute_metrics_from_jsonl
    except Exception as e:  # noqa: BLE001
        sys.exit(f"Could not import the SLURP-TN baseline scorer from {args.vendor} "
                 f"(evaluation/scoring.py): {e}\nRun the data-prep step first so the "
                 f"repo is cloned, or pass --vendor.")

    metrics = compute_metrics_from_jsonl(predictions_file=str(predictions_path), task="slu")
    metrics = {k: (round(v, 4) if isinstance(v, float) else v) for k, v in dict(metrics).items()}
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))
    print(f"Wrote {out_dir/'metrics.json'}")


if __name__ == "__main__":
    main()
