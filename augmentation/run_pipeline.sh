#!/bin/bash
# ============================================================================
# Synthetic Tunisian Derja SLU data - TEXT stages (no GPU, resume-safe).
#
#   source config.sh && export GOOGLE_API_KEY=...
#   bash augmentation/run_pipeline.sh
#
# Stages: seed pool + quotas -> few-shot generation (rare intents on the pro
# model, bulk on flash) -> deterministic validation -> 3-model judging panel.
# Outputs land in augmentation/data/. Every stage skips work already done, so
# it is safe to re-run after an interruption.
#
# After it finishes:
#   1. speech:   bash augmentation/tts/submit_all_tts.sh     (base + LoRA VoxCPM2)
#   2. package:  python augmentation/build_manifests.py      (acoustic screen + manifests)
# ============================================================================
set -e
: "${PROJ:?source config.sh first}"
PY="${PY:-python}"
cd "$PROJ"

echo "== stage 0: seed pool + per-intent quotas (train split only) =="
$PY augmentation/seed_pool.py "$@"

echo "== stage 1a: rare / zero-coverage intents (${RARE_GEN_MODEL:-gemini-3.1-pro-preview}) =="
GEN_MODEL="${RARE_GEN_MODEL:-gemini-3.1-pro-preview}" $PY augmentation/generate.py \
    --intents Emails set addcontact joke querycontact sendemail greet quirky query general_greet email_addcontact

echo "== stage 1b: bulk generation (${GEN_MODEL:-gemini-3.6-flash}) =="
$PY augmentation/generate.py

echo "== stage 2: deterministic validation =="
$PY augmentation/validate.py

echo "== stage 3: multi-annotator judging panel (keep on 2-of-3 majority) =="
$PY augmentation/judge.py ${JUDGE_ARGS:---models panel}

echo "== text pipeline complete =="
echo "kept: $(wc -l < augmentation/data/gen_kept.jsonl) utterances"
echo "next: bash augmentation/tts/submit_all_tts.sh"
