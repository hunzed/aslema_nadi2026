# Synthetic data augmentation — LLM text generation + voice-cloned TTS

Produces Tunisian Derja SLU training data with inline slot annotations and
voice-cloned speech, targeting the intents that are underrepresented in the
2,677-utterance SLURP-TN training split.

**All prompts are in [`../prompts.txt`](../prompts.txt) (Part 3)** — generated from
this code by `../tools/build_prompts_txt.py`, so it always matches what runs.

## Funnel (numbers from our run)

| # | stage | tool | out |
|---|---|---|---|
| 0 | per-intent quotas, inverse to training frequency; few-shot seeds **from the train split only** | `seed_pool.py` | quotas + seeds |
| 1 | few-shot generation, 3 modes: *coverage* (new utterances, fresh slot values), *paraphrase* (same slots, reworded), *distractor* (slot-free, same intent) | `generate.py` | 20,937 raw |
| 2 | deterministic validation: annotation grammar, text/annotation agreement, label inventory, Arabic-script ratio, 2–25 token length, char-3gram near-duplicate vs gold and vs accepted synthetic | `validate.py` | 13,876 |
| 3 | 3 LLM annotators score Derja authenticity, naturalness, intent match, slot correctness; keep on **2-of-3 majority**; minimal markup repairs re-validated | `judge.py` | **12,138** |
| 4 | voice-cloned TTS, two variants: base VoxCPM2 and VoxCPM2+LoRA (fine-tuned on the train split), sharing one 152-clip reference pool | `tts/` | 23,300 renditions |
| 5 | deterministic acoustic screen (duration, RMS, clipping, voiced-frame activity, speaking rate) | `build_manifests.py` | **22,940** |
| 6 | manifests + ms-swift jsonls through the same converters as the real data | `build_manifests.py` | `synth`, `mix` (25,617) |

**Scope of filtering on the audio.** Stage 5 is the only filter applied to the
synthetic speech: it is code-only and model-free. No per-clip LLM/ASR
authenticity or naturalness check was run. A 650-clip listening study
(real vs base vs LoRA) informed only the choice of the base variant for the
mixed training set — it did not accept or reject individual clips.

## Run it

```bash
source ../config.sh && export GOOGLE_API_KEY=...
bash run_pipeline.sh                     # stages 0-3 (CPU, resume-safe)

# stage 4 - speech (GPU). Reference pool first:
python tts/gemini_asr_wer.py --train-refs        # score train clips with an LLM ASR
python tts/build_ref_pool.py --top 150           # keep the cleanest as cloning refs
bash tts/finetune/sbatch_finetune.sh             # optional: LoRA-tune VoxCPM2 on train
bash tts/submit_all_tts.sh                       # base + LoRA x 2 shards

# stages 5-6 - screen + package
python build_manifests.py --max-ratio 2.0
```

Then train on it with `AUG_TAG=mix` (real + synthetic) or `AUG_TAG=synth`:

```bash
sbatch --mem=200GB --export=ALL,MODEL=Qwen/Qwen3-Omni-30B-A3B-Instruct,\
MODEL_NAME=qwen3_omni_30b_a3b,FAMILY=qwen,SUBTASK=joint,NUM_EPOCHS=2,AUG_TAG=mix \
    ../training/sft_lora.sh
```

## Models used

| stage | model |
|---|---|
| generation, rare/zero-coverage intents | `gemini-3.1-pro-preview` |
| generation, bulk | `gemini-3.6-flash` |
| judging panel (3 annotators) | `gemini-3.1-pro-preview`, `gemini-3.6-flash`, `gemini-2.5-pro` |
| reference-clip ASR scoring | `gemini-3.1-pro-preview` |
| TTS | `openbmb/VoxCPM2` (base and LoRA-fine-tuned) |

All overridable via `config.sh`.
