#!/usr/bin/env python3
"""Regenerate ../prompts.txt from the code that actually sends the prompts.

Every prompt in prompts.txt is imported from its source module here, so the
released prompt file cannot drift from what the pipeline runs.

    python tools/build_prompts_txt.py          # rewrites prompts.txt
    python tools/build_prompts_txt.py --check  # non-zero exit if out of date (CI)

Sources:
  Part 1  data_prep/prompt_builders/{intent,slot_filling}_prompts.py
  Part 2  data_prep/make_open_inventory_infer.py
  Part 3  augmentation/generate.py, augmentation/judge.py,
          augmentation/tts/gemini_asr_wer.py
"""
import argparse
import importlib.util
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("PROJ", str(ROOT))          # some modules read PROJ at import
BAR = "=" * 78


def load(relpath, name):
    """Import a module by file path, tolerating a module-level SystemExit."""
    spec = importlib.util.spec_from_file_location(name, ROOT / relpath)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    try:
        spec.loader.exec_module(mod)
    except SystemExit:
        pass
    return mod


def section(title):
    return f"\n{BAR}\n{title}\n{BAR}\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true",
                    help="verify prompts.txt is up to date instead of writing it")
    args = ap.parse_args()

    sys.path.insert(0, str(ROOT / "data_prep"))
    ip = load("data_prep/prompt_builders/intent_prompts.py", "intent_prompts")
    sp = load("data_prep/prompt_builders/slot_filling_prompts.py", "slot_filling_prompts")
    op = load("data_prep/make_open_inventory_infer.py", "open_inv")
    gen = load("augmentation/generate.py", "aug_generate")
    jud = load("augmentation/judge.py", "aug_judge")
    asr = load("augmentation/tts/gemini_asr_wer.py", "aug_asr")

    labels = ROOT / "data_prep/labels"
    intents = ip.load_labels(labels / "intent_labels.txt")
    slots = sp.load_labels(labels / "slot_labels.txt")

    out = [f"""ASLEMA @ NADI-2026 SHARED TASK 5 - ALL PROMPTS
{BAR}
GENERATED FILE - do not edit by hand.
Regenerate with:  python tools/build_prompts_txt.py
Every prompt below is imported from the module that sends it, so this file
cannot drift from the code that produced our results.

  Part 1  Task prompts       intent recognition and slot filling. The SAME
                             prompts are used zero-shot and as fine-tuning
                             targets, so both flow through identical evaluators.
  Part 2  Open inventory      60 SLURP labels + `unknown`, used only for the
                             official blind test set (the abstention result).
  Part 3  Augmentation        synthetic data generation, judging, and the
                             reference-selection ASR prompt.
"""]

    out.append(section("PART 1.1  SUBTASK 5.1 - INTENT RECOGNITION"))
    out.append("--- system prompt ---\n" + ip.build_system_prompt(intents)
               + "\n\n--- user turn ---\n<audio>\n" + ip.USER_INSTRUCTION + "\n")

    out.append(section("PART 1.2  SUBTASK 5.2 - SLOT FILLING"))
    out.append("--- system prompt ---\n" + sp.build_system_prompt(slots)
               + "\n\n--- user turn ---\n<audio>\n" + sp.USER_INSTRUCTION + "\n")

    out.append(section("PART 2  OPEN INVENTORY (60 SLURP LABELS + `unknown`)"))
    out.append("Used ONLY for the official blind test set. Differs from Part 1.1 in three\n"
               "ways, all prompt-side: the full 60-label inventory, `unknown` as an allowed\n"
               "answer, and the six trained scenarios named explicitly.\n\n"
               "--- system prompt ---\n" + op.SYSTEM
               + "\n\n--- user turn ---\n<audio>\n" + op.USER + "\n")

    out.append(section("PART 3  SYNTHETIC DATA AUGMENTATION"))
    out.append(
        "STAGE 1  GENERATION  (augmentation/generate.py)\n"
        f"  bulk intents      : {gen.GEN_MODEL}\n"
        "  rare/zero-coverage: set GEN_MODEL=<pro model> (see augmentation/run_pipeline.sh)\n"
        "  Each call = system prompt + slot inventory + few-shot examples\n"
        "  (2 fixed format anchors + up to 4 same-intent seeds drawn ONLY from the\n"
        "  train split) + one task line, requesting a JSON array of items.\n")
    out.append("\n--- generator system prompt ---\n" + gen.SYSTEM + "\n")
    out.append("\n--- task line: COVERAGE mode (the main mode) ---\n" + gen.TASK_COVERAGE
               + "\n[[\"time\"], [], [\"person\",\"event_name\"], ...]   <- slot-label sets, "
                 "sampled from the gold combination distribution\n")
    out.append("\n--- task line: PARAPHRASE mode ---\n" + gen.TASK_PARAPHRASE
               + '\n{"annotated": "<one real training row>"}\n')
    out.append("\n--- task line: DISTRACTOR mode ---\n" + gen.TASK_DISTRACTOR + "\n")

    out.append("\n" + "-" * 78 + "\nSTAGE 2  DETERMINISTIC VALIDATION  (augmentation/validate.py) - no model\n"
               + "-" * 78 + "\n"
               "  In order: annotation grammar; text == annotated minus markup; label and\n"
               "  intent inventory membership; Arabic-script ratio floor; 2-25 token length\n"
               "  window; character-3gram near-duplicate check against gold rows and against\n"
               "  previously accepted synthetic rows of the same intent.\n")

    out.append("\n" + "-" * 78 + "\nSTAGE 3  JUDGING PANEL  (augmentation/judge.py)\n" + "-" * 78 + "\n"
               f"  annotators: {', '.join(jud.JUDGE_MODELS)}\n"
               "  An item is kept on a 2-of-3 majority. A single annotator passes an item when\n"
               "  derja >= 4, natural >= 3, intent_match and slots_ok all hold; a markup repair\n"
               "  is accepted only if the repaired string re-validates in the Stage-2 parser.\n")
    out.append("\n--- judge rubric prompt ---\n" + jud.RUBRIC + "\n")

    out.append("\n" + "-" * 78 + "\nSTAGE 4  SPEECH  (augmentation/tts/) - no prompt\n" + "-" * 78 + "\n"
               "  Two renditions of every kept text, so one text yields up to two utterances:\n"
               "    (a) base VoxCPM2\n"
               "    (b) VoxCPM2 + LoRA fine-tuned on the SLURP-TN TRAIN split only\n"
               "  Both use the SAME reference pool for voice cloning; each item's reference is\n"
               "  chosen deterministically by a hash of its id. The pool is built by scoring\n"
               "  train clips with the ASR prompt below and keeping the cleanest, plus two\n"
               "  clips verified by ear. A silence guard retries generation up to 3x.\n")
    out.append("\n--- reference-selection ASR prompt ---\n" + asr.PROMPT + "\n")

    out.append("\n" + "-" * 78 + "\nSTAGE 5  ACOUSTIC SCREEN  (augmentation/build_manifests.py) - no model\n"
               + "-" * 78 + "\n"
               "  Deterministic, code-only screen applied to EVERY rendition, scored against\n"
               "  its own text: duration, signal level, clipping, voiced-frame activity and\n"
               "  speaking rate (seconds per character).\n\n"
               "  SCOPE: this is the ONLY filter applied to the synthetic audio. No per-clip\n"
               "  LLM or ASR authenticity/naturalness check was run. A 650-clip listening study\n"
               "  (real vs base vs LoRA) informed only the choice of TTS variant for the mixed\n"
               "  training set; it did not accept or reject individual clips.\n")
    out.append("\n" + BAR + "\n")

    text = "".join(out)
    target = ROOT / "prompts.txt"
    if args.check:
        current = target.read_text(encoding="utf-8") if target.exists() else ""
        if current != text:
            print("prompts.txt is OUT OF DATE - run: python tools/build_prompts_txt.py")
            return 1
        print("prompts.txt is up to date")
        return 0
    target.write_text(text, encoding="utf-8")
    print(f"wrote {target} ({len(text.splitlines())} lines)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
