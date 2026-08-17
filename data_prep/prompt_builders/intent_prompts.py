#!/usr/bin/env python3
"""Zero-shot prompt builders for NADI-2026 Subtask 5.1 (Intent Recognition).

The intent-recognition tutorial in this repo's reference notebooks trains a
Whisper classifier; here we instead prompt a multimodal LLM zero-shot, so the
label inventory has to be *described in the prompt*. The system prompt is
model-family agnostic (identical text for Qwen and Gemma) so the comparison
stays "same prompt, different model". What differs per family is only where the
`<audio>` placeholder sits in the user turn (see to_swift_intent.py) -
Gemma-4 wants audio placed AFTER the text, Qwen2.5-Omni is order-insensitive.

The 23-label intent inventory is NOT hardcoded: it is read from the file
`data/slurptn/intent_labels.txt` produced by extract_labels.py during data
prep, so a change in the dataset can never silently desync the prompt from the
scorer.
"""
from pathlib import Path


def load_labels(path):
    """Read one label per line, ignoring blanks/comments. Returns a sorted list."""
    labels = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            labels.append(line)
    return sorted(set(labels))


def build_system_prompt(intent_labels):
    label_list = ", ".join(intent_labels)
    return (
        "You are a spoken language understanding system for Tunisian Arabic "
        "(Tunisian dialect; code-switching with French/English words is common).\n\n"
        "Task: listen to the audio utterance and classify the speaker's INTENT. "
        "Choose exactly one label from the fixed inventory below - do not invent "
        "new labels, do not translate, do not explain.\n\n"
        f"Valid intent labels ({len(intent_labels)}): {label_list}\n\n"
        'OUTPUT FORMAT (valid single-line JSON, no markdown or extra text): '
        '{"intent": "<one label copied exactly from the list above>"}'
    )


# The audio placeholder is inserted by to_swift_intent.py (before or after this
# text depending on the model family), so it is intentionally NOT in here.
USER_INSTRUCTION = (
    "Listen to the utterance and identify its intent.\n"
    'Respond ONLY with a single-line JSON object: {"intent": "<label>"}'
)
