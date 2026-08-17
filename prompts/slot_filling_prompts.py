#!/usr/bin/env python3
"""Zero-shot prompt builders for NADI-2026 Subtask 5.2 (Slot Filling).

The SYSTEM_PROMPT text here is kept WORD-FOR-WORD identical to the reference
notebook (NADI_2026_Subtask_5_2_SLU_Slot_Filling_Omni3B_Inference*.ipynb) so
that these cluster runs are directly comparable to the Colab baseline. As with
the intent task, the model-family difference is only the position of the
`<audio>` placeholder in the user turn, not the prompt content.

The slot-label inventory is read from `data/slurptn/slot_labels.txt` (written
by extract_labels.py during data prep).
"""
from pathlib import Path


def load_labels(path):
    labels = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            labels.append(line)
    return sorted(set(labels))


def build_system_prompt(slot_labels):
    slot_label_list = ", ".join(f"<{label}>" for label in slot_labels)
    return (
        "You are an automatic speech recognition and spoken language understanding "
        "system for Tunisian Arabic (Tunisian dialect, written in Arabic script; "
        "code-switching with French/English words is common).\n\n"
        "Task: listen to the audio and output ONE line that is the exact spoken "
        "transcription, with semantic slots marked inline using this scheme:\n"
        "  <label> slot value >\n"
        "A slot opens with its label in angle brackets and closes with a lone '>'. "
        "Words outside any <label> ... > span are left as plain transcription.\n\n"
        f"Valid slot labels: {slot_label_list}\n\n"
        "Output only the annotated transcription line: no translation, no "
        "explanation, no surrounding quotes."
    )


# to_swift_slot_filling.py positions the <audio> placeholder around this text.
USER_INSTRUCTION = (
    "Transcribe the audio with inline semantic slots as instructed."
)
