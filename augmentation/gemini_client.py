#!/usr/bin/env python3
"""Shared Gemini client for the augmentation pipeline (new google-genai SDK).

Key loading order: $GOOGLE_API_KEY, else parsed from the project's known env
file (same one pred_gemini_multimodal.py uses). Models are env-overridable:
  GEN_MODEL   (default gemini-3.6-flash)  - bulk generation, cheap + diverse
  JUDGE_MODEL (default gemini-3.1-pro)    - quality judging, different model
                                            family from the generator on purpose
All calls request JSON output and go through retry with exponential backoff.
"""
import json
import os
import re
import time

ENV_FILE = "${HOME}/.config/gemini.env"
# Verified by DIRECT calls on this key 2026-07-25 (models.list() lags):
# gemini-3.6-flash WORKS (released 2026-07-21), gemini-3.1-pro-preview WORKS.
GEN_MODEL = os.environ.get("GEN_MODEL", "gemini-3.6-flash")
JUDGE_MODEL = os.environ.get("JUDGE_MODEL", "gemini-3.1-pro-preview")
# multi-annotator panel (judge.py): comma-separated, majority vote
JUDGE_MODELS = [m.strip() for m in os.environ.get(
    "JUDGE_MODELS",
    "gemini-3.1-pro-preview,gemini-3.6-flash,gemini-2.5-pro").split(",") if m.strip()]

_client = None


def _load_key():
    key = os.environ.get("GOOGLE_API_KEY")
    if key:
        return key
    if os.path.exists(ENV_FILE):
        for line in open(ENV_FILE):
            line = line.strip()
            if line.startswith("GOOGLE_API_KEY"):
                return line.split("=", 1)[1].strip().strip('"').strip("'")
    raise SystemExit(
        f"FATAL: GOOGLE_API_KEY not set and not found in {ENV_FILE}")


def client():
    global _client
    if _client is None:
        from google import genai
        _client = genai.Client(api_key=_load_key())
    return _client


def extract_json(text):
    """Parse a JSON value from model text, tolerating code fences / prose."""
    text = text.strip()
    m = re.search(r"```(?:json)?\s*(.*?)```", text, re.S)
    if m:
        text = m.group(1).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # last resort: first [...] or {...} block
    for opener, closer in (("[", "]"), ("{", "}")):
        i = text.find(opener)
        j = text.rfind(closer)
        if i != -1 and j > i:
            try:
                return json.loads(text[i:j + 1])
            except json.JSONDecodeError:
                continue
    raise ValueError(f"no JSON found in response: {text[:200]!r}")


def generate_json(model, prompt, temperature=1.0, max_retries=5,
                  request_delay=0.1):
    """One JSON-mode call; returns the parsed JSON value."""
    from google.genai import types
    cli = client()
    last = None
    for attempt in range(max_retries):
        try:
            resp = cli.models.generate_content(
                model=model,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=temperature,
                    response_mime_type="application/json",
                ),
            )
            time.sleep(request_delay)
            return extract_json(resp.text)
        except Exception as e:  # rate limits, transient 5xx, parse errors
            last = e
            wait = min(2 ** attempt * 2, 60)
            print(f"    retry {attempt + 1}/{max_retries} after error: "
                  f"{str(e)[:140]} (sleep {wait}s)")
            time.sleep(wait)
    raise RuntimeError(f"gemini call failed after {max_retries} tries: {last}")
