#!/usr/bin/env python
"""Manual LoRA merge, bypassing PeftModel.from_pretrained's merge path.

swift export --merge_lora crashes on this project's env (transformers>=5.8.0
refactored WeightTransform from a dataclass to a plain class, so
distributed_operation is no longer a constructor kwarg - peft 0.19.1, latest
on PyPI as of this writing, still calls it as one). See
logs/setup/check_peft_tf_*.out for the confirmed root cause.

This script does the same arithmetic PeftModel's merge does
(W_merged = W + scaling * lora_B @ lora_A, scaling = alpha/r or
alpha/sqrt(r) for rslora) directly on the base model's safetensors shards,
with no PeftModel/transformers model class involved. It refuses to run (raises)
on adapter shapes it can't handle correctly (DoRA, rank/alpha overrides,
fan_in_fan_out, modules_to_save, MoE/expert target modules) rather than
silently producing a wrong merge - extend the relevant function first if you
hit one of those.
"""
import argparse
import json
import math
import re
import shutil
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

LORA_KEY_RE = re.compile(r"^(?P<base>.+)\.lora_(?P<ab>[AB])(?:\.[^.]+)?\.weight$")


def strip_adapter_prefix(base_key: str) -> str:
    prefix = "base_model.model."
    if not base_key.startswith(prefix):
        raise ValueError(f"unexpected adapter key prefix (expected '{prefix}...'): {base_key}")
    return base_key[len(prefix):]


def load_adapter_state(adapter_dir: Path):
    st_path = adapter_dir / "adapter_model.safetensors"
    if not st_path.exists():
        raise FileNotFoundError(f"no adapter_model.safetensors under {adapter_dir}")
    state = {}
    with safe_open(str(st_path), framework="pt", device="cpu") as f:
        for key in f.keys():
            state[key] = f.get_tensor(key)
    return state


def build_lora_deltas(adapter_state: dict, config: dict) -> dict:
    r = config["r"]
    alpha = config["lora_alpha"]
    use_rslora = config.get("use_rslora", False)
    use_dora = config.get("use_dora", False)
    fan_in_fan_out = config.get("fan_in_fan_out", False)
    rank_pattern = config.get("rank_pattern") or {}
    alpha_pattern = config.get("alpha_pattern") or {}

    if use_dora:
        raise NotImplementedError(
            "adapter_config.json has use_dora=true - DoRA merges also need a "
            "per-output-channel magnitude vector on top of B@A. This script only "
            "implements plain LoRA merge; extend build_lora_deltas() before trusting "
            "its output on a DoRA adapter."
        )
    if fan_in_fan_out:
        raise NotImplementedError(
            "adapter_config.json has fan_in_fan_out=true (Conv1D-style layout) - "
            "this script assumes nn.Linear's [out_features, in_features] layout and "
            "does not transpose. Not expected for Qwen3-Omni attention projections; "
            "verify target_modules before extending."
        )
    if rank_pattern or alpha_pattern:
        raise NotImplementedError(
            f"adapter_config.json has per-layer overrides "
            f"(rank_pattern={rank_pattern}, alpha_pattern={alpha_pattern}) - this "
            f"script only applies one uniform r/alpha to every target module. "
            f"Extend build_lora_deltas() to look up the per-module value first."
        )

    pairs: dict[str, dict[str, torch.Tensor]] = {}
    for key, tensor in adapter_state.items():
        m = LORA_KEY_RE.match(key)
        if not m:
            continue  # e.g. modules_to_save entries - handled by the caller
        pairs.setdefault(m.group("base"), {})[m.group("ab")] = tensor

    scaling = alpha / math.sqrt(r) if use_rslora else alpha / r
    print(f"scaling = {scaling} (r={r}, alpha={alpha}, use_rslora={use_rslora})")

    deltas = {}
    for base, ab in pairs.items():
        if "A" not in ab or "B" not in ab:
            raise ValueError(f"incomplete LoRA pair for {base}: found {sorted(ab)}")
        lora_a = ab["A"].to(torch.float32)  # [r, in_features]
        lora_b = ab["B"].to(torch.float32)  # [out_features, r]
        delta = (lora_b @ lora_a) * scaling  # [out_features, in_features]
        target_key = strip_adapter_prefix(base) + ".weight"
        deltas[target_key] = delta
    return deltas


def collect_modules_to_save(config: dict) -> dict:
    modules_to_save = config.get("modules_to_save") or []
    if not modules_to_save:
        return {}
    raise NotImplementedError(
        f"adapter_config.json declares modules_to_save={modules_to_save} - this "
        f"script only handles LoRA A/B deltas, not full-weight replacement modules. "
        f"Inspect adapter_model.safetensors's actual keys for these before extending "
        f"collect_modules_to_save()."
    )


def merge(base_dir: Path, adapter_dir: Path, output_dir: Path) -> None:
    config = json.loads((adapter_dir / "adapter_config.json").read_text())
    print(
        f"adapter_config: r={config['r']} alpha={config['lora_alpha']} "
        f"use_rslora={config.get('use_rslora', False)} use_dora={config.get('use_dora', False)} "
        f"target_modules={config.get('target_modules')}"
    )

    target_modules = config.get("target_modules") or []
    suspicious = [m for m in target_modules if "expert" in m.lower() or "moe" in m.lower()]
    if suspicious:
        raise NotImplementedError(
            f"target_modules includes what look like MoE/expert layers: {suspicious}. "
            f"Those need transformers' block-diagonal PeftConcatenate merge, not the "
            f"plain B@A this script implements - do not trust this script's output for "
            f"them without extending it."
        )

    adapter_state = load_adapter_state(adapter_dir)
    deltas = build_lora_deltas(adapter_state, config)
    replacements = collect_modules_to_save(config)
    print(f"found {len(deltas)} LoRA deltas to merge, {len(replacements)} full-weight replacements")
    if not deltas:
        raise RuntimeError("no LoRA deltas found in adapter_model.safetensors - refusing to write a no-op 'merge'")

    index_path = base_dir / "model.safetensors.index.json"
    if not index_path.exists():
        raise FileNotFoundError(
            f"no model.safetensors.index.json under {base_dir} - single-shard model? "
            f"extend this script to handle a bare model.safetensors."
        )
    index = json.loads(index_path.read_text())
    weight_map = index["weight_map"]
    shard_files = sorted(set(weight_map.values()))

    output_dir.mkdir(parents=True, exist_ok=True)
    merged_keys_seen = set()

    for shard_name in shard_files:
        src_shard = base_dir / shard_name
        dst_shard = output_dir / shard_name
        print(f"processing shard {shard_name} ...")
        out_tensors = {}
        with safe_open(str(src_shard), framework="pt", device="cpu") as f:
            for key in f.keys():
                tensor = f.get_tensor(key)
                if key in deltas:
                    merged = tensor.to(torch.float32) + deltas[key]
                    out_tensors[key] = merged.to(tensor.dtype).contiguous()
                    merged_keys_seen.add(key)
                elif key in replacements:
                    out_tensors[key] = replacements[key].to(tensor.dtype).contiguous()
                    merged_keys_seen.add(key)
                else:
                    out_tensors[key] = tensor
        save_file(out_tensors, str(dst_shard), metadata={"format": "pt"})

    missing = (set(deltas) | set(replacements)) - merged_keys_seen
    if missing:
        raise RuntimeError(
            f"{len(missing)} LoRA target keys were never found in the base model's "
            f"safetensors shards - the adapter-key-to-base-key naming assumption in "
            f"strip_adapter_prefix() is probably wrong for this model. Merge is "
            f"INCOMPLETE, refusing to call this done. Missing keys (first 10): "
            f"{sorted(missing)[:10]}"
        )
    print(f"merged {len(merged_keys_seen)} tensors across {len(shard_files)} shards")

    shutil.copy2(index_path, output_dir / "model.safetensors.index.json")
    for src in base_dir.iterdir():
        if src.name.endswith(".safetensors") or src.name == "model.safetensors.index.json":
            continue
        dst = output_dir / src.name
        if src.is_dir():
            shutil.copytree(src, dst, dirs_exist_ok=True)
        else:
            shutil.copy2(src, dst)

    print(f"done -> {output_dir}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base-dir", required=True, type=Path, help="local HF snapshot dir of the base model")
    ap.add_argument("--adapter-dir", required=True, type=Path, help="LoRA checkpoint dir (adapter_config.json + adapter_model.safetensors)")
    ap.add_argument("--output-dir", required=True, type=Path)
    args = ap.parse_args()
    merge(args.base_dir, args.adapter_dir, args.output_dir)
