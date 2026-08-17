#!/bin/bash
# ============================================================================
# Find the best surviving checkpoint from an ms-swift SFT output dir.
#
# Why this is needed rather than just taking the last checkpoint: the sft
# runs in this project use load_best_model_at_end=false (swift never
# auto-restores a "best" checkpoint) and save_total_limit=2 (only the last 2
# checkpoint dirs survive on disk - everything earlier gets deleted as
# training progresses). So "best" must be reconstructed from the eval_loss
# history and then checked against what actually still exists on disk.
#
# Each checkpoint-N/trainer_state.json holds the *cumulative* log_history
# from step 0, so reading just the highest-numbered checkpoint's file is
# enough to see every eval_loss ever recorded during the run - no need to
# open all of them.
#
# Plain shell script, NOT a Slurm job - reading a small JSON file needs no
# GPU/conda env, just a system python3. Run directly on the login node:
#
# Usage:
#   scripts/training/find_best_checkpoint.sh $PROJ/models/gemma/gemma_4_e4b_intent_lora
#
# Then merge the printed checkpoint:
#   sbatch --export=ALL,CKPT=<printed path> scripts/training/merge_lora.sh
# ============================================================================
set -e

OUTDIR="${1:?Usage: $0 <sft_output_dir>}"

if [ ! -d "$OUTDIR" ]; then
    echo "FATAL: $OUTDIR not found" >&2
    exit 1
fi

python3 - "$OUTDIR" <<'EOF'
import json
import re
import sys
from pathlib import Path

outdir = Path(sys.argv[1])
ckpts = sorted(outdir.glob("checkpoint-*"),
               key=lambda p: int(re.search(r"checkpoint-(\d+)", p.name).group(1)))
if not ckpts:
    sys.exit(f"No checkpoint-* dirs found under {outdir}")

last = ckpts[-1]
state_path = last / "trainer_state.json"
if not state_path.exists():
    sys.exit(f"{state_path} missing - can't read eval history")

state = json.loads(state_path.read_text())
evals = [(e["step"], e["eval_loss"]) for e in state["log_history"] if "eval_loss" in e]
surviving_steps = {int(re.search(r"checkpoint-(\d+)", p.name).group(1)) for p in ckpts}

if not evals:
    sys.exit("No eval_loss entries in log_history - was --eval_steps/--split_dataset_ratio set?")

print(f"{'step':>8}  {'eval_loss':>10}  on_disk")
for step, loss in evals:
    marker = "yes" if step in surviving_steps else "no (rotated out by save_total_limit)"
    print(f"{step:>8}  {loss:>10.4f}  {marker}")

global_best_step, global_best_loss = min(evals, key=lambda x: x[1])
surviving_evals = [(s, l) for s, l in evals if s in surviving_steps]
best_step, best_loss = min(surviving_evals, key=lambda x: x[1])

print()
if best_step != global_best_step:
    print(f"NOTE: the global-best eval_loss ({global_best_loss:.4f} @ step {global_best_step}) "
          "was already deleted by save_total_limit rotation.")
    print(f"Best SURVIVING checkpoint: step {best_step}, eval_loss={best_loss:.4f}")
else:
    print(f"Best checkpoint: step {best_step}, eval_loss={best_loss:.4f} (also the global best)")

best_path = outdir / f"checkpoint-{best_step}"
print(f"\n{best_path}")
print(f"\nMerge it with:\n  sbatch --export=ALL,CKPT={best_path} scripts/training/merge_lora.sh")
EOF
