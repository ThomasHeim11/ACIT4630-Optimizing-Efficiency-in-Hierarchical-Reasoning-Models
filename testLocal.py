"""
Local variant of test.py — uses checkpoints already present in `checkpoints/`
at the repo root, no HuggingFace downloads.

Reads the same config.yaml (paths, expected accuracies), but:
  - model_dir is hardcoded to `checkpoints/` (not `model/`)
  - data_dir is hardcoded to `data/` (must be already populated)
  - no HuggingFace download attempts

Useful for quick iteration when you have the checkpoints locally already
(e.g. after scp from cluster) without re-downloading 16 GB from HF.

Hardware
--------
Same as test.py — requires NVIDIA GPU with CUDA 12.6 (flash-attn).

Usage
-----
    python testLocal.py
"""

import os
import re
import subprocess
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent
CONFIG_PATH = REPO_ROOT / "config.yaml"


def load_config():
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


def evaluate_checkpoint(model_code_dir: Path, checkpoint_path: Path):
    """Run model_code_dir/evaluate.py on the given checkpoint, return exact_accuracy or None."""
    if not checkpoint_path.exists():
        print(f"[skip] checkpoint not found: {checkpoint_path}")
        return None

    cmd = [sys.executable, "evaluate.py", f"checkpoint={checkpoint_path}"]
    print(f"[eval] {checkpoint_path}")
    result = subprocess.run(
        cmd,
        cwd=str(model_code_dir),
        capture_output=True,
        text=True,
        env={**os.environ, "OMP_NUM_THREADS": "8"},
    )
    output = result.stdout + result.stderr

    m = re.search(r"['\"]exact_accuracy['\"]\s*:\s*([\d\.eE+-]+)", output)
    if m:
        return float(m.group(1))

    print("  [warn] could not parse exact_accuracy from evaluate.py output")
    print("  --- last 20 lines of output ---")
    for line in output.splitlines()[-20:]:
        print(f"    {line}")
    return None


def print_results(results):
    print()
    print("=" * 60)
    print(f"{'Model':<22} {'Task':<18} {'Accuracy':>10}")
    print("-" * 60)
    for r in results:
        actual = f"{r['actual']:.1%}" if r["actual"] is not None else "n/a"
        print(f"{r['name']:<22} {r['task']:<18} {actual:>10}")
    print("=" * 60)


def main():
    cfg = load_config()

    # Use local `checkpoints/` and `data/` directly — no HF downloads.
    model_dir = REPO_ROOT / "checkpoints"
    data_dir = REPO_ROOT / "data"

    if not model_dir.exists() or not any(model_dir.iterdir()):
        sys.exit(
            f"ERROR: {model_dir} is empty or missing.\n"
            "Populate it from cluster (scp) or run test.py once to download from HuggingFace."
        )
    if not data_dir.exists() or not any(data_dir.iterdir()):
        sys.exit(
            f"ERROR: {data_dir} is empty or missing.\n"
            "Populate it from cluster (scp) or run test.py once to download from HuggingFace."
        )

    results = []
    for entry in cfg["evaluations"]:
        ckpt = model_dir / entry["checkpoint_subpath"]
        actual = evaluate_checkpoint(REPO_ROOT / entry["model_code_dir"], ckpt)
        results.append({
            "name": entry["name"],
            "task": entry["task"],
            "expected": entry["expected_accuracy"],
            "actual": actual,
        })

    print_results(results)

    if any(r["actual"] is None for r in results):
        sys.exit(1)


if __name__ == "__main__":
    main()
