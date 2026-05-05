"""
SHREK-HRM submission test script.

Runs evaluation on every (model, task) listed in config.yaml and prints an
accuracy table that matches the report (Tables II, III).

Behavior
--------
1. Reads paths and HuggingFace repo IDs from config.yaml — no hardcoded paths.
2. Downloads checkpoints from HuggingFace if the local `model/` dir is empty.
3. Downloads test data from HuggingFace if the local `data/` dir is empty.
4. For each evaluation, calls that model's evaluate.py with the right checkpoint.
5. Parses `all/exact_accuracy` from each run's stdout and prints a summary.

Hardware
--------
Requires NVIDIA GPU with CUDA 12.6 (the model uses flash-attn).
Total runtime: roughly 10-20 minutes on a single GH200.

Usage
-----
    python test.py
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


def ensure_dir_populated(local_dir: Path, hf_repo: str, repo_type: str = "model"):
    """Download `hf_repo` from HuggingFace into `local_dir` if it's empty/missing."""
    local_dir.mkdir(parents=True, exist_ok=True)
    if any(local_dir.iterdir()):
        print(f"[ok] {local_dir} already populated, skipping download.")
        return

    print(f"[download] {hf_repo} -> {local_dir} (this may take a while)")
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        sys.exit(
            "ERROR: huggingface_hub not installed.\n"
            "    pip install huggingface_hub\n"
            "Then re-run test.py."
        )

    snapshot_download(
        repo_id=hf_repo,
        repo_type=repo_type,
        local_dir=str(local_dir),
    )


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

    # evaluate.py prints the metrics dict at the end. Match the exact_accuracy entry.
    m = re.search(r"['\"]?all[/\.]exact_accuracy['\"]?\s*[:=]\s*([\d\.eE+-]+)", output)
    if m:
        return float(m.group(1))

    print("  [warn] could not parse exact_accuracy from evaluate.py output")
    print("  --- last 20 lines of output ---")
    for line in output.splitlines()[-20:]:
        print(f"    {line}")
    return None


def print_results(results):
    print()
    print("=" * 88)
    print(f"{'Model':<22} {'Task':<18} {'Expected':>10} {'Actual':>10} {'Δ':>9} {'Status':>8}")
    print("-" * 88)
    for r in results:
        if r["actual"] is None:
            print(
                f"{r['name']:<22} {r['task']:<18} {r['expected']:>9.1%} "
                f"{'n/a':>10} {'n/a':>9} {'FAIL':>8}"
            )
            continue
        delta = r["actual"] - r["expected"]
        status = "OK" if abs(delta) < 0.05 else "OFF"
        print(
            f"{r['name']:<22} {r['task']:<18} {r['expected']:>9.1%} "
            f"{r['actual']:>9.1%} {delta:>+9.1%} {status:>8}"
        )
    print("=" * 88)


def main():
    cfg = load_config()

    model_dir = REPO_ROOT / cfg["model_dir"]
    data_dir = REPO_ROOT / cfg["data_dir"]

    # 1. Ensure checkpoints + data are present (download from HF on first run).
    ensure_dir_populated(model_dir, cfg["hf_checkpoints_repo"], repo_type="model")
    ensure_dir_populated(data_dir, cfg["hf_dataset_repo"], repo_type="dataset")

    # 2. Run each evaluation.
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

    # 3. Summary.
    print_results(results)

    # Exit non-zero if any eval failed (so the grader / CI can detect).
    if any(r["actual"] is None for r in results):
        sys.exit(1)


if __name__ == "__main__":
    main()
