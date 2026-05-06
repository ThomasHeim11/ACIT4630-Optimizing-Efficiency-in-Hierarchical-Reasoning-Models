"""
SHREK-HRM submission test script — fully self-contained.

Runs evaluation on every (model, task) listed in config.yaml and prints an
accuracy table that matches the report (Tables II, III).

Behavior
--------
1. Reads paths and HuggingFace repo IDs from config.yaml — no hardcoded paths.
2. Downloads checkpoints from HuggingFace if the local `model/` dir is empty.
3. Downloads test data from HuggingFace if the local `data/` dir is empty.
4. For each evaluation, spawns a fresh Python subprocess (`python -c <inline>`)
   that imports the matching model's `pretrain.py`, loads the checkpoint, runs
   evaluation, and prints the metrics dict. test.py contains the full eval
   logic — no external evaluate.py needed for any of the 3 model families
   (SHREK / HRM / TRM). The subprocess auto-detects API differences (TRM has
   extra `evaluators` and `cpu_group` args).
5. Parses `exact_accuracy` from each run's stdout and prints a summary table.

Hardware
--------
Requires NVIDIA GPU with CUDA 12.6 (the models use flash-attn).
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

# Disable HuggingFace's xet-based downloads. The xet client has known native-library
# issues on some systems (missing .so files); the regular HTTP download path is more
# reliable. Set before importing huggingface_hub so the env var is honored.
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

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


# Inline evaluation script — runs as `python -c` subprocess so each call gets a
# fresh CUDA context and clean module state. test.py only evaluates SHREK
# models, so this script targets SHREK's pretrain.py API directly.
INLINE_EVAL = r"""
import os, sys, yaml, torch
sys.path.insert(0, os.getcwd())

from pretrain import PretrainConfig, init_train_state, create_dataloader, evaluate as _evaluate

ckpt = sys.argv[1]
ckpt_dir = os.path.dirname(ckpt)
with open(os.path.join(ckpt_dir, 'all_config.yaml')) as f:
    config = PretrainConfig(**yaml.safe_load(f))
config.eval_save_outputs = []
config.checkpoint_path = ckpt_dir

train_loader, train_metadata = create_dataloader(
    config, 'train',
    test_set_mode=False, epochs_per_iter=1,
    global_batch_size=config.global_batch_size,
    rank=0, world_size=1,
)
eval_loader, eval_metadata = create_dataloader(
    config, 'test',
    test_set_mode=True, epochs_per_iter=1,
    global_batch_size=config.global_batch_size,
    rank=0, world_size=1,
)

train_state = init_train_state(config, train_metadata, world_size=1)

# Load checkpoint, unwrap torch.compile prefix if present.
state = torch.load(ckpt, map_location='cuda')
try:
    train_state.model.load_state_dict(state, assign=True)
except Exception:
    train_state.model.load_state_dict(
        {k.removeprefix('_orig_mod.'): v for k, v in state.items()},
        assign=True,
    )

train_state.step = 0
fname = os.path.basename(ckpt)
if fname.startswith('step_'):
    train_state.step = int(fname.removeprefix('step_'))

train_state.model.eval()
print('Starting evaluation', flush=True)

metrics = _evaluate(
    config, train_state, eval_loader, eval_metadata,
    rank=0, world_size=1,
)

if metrics is not None:
    print(metrics)
"""


def evaluate_checkpoint(model_code_dir: Path, checkpoint_path: Path):
    """Run an inline `python -c` subprocess that evaluates the checkpoint.
    Returns the parsed `exact_accuracy` (float in [0, 1]) or None on failure.
    """
    if not checkpoint_path.exists():
        print(f"[skip] checkpoint not found: {checkpoint_path}")
        return None

    print(f"[eval] {checkpoint_path}")
    result = subprocess.run(
        [sys.executable, "-u", "-c", INLINE_EVAL, str(checkpoint_path)],
        cwd=str(model_code_dir),
        capture_output=True,
        text=True,
        env={**os.environ, "OMP_NUM_THREADS": "8"},
    )
    output = result.stdout + result.stderr

    # The metrics dict prints as Python repr ending with
    # `{'all': {..., 'exact_accuracy': Y, ...}}`. Match the value directly.
    m = re.search(r"['\"]exact_accuracy['\"]\s*:\s*([\d\.eE+-]+)", output)
    if m:
        return float(m.group(1))

    print("  [warn] could not parse exact_accuracy from inline eval output")
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

    model_dir = REPO_ROOT / cfg["model_dir"]
    data_dir = REPO_ROOT / cfg["data_dir"]

    # 1. Ensure checkpoints + data are present (download from HF on first run).
    ensure_dir_populated(model_dir, cfg["hf_checkpoints_repo"], repo_type="model")
    ensure_dir_populated(data_dir, cfg["hf_dataset_repo"], repo_type="dataset")

    # 2. Run each evaluation. Sleep briefly between subprocesses so the GPU
    #    driver fully releases state before the next evaluate.py spawns —
    #    consecutive CUDA loads without a pause have caused numpy/torch
    #    import failures on this cluster.
    import time
    results = []
    for i, entry in enumerate(cfg["evaluations"]):
        if i > 0:
            time.sleep(5)
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
