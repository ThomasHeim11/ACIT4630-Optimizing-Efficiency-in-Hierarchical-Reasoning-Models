"""
FLOPs Measurement & Visualization for HRM-family Models

Two modes:
  1. Measure (on GPU cluster) — runs inference, counts FLOPs, saves JSON
  2. Plot (locally, no GPU)   — reads JSONs, produces Bianco et al. bubble chart

Measure (run from each model's directory):
  cd ~/HMR/models/SHREK-HRM
  DISABLE_COMPILE=1 python3 ../../flops/flops.py measure \
      --checkpoint <CKPT> --name "SHREK Large" --task sudoku

Plot (no GPU needed):
  python3 flops/flops.py plot --results-dir flops/results

Reference: Bianco et al., Benchmark Analysis of Representative Deep Neural
Network Architectures, IEEE Access, 2018.
"""

import argparse
import os
import sys
import json
import glob as globmod

# =====================================================================
# Measure mode (GPU required)
# =====================================================================

def resolve_checkpoint(path):
    """Accept a checkpoint file or directory; if directory, pick the latest step_* file.

    If the directory has no step_* files but has exactly one subdirectory that does,
    descend into it (handles auto-generated run_name directories).
    """
    import re
    if not os.path.isdir(path):
        return path

    def find_latest_step(d):
        steps = []
        for f in os.listdir(d):
            m = re.match(r"step_(\d+)$", f)
            if m:
                steps.append((int(m.group(1)), os.path.join(d, f)))
        return steps

    steps = find_latest_step(path)
    if not steps:
        # Look one level deeper (e.g. checkpoints/project_name/auto_run_name/)
        subdirs = [os.path.join(path, d) for d in os.listdir(path)
                    if os.path.isdir(os.path.join(path, d))]
        for sd in subdirs:
            steps = find_latest_step(sd)
            if steps:
                print(f"  Found checkpoints in subdirectory: {sd}")
                break
    if not steps:
        raise FileNotFoundError(f"No step_* checkpoints found in {path} (or subdirs)")

    steps.sort()
    chosen = steps[-1][1]
    print(f"  Auto-selected latest checkpoint: {chosen}")
    return chosen


def load_model(checkpoint_path, model_dir=None):
    """Load model from checkpoint and its config."""
    import yaml
    import torch
    if model_dir:
        sys.path.insert(0, os.path.abspath(model_dir))
    from pretrain import PretrainConfig, init_train_state, create_dataloader

    checkpoint_path = resolve_checkpoint(checkpoint_path)
    checkpoint_dir = os.path.dirname(checkpoint_path)
    config_path = os.path.join(checkpoint_dir, "all_config.yaml")

    with open(config_path, "r") as f:
        config = PretrainConfig(**yaml.safe_load(f))

    # Resolve relative data paths (they're relative to model_dir)
    base = os.path.abspath(model_dir) if model_dir else os.getcwd()
    if hasattr(config, "data_path") and config.data_path and not os.path.isabs(config.data_path):
        config.data_path = os.path.normpath(os.path.join(base, config.data_path))
    if hasattr(config, "data_paths") and config.data_paths:
        config.data_paths = [
            os.path.normpath(os.path.join(base, p)) if not os.path.isabs(p) else p
            for p in config.data_paths
        ]

    torch.random.manual_seed(config.seed)

    train_loader, train_metadata = create_dataloader(
        config, "train",
        test_set_mode=False,
        epochs_per_iter=1,
        global_batch_size=1,
        rank=0,
        world_size=1
    )

    train_state = init_train_state(config, train_metadata, world_size=1)

    checkpoint_data = torch.load(checkpoint_path, map_location="cuda")
    try:
        train_state.model.load_state_dict(checkpoint_data, assign=True)
    except RuntimeError:
        cleaned = {k.removeprefix("_orig_mod."): v for k, v in checkpoint_data.items()}
        train_state.model.load_state_dict(cleaned, assign=True)

    train_state.model.eval()
    return train_state.model, config


def measure_flops(model, config, num_samples=100, batch_size=10):
    """Measure FLOPs per puzzle and average reasoning steps."""
    import torch
    import numpy as np
    from torch.utils.flop_counter import FlopCounterMode

    # HRM/SHREK use config.data_path (str), TRM uses config.data_paths (list)
    data_path = getattr(config, "data_path", None) or config.data_paths[0]

    all_inputs = torch.from_numpy(
        np.load(f"{data_path}/test/all__inputs.npy")
    ).long().cuda()
    all_labels = torch.from_numpy(
        np.load(f"{data_path}/test/all__labels.npy")
    ).long().cuda()

    num_batches = num_samples // batch_size
    total_flops = 0
    total_steps = 0
    total_puzzles = 0
    total_correct = 0

    with torch.no_grad():
        for idx in range(num_batches):
            start = idx * batch_size
            end = start + batch_size

            batch_inputs = all_inputs[start:end]
            batch_labels = all_labels[start:end]

            batch = {
                "inputs": batch_inputs,
                "labels": batch_labels,
                "puzzle_identifiers": torch.zeros(batch_size, dtype=torch.long, device="cuda")
            }

            carry = model.initial_carry(batch_size, batch_inputs.device)

            flop_counter = FlopCounterMode(display=False)
            with flop_counter:
                out = model(carry=carry, batch=batch, return_keys=[])
            # Forward returns vary: 5 vals (HRM/TRM), 6 (Augmented/SHREK), 7 (with trace)
            # metrics is always the 3rd element (index 2) when carry is first,
            # or index 3 when trace is first
            if hasattr(out[0], 'halted'):
                # carry is first: (carry, loss, metrics, ...)
                metrics = out[2]
            else:
                # trace is first: (trace, carry, loss, metrics, ...)
                metrics = out[3]

            batch_flops = flop_counter.get_total_flops()
            total_flops += batch_flops

            if "steps" in metrics:
                total_steps += metrics["steps"].item()
            if "exact_accuracy" in metrics:
                total_correct += metrics["exact_accuracy"].item()

            total_puzzles += batch_size

            print(f"  Batch {idx+1}/{num_batches}: "
                  f"FLOPs={batch_flops/batch_size:.2e}/puzzle, "
                  f"Steps={metrics.get('steps', torch.tensor(0)).item()/batch_size:.1f}")

    avg_flops = total_flops / total_puzzles
    avg_steps = total_steps / total_puzzles
    accuracy = total_correct / total_puzzles

    return {
        "avg_flops_per_puzzle": avg_flops,
        "avg_gflops_per_puzzle": avg_flops / 1e9,
        "avg_steps": avg_steps,
        "exact_accuracy": accuracy,
        "num_puzzles": total_puzzles,
        "total_flops": total_flops,
    }


def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def run_measure(args):
    """Run FLOPs measurement on GPU and save results JSON."""
    import torch

    torch.cuda.set_device(0)

    print(f"Loading model from: {args.checkpoint}")
    print(f"Model directory: {args.model_dir}")
    model, config = load_model(args.checkpoint, model_dir=args.model_dir)

    total_params, trainable_params = count_parameters(model)
    data_path = getattr(config, "data_path", None) or config.data_paths[0]
    print(f"Parameters: {total_params:,} total, {trainable_params:,} trainable")
    print(f"Data path: {data_path}")
    print(f"Measuring FLOPs on {args.num_samples} test puzzles...")
    print("=" * 60)

    results = measure_flops(model, config, args.num_samples, args.batch_size)

    print("=" * 60)
    print(f"Model:                 {args.name}")
    print(f"Task:                  {args.task}")
    print(f"Parameters:            {total_params:,}")
    print(f"Puzzles evaluated:     {results['num_puzzles']}")
    print(f"Exact accuracy:        {results['exact_accuracy']:.4f}")
    print(f"Avg reasoning steps:   {results['avg_steps']:.2f}")
    print(f"Avg GFLOPs per puzzle: {results['avg_gflops_per_puzzle']:.4f}")
    print("=" * 60)

    output = {
        "name": args.name,
        "task": args.task,
        "checkpoint": args.checkpoint,
        "parameters": total_params,
        "num_puzzles": results["num_puzzles"],
        "exact_accuracy": results["exact_accuracy"],
        "avg_steps": results["avg_steps"],
        "avg_flops_per_puzzle": results["avg_flops_per_puzzle"],
        "avg_gflops_per_puzzle": results["avg_gflops_per_puzzle"],
        "total_flops": results["total_flops"],
    }

    # Save to central results directory
    os.makedirs(args.results_dir, exist_ok=True)
    safe_name = args.name.lower().replace(" ", "_")
    output_path = os.path.join(args.results_dir, f"{safe_name}_{args.task}.json")
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"Results saved to: {output_path}")


# =====================================================================
# Plot mode (no GPU needed)
# =====================================================================

MODEL_COLORS = {
    "Original HRM":  "#e74c3c",
    "Augmented HRM": "#f39c12",
    "SHREK Large":   "#3498db",
    "SHREK Tiny":    "#1abc9c",
    "TRM Attention":  "#9b59b6",
    "TRM MLP":        "#2ecc71",
}


def make_chart(models, title, filename):
    """Bianco et al. style bubble chart: GFLOPs vs Accuracy, bubble size = params."""
    import matplotlib.pyplot as plt
    import numpy as np

    fig, ax = plt.subplots(figsize=(12, 8))

    # Collect param range for scaling bubbles
    all_params = [m["parameters"] for m in models]
    min_p, max_p = min(all_params), max(all_params)

    for m in models:
        color = MODEL_COLORS.get(m["name"], "#7f8c8d")
        # Scale bubble area: smallest model ~150, largest ~800
        if max_p > min_p:
            norm = (m["parameters"] - min_p) / (max_p - min_p)
        else:
            norm = 0.5
        size = 150 + norm * 650

        ax.scatter(
            m["avg_gflops_per_puzzle"],
            m["exact_accuracy"] * 100,
            s=size, c=color, alpha=0.75,
            edgecolors="black", linewidth=1.2, zorder=5,
        )

        # Label offset — push "Original HRM" down to avoid overlap
        ox, oy = 14, 10
        if "Original" in m["name"]:
            oy = -18
        ax.annotate(
            m["name"],
            (m["avg_gflops_per_puzzle"], m["exact_accuracy"] * 100),
            textcoords="offset points", xytext=(ox, oy),
            fontsize=11, fontweight="bold", color=color,
        )

    ax.set_xlabel("Operations [G-FLOPs]", fontsize=14)
    ax.set_ylabel("Test Exact Accuracy [%]", fontsize=14)
    ax.set_title(title, fontsize=16, fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.tick_params(labelsize=12)

    # Parameter-size legend (fixed reference bubbles)
    param_vals = sorted(set(all_params))
    # Pick up to 3 representative sizes
    if len(param_vals) >= 3:
        legend_params = [param_vals[0], param_vals[len(param_vals)//2], param_vals[-1]]
    else:
        legend_params = param_vals
    legend_handles = []
    legend_labels = []
    for p in legend_params:
        if max_p > min_p:
            norm = (p - min_p) / (max_p - min_p)
        else:
            norm = 0.5
        s = 150 + norm * 650
        h = ax.scatter([], [], s=s, c="gray", alpha=0.4, edgecolors="black", linewidth=1)
        legend_handles.append(h)
        if p >= 1e6:
            legend_labels.append(f"{p/1e6:.0f}M")
        else:
            legend_labels.append(f"{p/1e3:.0f}K")

    ax.legend(
        legend_handles, legend_labels,
        loc="lower right", title="Parameters", title_fontsize=11,
        fontsize=10, framealpha=0.9,
    )

    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches="tight")
    print(f"Chart saved to: {filename}")

    # Print summary table
    print(f"\n{'Model':<20} {'Params':>8} {'Steps':>6} {'GFLOPs':>8} {'Accuracy':>9}")
    print("-" * 55)
    for m in sorted(models, key=lambda x: x["avg_gflops_per_puzzle"]):
        print(f"{m['name']:<20} {m['parameters']/1e6:>6.1f}M "
              f"{m['avg_steps']:>6.1f} {m['avg_gflops_per_puzzle']:>8.2f} "
              f"{m['exact_accuracy']*100:>8.1f}%")


def run_plot(args):
    """Read all result JSONs and produce comparative charts."""
    json_files = globmod.glob(os.path.join(args.results_dir, "*.json"))
    if not json_files:
        print(f"No JSON result files found in {args.results_dir}")
        sys.exit(1)

    # Load and group by task
    by_task = {}
    for path in json_files:
        with open(path) as f:
            data = json.load(f)
        task = data.get("task", "unknown")
        by_task.setdefault(task, []).append(data)

    os.makedirs(args.output_dir, exist_ok=True)

    for task, models in sorted(by_task.items()):
        title = f"{task.replace('_', ' ').title()}: Accuracy vs Computational Cost"
        filename = os.path.join(args.output_dir, f"{task}_accuracy_vs_flops.png")
        print(f"\n{'='*60}")
        print(f"  {task.upper()} ({len(models)} models)")
        print(f"{'='*60}")
        make_chart(models, title, filename)

    # Save combined JSON
    combined_path = os.path.join(args.output_dir, "all_results.json")
    with open(combined_path, "w") as f:
        json.dump(by_task, f, indent=2)
    print(f"\nCombined results saved to: {combined_path}")


# =====================================================================
# CLI
# =====================================================================

def main():
    parser = argparse.ArgumentParser(
        description="FLOPs measurement & Bianco-style visualization for HRM models"
    )
    sub = parser.add_subparsers(dest="command")

    # -- measure --
    m = sub.add_parser("measure", help="Measure FLOPs on GPU (run on cluster)")
    m.add_argument("--checkpoint", required=True, help="Path to model checkpoint")
    m.add_argument("--model-dir", required=True, help="Path to model source (contains pretrain.py)")
    m.add_argument("--name", required=True, help="Model name (e.g. 'SHREK Large')")
    m.add_argument("--task", required=True, help="Task name (e.g. 'sudoku', 'maze')")
    m.add_argument("--num-samples", type=int, default=100, help="Test puzzles to measure")
    m.add_argument("--batch-size", type=int, default=10, help="Batch size")
    m.add_argument("--results-dir", default="../../flops/results",
                    help="Directory to save result JSONs")

    # -- plot --
    p = sub.add_parser("plot", help="Generate comparative charts (no GPU needed)")
    p.add_argument("--results-dir", default="flops/results",
                    help="Directory containing result JSONs")
    p.add_argument("--output-dir", default="flops",
                    help="Directory to save charts")

    args = parser.parse_args()

    if args.command == "measure":
        run_measure(args)
    elif args.command == "plot":
        run_plot(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
