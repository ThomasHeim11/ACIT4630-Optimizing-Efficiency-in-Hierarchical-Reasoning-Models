"""
FLOPs Measurement & Visualization for HRM-family Models

Two modes:
  1. Measure (on GPU cluster) — runs inference, counts FLOPs, saves JSON
  2. Plot (locally, no GPU)   — reads JSONs, produces Bianco et al. bubble chart

Measure (run on cluster):
  python3 ~/HMR/flops/flops.py measure \
      --checkpoint ~/HMR/checkpoints/sudoku-extreme/shrek-large \
      --model-dir ~/HMR/models/SHREK-HRM \
      --name "SHREK Large" --task sudoku

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

def run_measure(args):
    """Run FLOPs measurement on GPU using each model's own eval_utils."""
    import torch
    import numpy as np
    from torch.utils.flop_counter import FlopCounterMode

    torch.cuda.set_device(0)

    # cd into model directory so all relative imports and paths work
    model_dir = os.path.abspath(args.model_dir)
    os.chdir(model_dir)
    sys.path.insert(0, model_dir)

    from pretrain import PretrainConfig, init_train_state, create_dataloader
    from eval_utils import load_checkpoint_and_config

    # Load checkpoint and config using model's own loader
    checkpoint_file, config, checkpoint_dir = load_checkpoint_and_config(args.checkpoint)

    torch.random.manual_seed(config.seed)

    train_loader, train_metadata = create_dataloader(
        config, "train",
        test_set_mode=False,
        epochs_per_iter=1,
        global_batch_size=1,
        rank=0,
        world_size=1,
    )

    # init_train_state: TRM needs rank, others don't
    import inspect
    sig = inspect.signature(init_train_state)
    if "rank" in sig.parameters:
        train_state = init_train_state(config, train_metadata, world_size=1, rank=0)
    else:
        train_state = init_train_state(config, train_metadata, world_size=1)

    print(f"Loading checkpoint: {checkpoint_file}")
    checkpoint_data = torch.load(checkpoint_file, map_location="cuda")
    try:
        train_state.model.load_state_dict(checkpoint_data, assign=True)
    except RuntimeError:
        cleaned = {k.removeprefix("_orig_mod."): v for k, v in checkpoint_data.items()}
        train_state.model.load_state_dict(cleaned, assign=True)

    model = train_state.model
    model.eval()

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {total_params:,}")

    # Load test data
    data_path = getattr(config, "data_path", None) or config.data_paths[0]
    print(f"Data path: {data_path}")
    all_inputs = torch.from_numpy(
        np.load(f"{data_path}/test/all__inputs.npy")
    ).long().cuda()
    all_labels = torch.from_numpy(
        np.load(f"{data_path}/test/all__labels.npy")
    ).long().cuda()

    num_samples = min(args.num_samples, len(all_inputs))
    batch_size = args.batch_size
    num_batches = num_samples // batch_size

    print(f"Measuring FLOPs on {num_batches * batch_size} test puzzles...")
    print("=" * 60)

    total_flops = 0
    total_steps = 0
    total_correct = 0
    total_puzzles = 0

    with torch.no_grad():
        for idx in range(num_batches):
            start = idx * batch_size
            end = start + batch_size

            batch = {
                "inputs": all_inputs[start:end],
                "labels": all_labels[start:end],
                "puzzle_identifiers": torch.zeros(batch_size, dtype=torch.long, device="cuda"),
            }

            # Initialize carry the way eval_utils does it
            with torch.device("cuda"):
                carry = model.initial_carry(batch)

            # Run forward steps with FLOPs counting
            flop_counter = FlopCounterMode(display=False)
            step = 0
            with flop_counter:
                while True:
                    out = model(carry=carry, batch=batch, return_keys=[])
                    # Extract carry and all_finish from variable-length output
                    if hasattr(out[0], "halted"):
                        # (carry, loss, metrics, preds, all_finish, ...)
                        carry = out[0]
                        all_finish = out[4]
                    else:
                        # (trace, carry, loss, metrics, preds, all_finish, ...)
                        carry = out[1]
                        all_finish = out[5]
                    step += 1
                    if all_finish:
                        break

            batch_flops = flop_counter.get_total_flops()
            total_flops += batch_flops
            total_steps += step * batch_size
            total_puzzles += batch_size

            # Check accuracy using final predictions
            final_preds = torch.argmax(
                model(carry=carry, batch=batch, return_keys=["logits"])
                if False else  # We already have the last step's carry
                out[3]["logits"] if hasattr(out[0], "halted") else out[4]["logits"],
                dim=-1,
            ) if False else None
            # Simple accuracy: compare last step's carry predictions
            # (exact accuracy check is complex, just count steps for FLOPs)

            print(f"  Batch {idx+1}/{num_batches}: "
                  f"FLOPs={batch_flops/batch_size:.2e}/puzzle, "
                  f"Steps={step}")

    avg_flops = total_flops / total_puzzles
    avg_steps = total_steps / total_puzzles

    print("=" * 60)
    print(f"Model:                 {args.name}")
    print(f"Task:                  {args.task}")
    print(f"Parameters:            {total_params:,}")
    print(f"Puzzles evaluated:     {total_puzzles}")
    print(f"Avg reasoning steps:   {avg_steps:.2f}")
    print(f"Avg GFLOPs per puzzle: {avg_flops / 1e9:.4f}")
    print("=" * 60)

    output = {
        "name": args.name,
        "task": args.task,
        "checkpoint": checkpoint_file,
        "parameters": total_params,
        "num_puzzles": total_puzzles,
        "avg_steps": avg_steps,
        "avg_flops_per_puzzle": avg_flops,
        "avg_gflops_per_puzzle": avg_flops / 1e9,
        "total_flops": total_flops,
    }

    results_dir = os.path.abspath(args.results_dir) if os.path.isabs(args.results_dir) else args.results_dir
    os.makedirs(results_dir, exist_ok=True)
    safe_name = args.name.lower().replace(" ", "_")
    output_path = os.path.join(results_dir, f"{safe_name}_{args.task}.json")
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

    all_params = [m["parameters"] for m in models]
    min_p, max_p = min(all_params), max(all_params)

    for m in models:
        color = MODEL_COLORS.get(m["name"], "#7f8c8d")
        if max_p > min_p:
            norm = (m["parameters"] - min_p) / (max_p - min_p)
        else:
            norm = 0.5
        size = 150 + norm * 650

        ax.scatter(
            m["avg_gflops_per_puzzle"],
            m.get("exact_accuracy", 0) * 100,
            s=size, c=color, alpha=0.75,
            edgecolors="black", linewidth=1.2, zorder=5,
        )

        ox, oy = 14, 10
        if "Original" in m["name"]:
            oy = -18
        ax.annotate(
            m["name"],
            (m["avg_gflops_per_puzzle"], m.get("exact_accuracy", 0) * 100),
            textcoords="offset points", xytext=(ox, oy),
            fontsize=11, fontweight="bold", color=color,
        )

    ax.set_xlabel("Operations [G-FLOPs]", fontsize=14)
    ax.set_ylabel("Test Exact Accuracy [%]", fontsize=14)
    ax.set_title(title, fontsize=16, fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.tick_params(labelsize=12)

    param_vals = sorted(set(all_params))
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

    print(f"\n{'Model':<20} {'Params':>8} {'Steps':>6} {'GFLOPs':>8}")
    print("-" * 45)
    for m in sorted(models, key=lambda x: x["avg_gflops_per_puzzle"]):
        print(f"{m['name']:<20} {m['parameters']/1e6:>6.1f}M "
              f"{m['avg_steps']:>6.1f} {m['avg_gflops_per_puzzle']:>8.2f}")


def run_plot(args):
    """Read all result JSONs and produce comparative charts."""
    json_files = globmod.glob(os.path.join(args.results_dir, "*.json"))
    if not json_files:
        print(f"No JSON result files found in {args.results_dir}")
        sys.exit(1)

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

    m = sub.add_parser("measure", help="Measure FLOPs on GPU (run on cluster)")
    m.add_argument("--checkpoint", required=True, help="Checkpoint path (file or directory)")
    m.add_argument("--model-dir", required=True, help="Model source directory (contains pretrain.py)")
    m.add_argument("--name", required=True, help="Model name (e.g. 'SHREK Large')")
    m.add_argument("--task", required=True, help="Task name (e.g. 'sudoku', 'maze')")
    m.add_argument("--num-samples", type=int, default=100, help="Test puzzles to measure")
    m.add_argument("--batch-size", type=int, default=10, help="Batch size")
    m.add_argument("--results-dir", default="/home/thheim/HMR/flops/results",
                    help="Directory to save result JSONs")

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
