"""
FLOPs Measurement Script for HRM-family Models

Measures FLOPs per puzzle and average reasoning steps for:
- Original HRM, Augmented HRM, SHREK Large, SHREK Tiny, TRM

Usage: Run from within each model's directory.

  cd ~/HMR/models/HRM(Original)/HRM-main
  DISABLE_COMPILE=1 python3 ../../flops/flops.py --checkpoint <CKPT_PATH>

  cd ~/HMR/models/hrm-mechanistic-analysis-main
  DISABLE_COMPILE=1 python3 ../../flops/flops.py --checkpoint <CKPT_PATH>

  cd ~/HMR/models/SHREK-HRM
  DISABLE_COMPILE=1 python3 ../../flops/flops.py --checkpoint <CKPT_PATH>

  cd ~/HMR/models/TinyRecursiveModels
  DISABLE_COMPILE=1 python3 ../../flops/flops.py --checkpoint <CKPT_PATH>
"""

import argparse
import os
import sys
import yaml
import json

import torch
import numpy as np
from torch.utils.flop_counter import FlopCounterMode

# Import from the model's own codebase (script must be run from model dir)
from pretrain import PretrainConfig, init_train_state, create_dataloader


def load_model(checkpoint_path):
    """Load model from checkpoint and its config."""
    checkpoint_dir = os.path.dirname(checkpoint_path)
    config_path = os.path.join(checkpoint_dir, "all_config.yaml")

    with open(config_path, "r") as f:
        config = PretrainConfig(**yaml.safe_load(f))

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

    # Load checkpoint weights
    checkpoint = torch.load(checkpoint_path, map_location="cuda")
    try:
        train_state.model.load_state_dict(checkpoint, assign=True)
    except:
        cleaned = {k.removeprefix("_orig_mod."): v for k, v in checkpoint.items()}
        train_state.model.load_state_dict(cleaned, assign=True)

    train_state.model.eval()
    return train_state.model, config


def measure_flops(model, config, num_samples=100, batch_size=10):
    """Measure FLOPs per puzzle and average reasoning steps."""
    # Load test data
    all_inputs = torch.from_numpy(
        np.load(f"{config.data_path}/test/all__inputs.npy")
    ).long().cuda()
    all_labels = torch.from_numpy(
        np.load(f"{config.data_path}/test/all__labels.npy")
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

            # Initialize carry
            carry = model.initial_carry(batch_size, batch_inputs.device)

            # Measure FLOPs for full forward pass
            flop_counter = FlopCounterMode(display=False)
            with flop_counter:
                new_carry, loss, metrics, _, _ = model(carry=carry, batch=batch, return_keys=[])

            batch_flops = flop_counter.get_total_flops()
            total_flops += batch_flops

            # Get steps from metrics
            if "steps" in metrics:
                total_steps += metrics["steps"].item()

            # Get accuracy
            if "exact_accuracy" in metrics:
                total_correct += metrics["exact_accuracy"].item()

            total_puzzles += batch_size

            print(f"Batch {idx+1}/{num_batches}: "
                  f"FLOPs={batch_flops/batch_size:.2e}/puzzle, "
                  f"Steps={metrics.get('steps', torch.tensor(0)).item()/batch_size:.1f}")

    # Calculate results
    avg_flops_per_puzzle = total_flops / total_puzzles
    avg_steps = total_steps / total_puzzles
    accuracy = total_correct / total_puzzles

    return {
        "total_flops": total_flops,
        "avg_flops_per_puzzle": avg_flops_per_puzzle,
        "avg_steps": avg_steps,
        "num_puzzles": total_puzzles,
        "exact_accuracy": accuracy,
    }


def count_parameters(model):
    """Count total and trainable parameters."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def main():
    parser = argparse.ArgumentParser(description="Measure FLOPs for HRM-family models")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--num_samples", type=int, default=100, help="Number of test puzzles to measure")
    parser.add_argument("--batch_size", type=int, default=10, help="Batch size for measurement")
    args = parser.parse_args()

    torch.cuda.set_device(0)

    print(f"Loading model from: {args.checkpoint}")
    model, config = load_model(args.checkpoint)

    total_params, trainable_params = count_parameters(model)
    print(f"Parameters: {total_params:,} total, {trainable_params:,} trainable")
    print(f"Data path: {config.data_path}")
    print(f"Measuring FLOPs on {args.num_samples} test puzzles...")
    print("=" * 60)

    results = measure_flops(model, config, args.num_samples, args.batch_size)

    print("=" * 60)
    print(f"RESULTS")
    print(f"=" * 60)
    print(f"Model parameters:      {total_params:,}")
    print(f"Puzzles evaluated:     {results['num_puzzles']}")
    print(f"Exact accuracy:        {results['exact_accuracy']:.4f}")
    print(f"Avg reasoning steps:   {results['avg_steps']:.2f}")
    print(f"Avg FLOPs per puzzle:  {results['avg_flops_per_puzzle']:.4e}")
    print(f"Total FLOPs:           {results['total_flops']:.4e}")
    print(f"FLOPs per step:        {results['avg_flops_per_puzzle'] / max(results['avg_steps'], 1):.4e}")
    print(f"=" * 60)

    # Save results to JSON
    output = {
        "checkpoint": args.checkpoint,
        "parameters": total_params,
        "num_puzzles": results["num_puzzles"],
        "exact_accuracy": results["exact_accuracy"],
        "avg_steps": results["avg_steps"],
        "avg_flops_per_puzzle": results["avg_flops_per_puzzle"],
        "total_flops": results["total_flops"],
        "flops_per_step": results["avg_flops_per_puzzle"] / max(results["avg_steps"], 1),
    }

    output_path = os.path.join(os.path.dirname(args.checkpoint), "flops_results.json")
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"Results saved to: {output_path}")


if __name__ == "__main__":
    main()
