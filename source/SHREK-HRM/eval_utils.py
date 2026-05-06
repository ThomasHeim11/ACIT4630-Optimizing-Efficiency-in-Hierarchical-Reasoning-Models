import argparse
import yaml
import os
from typing import Dict, Any, List

import torch
import numpy as np

from dataset.sudoku_transforms import sudoku_cyclic_shift

from pretrain import PretrainConfig, TrainState

import torch.distributed as dist
from puzzle_dataset import PuzzleDataset, PuzzleDatasetConfig, PuzzleDatasetMetadata

def load_checkpoint_and_config(checkpoint_path: str):
    """Load checkpoint and config, adapted from eval_checkpoint.py"""
    # Find the checkpoint directory
    if os.path.isfile(checkpoint_path):
        checkpoint_dir = os.path.dirname(checkpoint_path)
        checkpoint_file = checkpoint_path
    else:
        # Assume it's a directory, find the latest checkpoint
        checkpoint_dir = checkpoint_path
        import glob
        pattern = os.path.join(checkpoint_dir, "step_*")
        checkpoint_files = [f for f in glob.glob(pattern) if os.path.isfile(f)]
        
        if not checkpoint_files:
            raise FileNotFoundError(f"No checkpoint files found in {checkpoint_dir}")
        
        # Sort by step number and take the latest
        def extract_step(filepath):
            filename = os.path.basename(filepath)
            if filename.startswith("step_"):
                try:
                    return int(filename.removeprefix("step_"))
                except ValueError:
                    return 0
            return 0
        
        checkpoint_files.sort(key=extract_step)
        checkpoint_file = checkpoint_files[-1]
        print(f"Using latest checkpoint: {os.path.basename(checkpoint_file)}")
    
    # Load config from checkpoint directory
    config_path = os.path.join(checkpoint_dir, "all_config.yaml")
    
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, "r") as f:
        config = PretrainConfig(**yaml.safe_load(f))
    
    print(f"Loaded config from: {config_path}")
    print(f"Model: {config.arch.name}")
    print(f"Data path: {config.data_path}")
    
    return checkpoint_file, config, checkpoint_dir

def create_single_sample(inputs, labels, puzzle_id: int = 0):
    inputs_tensor = torch.from_numpy(inputs).long().unsqueeze(0)
    labels_tensor = torch.from_numpy(labels).long().unsqueeze(0)
    puzzle_ids_tensor = torch.from_numpy(np.asarray(puzzle_id)).long().unsqueeze(0)
    
    batch = {
        "inputs": inputs_tensor,
        "labels": labels_tensor,
        "puzzle_identifiers": puzzle_ids_tensor
    }
    
    return batch

def forward_single_sample(model, batch: Dict[str, torch.Tensor]):
    model.eval()
    
    # Move batch to device
    batch = {k: v.cuda() for k, v in batch.items()}
    
    # Initialize carry state (adapted from pretrain.py)
    with torch.device("cuda"):
        carry = model.initial_carry(batch)
    
    # Store results from each step
    all_logits = []
    all_predictions = []
    losses = []
    all_carries = []
    z_H_traces = []
    z_H_segments = []
    
    step = 0
    
    with torch.inference_mode():
        while True:
            with torch.no_grad():
                z_H_trace, carry, loss, metrics, preds, all_finish, act_halt = model(
                    carry=carry, 
                    batch=batch, 
                    require_trace=True, # This option allows one to access intermediate z_H states
                    return_keys=["logits"]  # Request logits in return
                )
                all_carries.append(carry)
                z_H_traces.extend(z_H_trace)

                z_H = carry.inner_carry.z_H.clone().squeeze(0)[:,:] # (82, 512)
                z_H_segments.append(z_H)

                losses.append(loss)
            
            logits = preds["logits"]  # [1, seq_len, vocab_size]
            predictions = torch.argmax(logits, dim=-1)  # [1, seq_len]
            all_logits.append(logits.cpu())
            all_predictions.append(predictions.cpu())
            
            step += 1

            if all_finish:
                break
    
    return {
        "all_losses": losses,
        "all_logits": all_logits,
        "all_predictions": all_predictions,
        "all_carries": all_carries,
        "total_steps": step,
        "z_H_segments": z_H_segments,
        "z_H_traces": z_H_traces,
    }

def create_batch(all_inputs, all_labels, puzzle_id: int, idx: int, batch_size, permute):
    
    inputs = torch.from_numpy(all_inputs[idx*batch_size:(idx+1)*batch_size]).cuda()
    labels = torch.from_numpy(all_labels[idx*batch_size:(idx+1)*batch_size]).cuda()

    inputs_tensor = sudoku_cyclic_shift(inputs.long(), permute)
    labels_tensor = sudoku_cyclic_shift(labels.long(), permute)
    puzzle_ids_tensor = torch.from_numpy(np.asarray(puzzle_id)).long()
    
    batch = {
        "inputs": inputs_tensor,
        "labels": labels_tensor,
        "puzzle_identifiers": puzzle_ids_tensor
    }
    
    return batch

def forward_batch(model, batch: Dict[str, torch.Tensor]):
    batch = {k: v.cuda() for k, v in batch.items()}
    
    with torch.device("cuda"):
        carry = model.initial_carry(batch)
    
    # Store results from each step
    all_logits = []
    all_predictions = []
    all_carries = []
    all_losses = []
    
    step = 0
    
    # Iterative forward passes until halting (adapted from pretrain.py evaluate function)
    with torch.inference_mode():
        while True:
            carry, loss, metrics, preds, all_finish, act_halt = model(
                carry=carry, 
                batch=batch, 
                return_keys=["logits"]
            )

            all_losses.append(float(loss))
            all_carries.append(carry)
            logits = preds["logits"]
            predictions = torch.argmax(logits, dim=-1)  # [B, seq_len]
            all_logits.append(logits)
            all_predictions.append(predictions)
            
            step += 1
            if all_finish:
                break

    return {
        "all_losses": all_losses,
        "all_logits": all_logits,
        "all_predictions": all_predictions,
        "all_carries": all_carries,
        "total_steps": step,
        "act_halt": act_halt
    }