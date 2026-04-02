#!/bin/bash -l
#SBATCH --job-name=flops
#SBATCH --partition=gh200q
#SBATCH --gres=gpu:1
#SBATCH --output=/home/thheim/HMR/logs/flops_%j.log
#SBATCH --error=/home/thheim/HMR/logs/flops_%j.err

source /etc/profile.d/modules.sh
source ~/.bash_profile
module load cuda12.6/toolkit/12.6.3

FLOPS_SCRIPT=~/HMR/flops/flops.py
RESULTS_DIR=~/HMR/flops/results
CKPT_ROOT=~/HMR/checkpoints
mkdir -p $RESULTS_DIR

# Checkpoint directories — the script auto-picks the latest step_* in each.
# Must cd into each model dir so `from pretrain import ...` works.

# ── Sudoku models ──

echo "=== Original HRM (Sudoku) ==="
cd ~/HMR/models/HRM\(Original\)/HRM-main
DISABLE_COMPILE=1 python3 $FLOPS_SCRIPT measure \
    --checkpoint "$CKPT_ROOT/sudoku-extreme/original-hrm" \
    --name "Original HRM" --task sudoku --results-dir $RESULTS_DIR

echo "=== Augmented HRM (Sudoku) ==="
cd ~/HMR/models/hrm-mechanistic-analysis-main
DISABLE_COMPILE=1 python3 $FLOPS_SCRIPT measure \
    --checkpoint "$CKPT_ROOT/sudoku-extreme/augmented-hrm" \
    --name "Augmented HRM" --task sudoku --results-dir $RESULTS_DIR

echo "=== SHREK Large (Sudoku) ==="
cd ~/HMR/models/SHREK-HRM
DISABLE_COMPILE=1 python3 $FLOPS_SCRIPT measure \
    --checkpoint "$CKPT_ROOT/sudoku-extreme/shrek-large" \
    --name "SHREK Large" --task sudoku --results-dir $RESULTS_DIR

echo "=== SHREK Tiny (Sudoku) ==="
cd ~/HMR/models/SHREK-HRM
DISABLE_COMPILE=1 python3 $FLOPS_SCRIPT measure \
    --checkpoint "$CKPT_ROOT/sudoku-extreme/shrek-tiny" \
    --name "SHREK Tiny" --task sudoku --results-dir $RESULTS_DIR

echo "=== TRM Attention (Sudoku) ==="
cd ~/HMR/models/TinyRecursiveModels
DISABLE_COMPILE=1 python3 $FLOPS_SCRIPT measure \
    --checkpoint "$CKPT_ROOT/sudoku-extreme/trm-attention" \
    --name "TRM Attention" --task sudoku --results-dir $RESULTS_DIR

echo "=== TRM MLP (Sudoku) ==="
cd ~/HMR/models/TinyRecursiveModels
DISABLE_COMPILE=1 python3 $FLOPS_SCRIPT measure \
    --checkpoint "$CKPT_ROOT/sudoku-extreme/trm-mlp" \
    --name "TRM MLP" --task sudoku --results-dir $RESULTS_DIR

# ── Maze models ──

echo "=== Original HRM (Maze) ==="
cd ~/HMR/models/HRM\(Original\)/HRM-main
DISABLE_COMPILE=1 python3 $FLOPS_SCRIPT measure \
    --checkpoint "$CKPT_ROOT/maze-hard/original-hrm" \
    --name "Original HRM" --task maze --results-dir $RESULTS_DIR

echo "=== SHREK Large (Maze) ==="
cd ~/HMR/models/SHREK-HRM
DISABLE_COMPILE=1 python3 $FLOPS_SCRIPT measure \
    --checkpoint "$CKPT_ROOT/maze-hard/shrek-large" \
    --name "SHREK Large" --task maze --results-dir $RESULTS_DIR

echo "=== SHREK Tiny (Maze) ==="
cd ~/HMR/models/SHREK-HRM
DISABLE_COMPILE=1 python3 $FLOPS_SCRIPT measure \
    --checkpoint "$CKPT_ROOT/maze-hard/shrek-tiny" \
    --name "SHREK Tiny" --task maze --results-dir $RESULTS_DIR

echo "=== TRM Attention (Maze) ==="
cd ~/HMR/models/TinyRecursiveModels
DISABLE_COMPILE=1 python3 $FLOPS_SCRIPT measure \
    --checkpoint "$CKPT_ROOT/maze-hard/trm-attention" \
    --name "TRM Attention" --task maze --results-dir $RESULTS_DIR

echo ""
echo "Done! To generate charts locally:"
echo "  python3 flops/flops.py plot --results-dir flops/results"
