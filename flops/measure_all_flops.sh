#!/bin/bash -l
#SBATCH --job-name=flops
#SBATCH --partition=gh200q
#SBATCH --gres=gpu:1
#SBATCH --output=/home/thheim/HMR/logs/flops_%j.log
#SBATCH --error=/home/thheim/HMR/logs/flops_%j.err

source /etc/profile.d/modules.sh
source ~/.bash_profile
module load cuda12.6/toolkit/12.6.3

FLOPS=~/HMR/flops/flops.py
RESULTS=~/HMR/flops/results
CKPT=~/HMR/checkpoints
HRM_DIR=~/HMR/source/HRM\(Original\)/HRM-main
SHREK_DIR=~/HMR/source/SHREK-HRM
TRM_DIR=~/HMR/source/TinyRecursiveModels

mkdir -p $RESULTS

# ── Sudoku ──

echo "=== Original HRM (Sudoku) ==="
DISABLE_COMPILE=1 python3 $FLOPS measure \
    --checkpoint "$CKPT/sudoku-extreme/original-hrm" \
    --model-dir "$HRM_DIR" \
    --name "Original HRM" --task sudoku --num-samples 1000 --results-dir $RESULTS

echo "=== SHREK Large (Sudoku) ==="
DISABLE_COMPILE=1 python3 $FLOPS measure \
    --checkpoint "$CKPT/sudoku-extreme/shrek-large" \
    --model-dir "$SHREK_DIR" \
    --name "SHREK Large" --task sudoku --num-samples 1000 --results-dir $RESULTS

echo "=== SHREK Small (Sudoku) ==="
DISABLE_COMPILE=1 python3 $FLOPS measure \
    --checkpoint "$CKPT/sudoku-extreme/shrek-tiny" \
    --model-dir "$SHREK_DIR" \
    --name "SHREK Small" --task sudoku --num-samples 1000 --results-dir $RESULTS

echo "=== TRM Attention (Sudoku) ==="
DISABLE_COMPILE=1 python3 $FLOPS measure \
    --checkpoint "$CKPT/sudoku-extreme/trm-attention" \
    --model-dir "$TRM_DIR" \
    --name "TRM Attention" --task sudoku --num-samples 1000 --results-dir $RESULTS

# ── Maze ──

echo "=== Original HRM (Maze) ==="
DISABLE_COMPILE=1 python3 $FLOPS measure \
    --checkpoint "$CKPT/maze-hard/original-hrm" \
    --model-dir "$HRM_DIR" \
    --name "Original HRM" --task maze --num-samples 1000 --results-dir $RESULTS

echo "=== SHREK Large (Maze) ==="
DISABLE_COMPILE=1 python3 $FLOPS measure \
    --checkpoint "$CKPT/maze-hard/shrek-large" \
    --model-dir "$SHREK_DIR" \
    --name "SHREK Large" --task maze --num-samples 1000 --results-dir $RESULTS

echo "=== SHREK Small (Maze) ==="
DISABLE_COMPILE=1 python3 $FLOPS measure \
    --checkpoint "$CKPT/maze-hard/shrek-tiny" \
    --model-dir "$SHREK_DIR" \
    --name "SHREK Small" --task maze --num-samples 1000 --results-dir $RESULTS

echo "=== TRM Attention (Maze) ==="
DISABLE_COMPILE=1 python3 $FLOPS measure \
    --checkpoint "$CKPT/maze-hard/trm-attention" \
    --model-dir "$TRM_DIR" \
    --name "TRM Attention" --task maze --num-samples 1000 --results-dir $RESULTS

echo ""
echo "Done! To generate charts locally:"
echo "  python3 flops/flops.py plot --results-dir flops/results"
