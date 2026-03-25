#!/bin/bash -l
#SBATCH --job-name=flops
#SBATCH --partition=gh200q
#SBATCH --gres=gpu:1
#SBATCH --output=/home/thheim/HMR/logs/flops_%j.log
#SBATCH --error=/home/thheim/HMR/logs/flops_%j.err

source /etc/profile.d/modules.sh
source ~/.bash_profile
module load cuda12.6/toolkit/12.6.3

echo "=== Original HRM ==="
cd ~/HMR/models/HRM\(Original\)/HRM-main
DISABLE_COMPILE=1 python3 ../../../flops/flops.py \
    --checkpoint "checkpoints/Sudoku-extreme-1k-aug-1000 ACT-torch/HierarchicalReasoningModel_ACTV1 liberal-bee/step_52080"

echo "=== Augmented HRM ==="
cd ~/HMR/models/hrm-mechanistic-analysis-main
DISABLE_COMPILE=1 python3 ../../flops/flops.py \
    --checkpoint "checkpoints/Sudoku-extreme-1k-aug-1000-hint ACT-torch/HierarchicalReasoningModel_ACTV1 hopeful-quetzal/step_36456"

echo "=== SHREK Large ==="
cd ~/HMR/models/SHREK-HRM
DISABLE_COMPILE=1 python3 ../../flops/flops.py \
    --checkpoint "checkpoints/HRM_Sudoku_Comparison/SHREK_Large_Sudoku/step_52080"

echo "=== SHREK Tiny ==="
cd ~/HMR/models/SHREK-HRM
DISABLE_COMPILE=1 python3 ../../flops/flops.py \
    --checkpoint "checkpoints/HRM_Sudoku_Comparison/SHREK_Tiny_Sudoku/step_52080"
