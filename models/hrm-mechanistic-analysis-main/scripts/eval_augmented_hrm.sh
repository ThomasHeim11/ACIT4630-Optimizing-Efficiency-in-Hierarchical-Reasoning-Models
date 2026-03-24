#!/bin/bash -l
#SBATCH --job-name=eval_aug_hrm
#SBATCH --partition=gh200q
#SBATCH --gres=gpu:1
#SBATCH --output=/home/thheim/HMR/logs/eval_aug_hrm_%j.log
#SBATCH --error=/home/thheim/HMR/logs/eval_aug_hrm_%j.err

source /etc/profile.d/modules.sh
source ~/.bash_profile
module load cuda12.6/toolkit/12.6.3

cd ~/HMR/models/hrm-mechanistic-analysis-main

CKPTS=$(python3 -c "prefix='checkpoints/Sudoku-extreme-1k-aug-1000-hint ACT-torch/HierarchicalReasoningModel_ACTV1 automatic-harrier/step_'; steps=[70308,72912,75516,78120,80724,83328,85932,88536,91140,93744]; print(','.join([f'{prefix}{s}' for s in steps]))")

python3 batch_inference.py --checkpoints "$CKPTS" --permutes 9 --num_batch 10 --batch_size 100
