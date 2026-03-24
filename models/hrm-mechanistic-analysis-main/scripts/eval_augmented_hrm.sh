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

CKPTS=$(python3 -c "prefix='checkpoints/Sudoku-extreme-1k-aug-1000-hint ACT-torch/HierarchicalReasoningModel_ACTV1 hopeful-quetzal/step_'; print(','.join([f'{prefix}{i*1302}' for i in range(16, 36, 2)]))")

DISABLE_COMPILE=1 PYTHONUNBUFFERED=1 python3 batch_inference.py --checkpoints "$CKPTS" --permutes 9
