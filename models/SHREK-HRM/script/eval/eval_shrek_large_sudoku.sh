#!/bin/bash -l
#SBATCH --job-name=eval_shrek_large
#SBATCH --partition=gh200q
#SBATCH --gres=gpu:1
#SBATCH --output=/home/thheim/HMR/logs/eval_shrek_large_%j.log
#SBATCH --error=/home/thheim/HMR/logs/eval_shrek_large_%j.err

source /etc/profile.d/modules.sh
source ~/.bash_profile
module load cuda12.6/toolkit/12.6.3

cd ~/HMR/models/SHREK-HRM

CKPTS=$(python3 -c "prefix='checkpoints/HRM_Sudoku_Comparison/SHREK_Large_Sudoku/step_'; print(','.join([f'{prefix}{i*1302}' for i in range(31, 41)]))")

DISABLE_COMPILE=1 PYTHONUNBUFFERED=1 python3 batch_inference.py --checkpoints "$CKPTS" --permutes 9 --num_batch 10 --batch_size 100
