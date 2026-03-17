#!/bin/bash -l
#SBATCH --job-name=eval_shrek_large_maze
#SBATCH --partition=gh200q
#SBATCH --gres=gpu:1
#SBATCH --output=/home/thheim/HMR/logs/eval_shrek_large_maze_%j.log
#SBATCH --error=/home/thheim/HMR/logs/eval_shrek_large_maze_%j.err

source /etc/profile.d/modules.sh
source ~/.bash_profile
module load cuda12.6/toolkit/12.6.3

pip install --force-reinstall pydantic pydantic-core numexpr bottleneck --user -q

cd ~/HMR/models/SHREK-HRM

python3 batch_inference.py --checkpoints "checkpoints/Maze-30x30-hard-1k ACT-torch/shrek-large-maze" \
--dataset maze \
--num_batch 10 --batch_size 100 --permutes 2

# Maze-Hard has 1000 test samples in total.
# Do not set --permute value other than 1 or 2 for maze ckpt evaluation.
