#!/bin/bash -l
# Quick verification: evaluate the local V1 Ablation_Full checkpoint to confirm
# it gives ~65% on Sudoku (the paper's reported number).
# Usage: sbatch verify_local.sh

#SBATCH --job-name=verify_local
#SBATCH --partition=gh200q
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --output=/home/thheim/HMR/logs/verify_%j.log
#SBATCH --error=/home/thheim/HMR/logs/verify_%j.err

source /etc/profile.d/modules.sh
source ~/.bash_profile
module load cuda12.6/toolkit/12.6.3

cd ~/HMR/source/SHREK-HRM
python3 evaluate.py checkpoint=checkpoints/SHREK_Ablation_Sudoku/SHREK_Tiny_Vanilla_Sudoku/step_37758
