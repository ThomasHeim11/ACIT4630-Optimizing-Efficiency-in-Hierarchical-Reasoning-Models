#!/bin/bash -l
#SBATCH --job-name=shrek_tiny
#SBATCH --partition=gh200q
#SBATCH --gres=gpu:1
#SBATCH --output=/home/thheim/HMR/logs/shrek_tiny_%j.log
#SBATCH --error=/home/thheim/HMR/logs/shrek_tiny_%j.err

source /etc/profile.d/modules.sh
source ~/.bash_profile
module load cuda12.6/toolkit/12.6.3

cd ~/HMR/models/SHREK-HRM

python3 pretrain.py \
    arch=shrek_tiny \
    data_path=../../dataset/data/sudoku-extreme-1k-aug-1000-hint \
    epochs=40000 \
    eval_interval=1000 \
    +ema=True


