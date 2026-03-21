#!/bin/bash -l
#SBATCH --job-name=baseline_hrm
#SBATCH --partition=gh200q
#SBATCH --gres=gpu:1
#SBATCH --output=/home/thheim/HMR/logs/baseline_hrm_%j.log
#SBATCH --error=/home/thheim/HMR/logs/baseline_hrm_%j.err

source /etc/profile.d/modules.sh
source ~/.bash_profile
module load cuda12.6/toolkit/12.6.3

cd ~/HMR/models/hrm-mechanistic-analysis-main

python3 pretrain.py \
    data_path=../../dataset/data/sudoku-extreme-1k-aug-1000-hint \
    epochs=40000 \
    eval_interval=1000 \
    global_batch_size=384 \
    lr=7e-5 \
    puzzle_emb_lr=7e-5 \
    weight_decay=1.0 \
    puzzle_emb_weight_decay=1.0
