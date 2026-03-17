#!/bin/bash -l
#SBATCH --job-name=shrek_tiny_maze
#SBATCH --partition=gh200q
#SBATCH --gres=gpu:1
#SBATCH --output=/home/thheim/HMR/logs/shrek_tiny_maze_%j.log
#SBATCH --error=/home/thheim/HMR/logs/shrek_tiny_maze_%j.err

source /etc/profile.d/modules.sh
source ~/.bash_profile
module load cuda12.6/toolkit/12.6.3

cd ~/HMR/models/SHREK-HRM

python3 pretrain.py \
    arch=shrek_tiny \
    data_path=../../dataset/data/maze-30x30-hard-1k \
    epochs=40000 \
    eval_interval=1000 \
    lr=1e-4 \
    puzzle_emb_lr=1e-4 \
    weight_decay=1.0 \
    puzzle_emb_weight_decay=1.0 \
    +run_name=shrek-tiny-maze \
    +ema=True



