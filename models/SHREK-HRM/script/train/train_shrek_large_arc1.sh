#!/bin/bash -l
#SBATCH --job-name=shrek_large_arc1
#SBATCH --partition=gh200q
#SBATCH --gres=gpu:1
#SBATCH --output=/home/thheim/HMR/logs/shrek_large_arc1_%j.log
#SBATCH --error=/home/thheim/HMR/logs/shrek_large_arc1_%j.err

source /etc/profile.d/modules.sh
source ~/.bash_profile
module load cuda12.6/toolkit/12.6.3

cd ~/HMR/models/SHREK-HRM

python3 pretrain.py \
    arch=shrek_large \
    data_path=../../dataset/data/arc-aug-1000 \
    epochs=40000 \
    eval_interval=1000 \
    lr=1e-4 \
    puzzle_emb_lr=1e-4 \
    weight_decay=1.0 \
    puzzle_emb_weight_decay=1.0 \
    +run_name=shrek-large-arc1 \
    +ema=True


