#!/bin/bash -l
#SBATCH --job-name=shrek_large
#SBATCH --partition=gh200q
#SBATCH --gres=gpu:1
#SBATCH --output=/home/thheim/HMR/logs/shrek_large_%j.log
#SBATCH --error=/home/thheim/HMR/logs/shrek_large_%j.err

source /etc/profile.d/modules.sh
source ~/.bash_profile
module load cuda12.6/toolkit/12.6.3

cd ~/HMR/models/SHREK-HRM

DISABLE_COMPILE=1 OMP_NUM_THREADS=8 python3 pretrain.py \
    arch=shrek_large \
    data_path=../../dataset/data/sudoku-extreme-1k-aug-1000-hint \
    epochs=40000 \
    eval_interval=1000 \
    global_batch_size=384 \
    lr=7e-5 \
    puzzle_emb_lr=7e-5 \
    weight_decay=1.0 \
    puzzle_emb_weight_decay=1.0 \
    +ema=True \
    +run_name=SHREK_Large_Sudoku


