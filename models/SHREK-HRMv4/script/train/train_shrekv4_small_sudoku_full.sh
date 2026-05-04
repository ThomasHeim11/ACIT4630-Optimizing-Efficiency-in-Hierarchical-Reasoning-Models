#!/bin/bash -l
#SBATCH --job-name=sv4_s_sud
#SBATCH --partition=gh200q
#SBATCH --gres=gpu:1
#SBATCH --output=/home/thheim/HMR/logs/shrekv4_small_sudoku_%j.log
#SBATCH --error=/home/thheim/HMR/logs/shrekv4_small_sudoku_%j.err

source /etc/profile.d/modules.sh
source ~/.bash_profile
module load cuda12.6/toolkit/12.6.3

cd ~/HMR/models/SHREK-HRMv4

OMP_NUM_THREADS=8 python3 pretrain.py \
    arch=shrek_small \
    data_path=../../dataset/data/sudoku-extreme-1k-aug-1000 \
    epochs=40000 \
    eval_interval=1000 \
    global_batch_size=768 \
    lr=1e-4 \
    puzzle_emb_lr=1e-4 \
    weight_decay=1.0 \
    puzzle_emb_weight_decay=1.0 \
    arch.enable_error_injection=True \
    +run_name=SHREKV4_Small_Sudoku_Full \
    +project_name=SHREKV4_Sudoku \
    +ema=True
