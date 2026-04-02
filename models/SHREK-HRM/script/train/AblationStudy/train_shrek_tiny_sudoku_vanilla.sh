#!/bin/bash -l
#SBATCH --job-name=shrek_tiny_van
#SBATCH --partition=gh200q
#SBATCH --gres=gpu:1
#SBATCH --output=/home/thheim/HMR/logs/shrek_tiny_vanilla_%j.log
#SBATCH --error=/home/thheim/HMR/logs/shrek_tiny_vanilla_%j.err

source /etc/profile.d/modules.sh
source ~/.bash_profile
module load cuda12.6/toolkit/12.6.3

cd ~/HMR/models/SHREK-HRM

OMP_NUM_THREADS=8 python3 pretrain.py \
    arch=shrek_tiny \
    data_path=../../dataset/data/sudoku-extreme-1k-aug-1000 \
    epochs=40000 \
    eval_interval=1000 \
    global_batch_size=768 \
    lr=1e-4 \
    puzzle_emb_lr=1e-4 \
    weight_decay=1.0 \
    puzzle_emb_weight_decay=1.0 \
    arch.enable_error_injection=True \
    arch.enable_stagnation_delta=True \
    +ema=True \
    +run_name=SHREK_Tiny_Vanilla_Sudoku \
    +project_name=SHREK_Ablation_Sudoku
