#!/bin/bash -l
#SBATCH --job-name=ablation_no_both
#SBATCH --partition=gh200q
#SBATCH --gres=gpu:1
#SBATCH --output=/home/thheim/HMR/logs/ablation_no_both_%j.log
#SBATCH --error=/home/thheim/HMR/logs/ablation_no_both_%j.err

source /etc/profile.d/modules.sh
source ~/.bash_profile
module load cuda12.6/toolkit/12.6.3

cd ~/HMR/models/SHREK-HRM

OMP_NUM_THREADS=8 python3 pretrain.py \
    arch=shrek_large \
    data_path=../../dataset/data/sudoku-extreme-1k-aug-1000 \
    epochs=40000 \
    eval_interval=1000 \
    global_batch_size=768 \
    lr=1e-4 \
    puzzle_emb_lr=1e-4 \
    weight_decay=1.0 \
    puzzle_emb_weight_decay=1.0 \
    arch.enable_error_injection=False \
    arch.enable_stagnation_delta=False \
    +ema=True \
    +run_name=Ablation_No_Both \
    +project_name=SHREK_Ablation_Sudoku
