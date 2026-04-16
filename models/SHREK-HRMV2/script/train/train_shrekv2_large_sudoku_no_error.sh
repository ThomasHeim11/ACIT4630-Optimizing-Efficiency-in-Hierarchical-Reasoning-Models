#!/bin/bash -l
#SBATCH --job-name=sv2_no_err
#SBATCH --partition=gh200q
#SBATCH --gres=gpu:1
#SBATCH --output=/home/thheim/HMR/logs/shrekv2_large_no_err_%j.log
#SBATCH --error=/home/thheim/HMR/logs/shrekv2_large_no_err_%j.err

source /etc/profile.d/modules.sh
source ~/.bash_profile
module load cuda12.6/toolkit/12.6.3

cd ~/HMR/models/SHREK-HRMV2

OMP_NUM_THREADS=8 python3 pretrain.py \
    arch=trm_shrek \
    data_paths="[../../dataset/data/sudoku-extreme-1k-aug-1000]" \
    evaluators="[]" \
    epochs=40000 \
    eval_interval=1000 \
    lr=1e-4 \
    puzzle_emb_lr=1e-4 \
    weight_decay=1.0 \
    puzzle_emb_weight_decay=1.0 \
    global_batch_size=768 \
    arch.L_layers=2 \
    arch.H_cycles=3 \
    arch.L_cycles=6 \
    arch.enable_error_injection=False \
    arch.enable_stagnation_delta=True \
    +run_name=SHREKV2_Large_No_Error \
    +project_name=SHREKV2_Ablation_Sudoku \
    ema=True
