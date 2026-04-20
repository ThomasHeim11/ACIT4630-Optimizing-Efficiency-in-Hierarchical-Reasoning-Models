#!/bin/bash -l
#SBATCH --job-name=sv3_n_maz
#SBATCH --partition=gh200q
#SBATCH --gres=gpu:1
#SBATCH --output=/home/thheim/HMR/logs/shrekv3_normal_maze_%j.log
#SBATCH --error=/home/thheim/HMR/logs/shrekv3_normal_maze_%j.err

source /etc/profile.d/modules.sh
source ~/.bash_profile
module load cuda12.6/toolkit/12.6.3

cd ~/HMR/models/SHREK-HRMV2

OMP_NUM_THREADS=8 python3 pretrain.py \
    arch=trm_shrekv3_normal \
    data_paths="[../../dataset/data/maze-30x30-hard-1k]" \
    evaluators="[]" \
    epochs=20000 \
    eval_interval=1000 \
    lr=1e-4 \
    puzzle_emb_lr=1e-4 \
    weight_decay=1.0 \
    puzzle_emb_weight_decay=1.0 \
    global_batch_size=128 \
    arch.L_layers=2 \
    arch.H_cycles=3 \
    arch.L_cycles=4 \
    arch.enable_error_injection=True \
    +run_name=SHREKV3_Normal_Maze \
    +project_name=SHREKV3_Maze \
    ema=True
