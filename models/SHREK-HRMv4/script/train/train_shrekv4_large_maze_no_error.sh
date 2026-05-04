#!/bin/bash -l
#SBATCH --job-name=sv4_l_maz_ne
#SBATCH --partition=gh200q
#SBATCH --gres=gpu:1
#SBATCH --output=/home/thheim/HMR/logs/shrekv4_large_maze_no_error_%j.log
#SBATCH --error=/home/thheim/HMR/logs/shrekv4_large_maze_no_error_%j.err

source /etc/profile.d/modules.sh
source ~/.bash_profile
module load cuda12.6/toolkit/12.6.3

cd ~/HMR/models/SHREK-HRMv4

OMP_NUM_THREADS=8 python3 pretrain.py \
    arch=shrek_large \
    data_path=../../dataset/data/maze-30x30-hard-1k \
    epochs=20000 \
    eval_interval=1000 \
    global_batch_size=128 \
    lr=1e-4 \
    puzzle_emb_lr=1e-4 \
    weight_decay=1.0 \
    puzzle_emb_weight_decay=1.0 \
    arch.enable_error_injection=False \
    +run_name=SHREKV4_Large_Maze_NoError \
    +project_name=SHREKV4_Maze \
    +ema=True
