#!/bin/bash -l
#SBATCH --job-name=sv4_s_maz_nem
#SBATCH --partition=gh200q
#SBATCH --gres=gpu:1
#SBATCH --output=/home/thheim/HMR/logs/shrekv4_small_maze_noema_%j.log
#SBATCH --error=/home/thheim/HMR/logs/shrekv4_small_maze_noema_%j.err

source /etc/profile.d/modules.sh
source ~/.bash_profile
module load cuda12.6/toolkit/12.6.3

cd ~/HMR/models/SHREK-HRMv4

OMP_NUM_THREADS=8 python3 pretrain.py \
    arch=shrek_small \
    data_path=../../dataset/data/maze-30x30-hard-1k \
    epochs=20000 \
    eval_interval=1000 \
    global_batch_size=128 \
    lr=1e-4 \
    puzzle_emb_lr=1e-4 \
    weight_decay=1.0 \
    puzzle_emb_weight_decay=1.0 \
    arch.enable_error_injection=True \
    +run_name=SHREKV4_Small_Maze_Full_NoEMA \
    +project_name=SHREKV4_Maze \
    +ema=False
