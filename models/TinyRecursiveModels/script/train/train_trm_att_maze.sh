#!/bin/bash -l
#SBATCH --job-name=trm_att_maze
#SBATCH --partition=gh200q
#SBATCH --gres=gpu:1
#SBATCH --output=/home/thheim/HMR/logs/trm_att_maze_%j.log
#SBATCH --error=/home/thheim/HMR/logs/trm_att_maze_%j.err

source /etc/profile.d/modules.sh
source ~/.bash_profile
module load cuda12.6/toolkit/12.6.3

cd ~/HMR/models/TinyRecursiveModels

OMP_NUM_THREADS=8 python3 pretrain.py \
    arch=trm \
    data_paths="[../../dataset/data/maze-30x30-hard-1k]" \
    evaluators="[]" \
    epochs=20000 \
    eval_interval=1000 \
    lr=1e-4 \
    puzzle_emb_lr=1e-4 \
    weight_decay=1.0 \
    puzzle_emb_weight_decay=1.0 \
    global_batch_size=768 \
    arch.L_layers=2 \
    arch.H_cycles=3 \
    arch.L_cycles=4 \
    +run_name=TRM_ATT_Maze \
    +project_name=HRM_Maze_Comparison \
    ema=True
