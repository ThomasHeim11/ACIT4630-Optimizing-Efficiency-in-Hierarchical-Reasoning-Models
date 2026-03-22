#!/bin/bash -l
#SBATCH --job-name=aug_hrm
#SBATCH --partition=gh200q
#SBATCH --gres=gpu:1
#SBATCH --output=/home/thheim/HMR/logs/aug_hrm_%j.log
#SBATCH --error=/home/thheim/HMR/logs/aug_hrm_%j.err

source /etc/profile.d/modules.sh
source ~/.bash_profile
module load cuda12.6/toolkit/12.6.3

cd ~/HMR/models/hrm-mechanistic-analysis-main

DISABLE_COMPILE=1 OMP_NUM_THREADS=8 python3 pretrain.py \
    data_path=../../dataset/data/sudoku-extreme-1k-aug-1000-hint \
    epochs=40000 \
    eval_interval=1000 \
    global_batch_size=384 \
    lr=7e-5 \
    puzzle_emb_lr=7e-5 \
    weight_decay=1.0 \
    puzzle_emb_weight_decay=1.0 \
    +run_name=Augmented_HRM_Sudoku \
    +project_name=HRM_Sudoku_Comparison
