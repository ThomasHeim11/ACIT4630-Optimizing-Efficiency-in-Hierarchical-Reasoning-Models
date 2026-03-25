#!/bin/bash -l
#SBATCH --job-name=trm_att
#SBATCH --partition=gh200q
#SBATCH --gres=gpu:1
#SBATCH --output=/home/thheim/HMR/logs/trm_att_%j.log
#SBATCH --error=/home/thheim/HMR/logs/trm_att_%j.err

source /etc/profile.d/modules.sh
source ~/.bash_profile
module load cuda12.6/toolkit/12.6.3

cd ~/HMR/models/TinyRecursiveModels

OMP_NUM_THREADS=8 python3 pretrain.py \
    arch=trm \
    data_paths="[../../dataset/data/sudoku-extreme-1k-aug-1000]" \
    evaluators="[]" \
    epochs=40000 \
    eval_interval=1000 \
    lr=1e-4 \
    puzzle_emb_lr=1e-4 \
    weight_decay=1.0 \
    puzzle_emb_weight_decay=1.0 \
    arch.L_layers=2 \
    arch.H_cycles=3 \
    arch.L_cycles=6 \
    +run_name=TRM_ATT_Sudoku \
    +project_name=HRM_Sudoku_Comparison \
    ema=True
