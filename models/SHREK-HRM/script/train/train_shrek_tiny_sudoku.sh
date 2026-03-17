#!/bin/bash -l
#SBATCH --job-name=shrek_tiny
#SBATCH --partition=gh200q
#SBATCH --gres=gpu:1
#SBATCH --output=/home/thheim/HMR/logs/shrek_tiny_%j.log
#SBATCH --error=/home/thheim/HMR/logs/shrek_tiny_%j.err

source /etc/profile.d/modules.sh
source ~/.bash_profile
module load cuda12.6/toolkit/12.6.3

pip install --force-reinstall pydantic pydantic-core numexpr bottleneck --user -q

cd ~/HMR/models/SHREK-HRM

python3 pretrain.py \
    arch=shrek_tiny \
    data_path=../../dataset/data/sudoku-extreme-1k-aug-1000-hint \
    epochs=40000 \
    eval_interval=1000 \
    lr=1e-4 \
    puzzle_emb_lr=1e-4 \
    weight_decay=1.0 \
    puzzle_emb_weight_decay=1.0 \
    +run_name=shrek-tiny-sudoku-extreme \
    +ema=True

echo "=============================="
echo "Training complete. Running evaluation..."
echo "=============================="


