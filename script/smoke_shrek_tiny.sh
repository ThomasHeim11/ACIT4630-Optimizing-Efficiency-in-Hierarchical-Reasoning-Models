#!/bin/bash -l
#SBATCH --job-name=smoke_tiny
#SBATCH --partition=gh200q
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --output=/home/thheim/HMR/logs/smoke_tiny_%j.log
#SBATCH --error=/home/thheim/HMR/logs/smoke_tiny_%j.err

source /etc/profile.d/modules.sh
source ~/.bashrc
module load cuda12.6/toolkit/12.6.3

cd ~/HMR/models/SHREK-HRM

pip install --force-reinstall pydantic pydantic-core --user -q

python3 pretrain.py \
    arch=shrek_tiny \
    data_path=../../dataset/data/sudoku-extreme-1k-aug-1000-hint \
    epochs=2 \
    eval_interval=1 \
    global_batch_size=32
