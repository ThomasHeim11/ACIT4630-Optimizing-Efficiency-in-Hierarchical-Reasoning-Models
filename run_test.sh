#!/bin/bash -l
# SLURM wrapper for test.py — submits the test on a GPU node.
# Usage: sbatch run_test.sh

#SBATCH --job-name=shrek_test
#SBATCH --partition=gh200q
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --output=/home/thheim/HMR/logs/test_py_%j.log
#SBATCH --error=/home/thheim/HMR/logs/test_py_%j.err

source /etc/profile.d/modules.sh
source ~/.bash_profile
module load cuda12.6/toolkit/12.6.3

cd ~/HMR
python3 test.py
