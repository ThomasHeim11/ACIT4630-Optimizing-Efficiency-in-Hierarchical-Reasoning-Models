#!/bin/bash -l
# SLURM wrapper for test.py — runs the evaluation on a GPU node.
# Usage (from the repository root):
#     mkdir -p logs
#     sbatch run_test.sh
#
# Logs are written into ./logs/ relative to the directory you submitted from.

#SBATCH --job-name=shrek_test
#SBATCH --partition=gh200q
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --output=logs/test_py_%j.log
#SBATCH --error=logs/test_py_%j.err

# Load the cluster's module system + CUDA toolkit. The test.py script also
# auto-discovers pip-bundled CUDA libs (libnvrtc-builtins.so.*) inside the
# Python venv, so it works even on nodes where the system CUDA install is
# incomplete.
source /etc/profile.d/modules.sh 2>/dev/null || true
module load cuda12.6/toolkit/12.6.3 2>/dev/null || true

cd "$SLURM_SUBMIT_DIR"
python3 test.py
