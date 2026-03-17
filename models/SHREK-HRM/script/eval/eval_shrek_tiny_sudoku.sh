#!/bin/bash -l
#SBATCH --job-name=eval_shrek_tiny_sudoku
#SBATCH --partition=gh200q
#SBATCH --gres=gpu:1
#SBATCH --output=/home/thheim/HMR/logs/eval_shrek_tiny_sudoku_%j.log
#SBATCH --error=/home/thheim/HMR/logs/eval_shrek_tiny_sudoku_%j.err

source /etc/profile.d/modules.sh
source ~/.bash_profile
module load cuda12.6/toolkit/12.6.3

cd ~/HMR/models/SHREK-HRM

CKPT_DIR="checkpoints/Sudoku-extreme-1k-aug-1000-hint ACT-torch/shrek-tiny-sudoku-extreme"

CKPTS=$(python3 -c "
import glob, os
files = [f for f in glob.glob('${CKPT_DIR}/step_*') if os.path.isfile(f) and '_all_preds' not in f]
files.sort(key=lambda f: int(os.path.basename(f).split('_')[1]))
print(','.join(files[-10:]))
")
python3 batch_inference.py --checkpoints "$CKPTS" --permutes 9 --num_batch 10 --batch_size 100

# This is a snapshot version. For the full result, use the following instead:
# python3 batch_inference.py --checkpoints "$CKPTS" --permutes 9
