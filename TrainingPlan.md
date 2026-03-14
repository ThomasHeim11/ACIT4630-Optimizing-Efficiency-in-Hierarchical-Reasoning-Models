# Training Plan

## Overview

1. Transfer code + datasets to cluster
2. Smoke test SHREK (catch bugs before wasting GPU hours)
3. Submit all 5 models as parallel SLURM jobs
4. Monitor via W&B — cancel early if loss diverges
5. Evaluate → measure avg_act_steps → compute GFLOPs

---

## Best Practices

**Always use `sbatch` for full training — never `sinteractive`.**
`sinteractive` jobs die if your SSH session drops. `sbatch` runs in the background regardless.

**No resume functionality exists in pretrain.py.**
If a job is cancelled or crashes, it starts from scratch. Do not cancel jobs unless loss is NaN or clearly broken.

**Set `--time=72:00:00`** (72h instead of 48h) to avoid jobs being killed before finishing. 48h may not be enough for all models.

**`checkpoint_every_eval=True` is already set in all configs.** This saves a checkpoint at every eval — critical for model bootstrapping (needs 10 checkpoints from later training).

**`eval_interval=1000`** gives 40 checkpoints across 40000 epochs. The last 10 (from epoch 30000 onward) are used for model bootstrapping.

**Monitor W&B during training.** If `lm_loss` is NaN or not decreasing after 3000 epochs, cancel and investigate before wasting GPU time.

**Verify data path before submitting.** A wrong path silently loads no data or crashes immediately — check path exists on cluster before running.

---

## Step 1 — Transfer to Cluster

From Mac (run once, then use rsync to sync updates):

```bash
# Datasets
rsync -av -e "ssh -p 60441" \
    /Users/thomasheim/OsloMet/2.semester/ACIT4630/HMR/dataset/data/ \
    thheim@dnat.simula.no:~/HMR/dataset/data/

# Model code
rsync -av -e "ssh -p 60441" \
    /Users/thomasheim/OsloMet/2.semester/ACIT4630/HMR/models/ \
    thheim@dnat.simula.no:~/HMR/models/

# Scripts
rsync -av -e "ssh -p 60441" \
    /Users/thomasheim/OsloMet/2.semester/ACIT4630/HMR/script/ \
    thheim@dnat.simula.no:~/HMR/script/
```

**Verify datasets landed correctly on cluster:**

```bash
ssh thheim@dnat.simula.no -p 60441
ls ~/HMR/dataset/data/
# Should show: arc-2-aug-1000  arc-aug-1000  maze-30x30-hard-1k  sudoku-extreme-1k-aug-1000  sudoku-extreme-1k-aug-1000-hint
```

---

## Step 2 — Smoke Test SHREK (Interactive)

Do this before submitting full jobs. Catches bugs without wasting 72h of GPU time.

```bash
sinteractive -p gh200q --gres=gpu:1 --time=01:00:00
module load cuda12.6/toolkit/12.6.3
cd ~/HMR/models/SHREK-HRM/

# SHREK-Large (2 epochs, small batch)
python3 pretrain.py arch=shrek_large data_path=../../dataset/data/sudoku-extreme-1k-aug-1000-hint epochs=2 eval_interval=1 global_batch_size=32

# SHREK-Tiny
python3 pretrain.py arch=shrek_tiny data_path=../../dataset/data/sudoku-extreme-1k-aug-1000-hint epochs=2 eval_interval=1 global_batch_size=32
```

**Must pass before continuing:**
- [ ] No import errors
- [ ] Loss is a real number (not NaN)
- [ ] `aux_loss` appears in metrics
- [ ] `alpha` in metrics (starts ~0.01)
- [ ] No CUDA device mismatch errors
- [ ] Checkpoint saved without crash

---

## Step 3 — Create SLURM Scripts

Save these to `~/HMR/script/` on the cluster.

### train_hrm.sh
```bash
#!/bin/bash
#SBATCH --job-name=hrm
#SBATCH --partition=gh200q
#SBATCH --gres=gpu:1
#SBATCH --time=72:00:00
#SBATCH --output=/home/thheim/HMR/logs/hrm_%j.log
#SBATCH --error=/home/thheim/HMR/logs/hrm_%j.err

module load cuda12.6/toolkit/12.6.3
cd ~/HMR/models/HRM\(Original\)/HRM-main/

python3 pretrain.py \
    data_path=../../dataset/data/sudoku-extreme-1k-aug-1000 \
    epochs=40000 eval_interval=1000 checkpoint_every_eval=True \
    lr=1e-4 puzzle_emb_lr=1e-4 \
    weight_decay=1.0 puzzle_emb_weight_decay=1.0
```

### train_aughrm.sh
```bash
#!/bin/bash
#SBATCH --job-name=aughrm
#SBATCH --partition=gh200q
#SBATCH --gres=gpu:1
#SBATCH --time=72:00:00
#SBATCH --output=/home/thheim/HMR/logs/aughrm_%j.log
#SBATCH --error=/home/thheim/HMR/logs/aughrm_%j.err

module load cuda12.6/toolkit/12.6.3
cd ~/HMR/models/hrm-mechanistic-analysis-main/

python3 pretrain.py \
    data_path=../../dataset/data/sudoku-extreme-1k-aug-1000-hint \
    epochs=40000 eval_interval=1000 checkpoint_every_eval=True \
    lr=1e-4 puzzle_emb_lr=1e-4 \
    weight_decay=1.0 puzzle_emb_weight_decay=1.0
```

### train_trm.sh
```bash
#!/bin/bash
#SBATCH --job-name=trm
#SBATCH --partition=gh200q
#SBATCH --gres=gpu:1
#SBATCH --time=72:00:00
#SBATCH --output=/home/thheim/HMR/logs/trm_%j.log
#SBATCH --error=/home/thheim/HMR/logs/trm_%j.err

module load cuda12.6/toolkit/12.6.3
cd ~/HMR/models/TinyRecursiveModels/

python3 pretrain.py \
    arch=trm \
    data_paths="[../../dataset/data/sudoku-extreme-1k-aug-1000]" \
    epochs=40000 eval_interval=1000 checkpoint_every_eval=True \
    lr=1e-4 weight_decay=1.0
```

### train_shrek_large.sh
```bash
#!/bin/bash
#SBATCH --job-name=shrek_large
#SBATCH --partition=gh200q
#SBATCH --gres=gpu:1
#SBATCH --time=72:00:00
#SBATCH --output=/home/thheim/HMR/logs/shrek_large_%j.log
#SBATCH --error=/home/thheim/HMR/logs/shrek_large_%j.err

module load cuda12.6/toolkit/12.6.3
cd ~/HMR/models/SHREK-HRM/

python3 pretrain.py \
    arch=shrek_large \
    data_path=../../dataset/data/sudoku-extreme-1k-aug-1000-hint \
    epochs=40000 eval_interval=1000 checkpoint_every_eval=True \
    lr=1e-4 puzzle_emb_lr=1e-4 \
    weight_decay=1.0 puzzle_emb_weight_decay=1.0
```

### train_shrek_tiny.sh
```bash
#!/bin/bash
#SBATCH --job-name=shrek_tiny
#SBATCH --partition=gh200q
#SBATCH --gres=gpu:1
#SBATCH --time=72:00:00
#SBATCH --output=/home/thheim/HMR/logs/shrek_tiny_%j.log
#SBATCH --error=/home/thheim/HMR/logs/shrek_tiny_%j.err

module load cuda12.6/toolkit/12.6.3
cd ~/HMR/models/SHREK-HRM/

python3 pretrain.py \
    arch=shrek_tiny \
    data_path=../../dataset/data/sudoku-extreme-1k-aug-1000-hint \
    epochs=40000 eval_interval=1000 checkpoint_every_eval=True \
    lr=1e-4 puzzle_emb_lr=1e-4 \
    weight_decay=1.0 puzzle_emb_weight_decay=1.0
```

---

## Step 4 — Submit All Jobs

```bash
mkdir -p ~/HMR/logs

sbatch ~/HMR/script/train_hrm.sh
sbatch ~/HMR/script/train_aughrm.sh
sbatch ~/HMR/script/train_trm.sh
sbatch ~/HMR/script/train_shrek_large.sh
sbatch ~/HMR/script/train_shrek_tiny.sh

# Check all are running
squeue -u thheim
```

**Monitor live logs:**
```bash
tail -f ~/HMR/logs/shrek_large_<JOBID>.log
```

**Monitor on W&B:** go to wandb.ai and watch `lm_loss` and `exact_accuracy` curves.

**Cancel a job if loss is NaN:**
```bash
scancel <JOBID>
```

---

## Step 5 — Evaluation + FLOPs

After training, pick the best checkpoint from W&B (highest `exact_accuracy`).

**Single checkpoint evaluation:**
```bash
cd ~/HMR/models/SHREK-HRM/
python3 evaluate.py --checkpoint <path_to_checkpoint>
```

**Model bootstrapping evaluation (AugmentedHRM + SHREK — 10 checkpoints):**

Pick 10 checkpoints from the last 20000 epochs (spaced ~2000 epochs apart):

```bash
python3 batch_inference.py \
    --checkpoints "ckpt_step_30000,ckpt_step_32000,...,ckpt_step_48000" \
    --permutes 9
```

Record `exact_accuracy` and `avg_act_steps` from output, then fill in `MeasureFlops.md`.

---

## Step 6 — Copy Results Back to Mac

```bash
rsync -av -e "ssh -p 60441" \
    thheim@dnat.simula.no:~/HMR/checkpoints/ \
    /Users/thomasheim/OsloMet/2.semester/ACIT4630/HMR/checkpoints/
```

---

## Checklist

| Item | Status |
|---|---|
| Datasets transferred to cluster | ❌ |
| Code transferred to cluster | ❌ |
| wandb login on cluster | ❌ Run `wandb login` once |
| SLURM scripts saved to `script/` | ❌ |
| Smoke test SHREK-Large passes | ❌ |
| Smoke test SHREK-Tiny passes | ❌ |
| All 5 jobs submitted | ❌ |
| W&B monitored — no NaN | ❌ |
| All jobs completed | ❌ |
| Best checkpoints identified via W&B | ❌ |
| Evaluation run on all models | ❌ |
| GFLOPs computed and table filled | ❌ |
