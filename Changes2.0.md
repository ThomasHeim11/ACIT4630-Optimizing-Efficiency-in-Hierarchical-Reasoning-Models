# SHREK 2.0 — Changes to Improve Accuracy and Reduce FLOPs

## Problem

SHREK models overfit on Sudoku after ~25k steps (accuracy crashes from 65% to ~8%).
SHREK Tiny is currently 14M params (hidden_size=512) instead of the intended ~7M.

## Changes

### 1. SHREK Tiny — Reduce to ~7M Parameters

Current: `hidden_size=512, num_heads=8, 2H+2L layers` → 14M params
Target: `hidden_size=256, num_heads=4, 2H+2L layers` → ~7M params

In `config/arch/shrek_tiny.yaml`:
```yaml
hidden_size: 256   # reduced from 512
num_heads: 4       # reduced from 8 (keeps head_dim=64)
```

This matches TRM's parameter range (~5-7M) for a fair "less is more" comparison.
Previous attempt with hidden_size=256 caused training collapse — may need to investigate further.

### 2. Hyperparameter Tuning — Prevent Overfitting

Current SHREK settings vs Original HRM:

| Setting          | Original HRM | TRM     | SHREK (current) | SHREK 2.0 (proposed) |
|------------------|-------------|---------|-----------------|---------------------|
| lr               | 7e-5        | 1e-4    | 1e-4            | 7e-5                |
| epochs (Sudoku)  | 20,000      | 40,000  | 40,000          | 40,000              |
| global_batch_size| 384         | 768     | 768             | 384                 |
| EMA              | No          | Yes     | Yes             | Yes                 |
| weight_decay     | 1.0         | 1.0     | 1.0             | 1.0                 |

Changes:
- **lr: 1e-4 → 7e-5** — slower learning, less overfitting
- **global_batch_size: 768 → 384** — match HRM's 1-GPU recipe
- **epochs: keep at 40,000** — same as TRM for fair comparison

### 3. Additional Options (if still overfitting)

- **Cosine LR decay:** Change `lr_min_ratio` from `1.0` to `0.1` in pretrain.py
  - Currently LR stays constant after warmup — adding decay reduces updates late in training
- **Early stopping:** Monitor `all.exact_accuracy` and save best checkpoint

## Expected Outcome

- Accuracy holds at peak (~65% or higher) instead of collapsing
- Fewer halting steps at peak checkpoint → lower GFLOPs
- SHREK Tiny at 7M matching or beating Original HRM at 27M
- Fair comparison: all models use same hyperparameters

## Reference: TRM Training Settings

TRM also uses lr=1e-4, epochs=40,000, batch_size=768 but does not overfit because:
- Much smaller model (5-7M params)
- Model size itself acts as regularizer

## How to Run

```bash
# On cluster
module load slurm
cd ~/HMR

# Retrain SHREK with new settings
sbatch models/SHREK-HRM/script/train/AblationStudy/train_shrek_large_sudoku_v2.sh
sbatch models/SHREK-HRM/script/train/AblationStudy/train_shrek_tiny_sudoku_v2.sh

# Monitor
squeue -u thheim
tail -f /home/thheim/HMR/logs/shrek_*_JOBID.log
```
