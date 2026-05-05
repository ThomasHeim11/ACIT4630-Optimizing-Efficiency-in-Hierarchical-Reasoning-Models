# SHREK V4

Folder: `models/SHREK-HRMv4/`
Goal: HRM + cosine-decayed error injection, NoEMA. Ablation = vanilla HRM (cited from TRM paper) vs SHREK V4 NoEMA Full.

## Main run set (4 scripts)

| Model | Task | Script | Epochs |
|---|---|---|---|
| Large (27.28M) | Sudoku | `train_shrekv4_large_sudoku_full_noema.sh` | 40000 |
| Large (27.28M) | Maze | `train_shrekv4_large_maze_full_noema.sh` | 20000 |
| Small (7.09M) | Sudoku | `train_shrekv4_small_sudoku_full_noema.sh` | 40000 |
| Small (7.09M) | Maze | `train_shrekv4_small_maze_full_noema.sh` | 20000 |

All 4 use `+ema=False`, `arch.enable_error_injection=True`. Hyperparameters identical to TRM (lr=1e-4, weight_decay=1.0, batch=768/128).

## TODO

- [ ] Commit + push on `ShrekV4` branch
- [ ] On cluster: `module load slurm`, `cd ~/HMR && git pull origin ShrekV4`
- [ ] Submit all 4 NoEMA scripts via `sbatch`
- [ ] Monitor `squeue -u thheim` and W&B
- [ ] Check Sudoku Large NoEMA at ~35k: does `all.exact_accuracy` hold? If no → next suspect (stablemax → weight_decay).
- [ ] Populate final results table for the paper

## Optional (if time)

- [ ] EMA × task ablation: re-add EMA Full scripts (flip `+ema=False` → `+ema=True`) and run on both tasks for the 2×2.
