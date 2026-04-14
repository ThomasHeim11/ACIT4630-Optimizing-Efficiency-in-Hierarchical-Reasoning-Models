# SHREK 2.0 — Changes to Improve Accuracy and Reduce FLOPs

## Problem

1. SHREK models overfit on Sudoku after ~25k steps (accuracy crashes from 65% to ~8%)
2. SHREK Tiny on Maze overfits after ~50k steps (73% drops to ~57%)
3. SHREK Tiny is currently 14M params (hidden_size=512) instead of the intended ~7M
4. Current SHREK uses HRM's 8-GPU settings (lr=1e-4) on 1 GPU — likely too aggressive

## Changes

### 1. SHREK Tiny — Reduce to ~7M Parameters

Current: `hidden_size=512, num_heads=8, 2H+2L layers` → 14M params
Target: `hidden_size=256, num_heads=4, 2H+2L layers` → ~3.4M params

In `config/arch/shrek_tiny.yaml`:

```yaml
hidden_size: 256 # reduced from 512
num_heads: 4 # reduced from 8 (keeps head_dim=64)
```

Even smaller than TRM (~5-7M). Strongest possible "less is more" claim if it works.
Previous attempt with hidden_size=256 caused training collapse — lower LR (7e-5) might fix this.

### 2. Hyperparameter Tuning — Prevent Overfitting

SHREK is HRM-sized (14-27M), so it should use HRM's 1-GPU recipe, not TRM's.
TRM gets away with lr=1e-4 because it's tiny (5-7M) — less capacity to overfit.

**What we actually trained HRM and TRM with (our scripts):**

| Setting         | HRM Sudoku | HRM Maze | TRM Sudoku | TRM Maze |
| --------------- | ---------- | -------- | ---------- | -------- |
| lr              | 7e-5       | 1e-4     | 1e-4       | 1e-4     |
| batch_size      | 384        | 128      | 768        | 128      |
| epochs          | 20,000     | 20,000   | 40,000     | 20,000   |
| EMA             | No         | No       | Yes        | Yes      |
| weight_decay    | 1.0        | 1.0      | 1.0        | 1.0      |

**SHREK current vs proposed:**

**SHREK Sudoku — current vs proposed:**

| Setting           | SHREK (current) | SHREK 2.0 (proposed) | Reason                                      |
| ----------------- | --------------- | -------------------- | ------------------------------------------- |
| lr                | 1e-4            | 7e-5                 | Match HRM Sudoku 1-GPU recipe               |
| global_batch_size | 768             | 384                  | Match HRM Sudoku 1-GPU recipe               |
| epochs            | 40,000          | 40,000               | Keep same as TRM for fair comparison         |
| EMA               | Yes             | Yes                  | Keep — helps stability                       |
| weight_decay      | 1.0             | 1.0                  | Same across all models                       |

**SHREK Maze — current vs proposed:**

| Setting           | SHREK (current) | SHREK 2.0 (proposed) | Reason                                      |
| ----------------- | --------------- | -------------------- | ------------------------------------------- |
| lr                | 1e-4            | 1e-4                 | Same as HRM Maze and TRM Maze               |
| global_batch_size | 768             | 128                  | Match HRM Maze and TRM Maze                 |
| epochs            | 20,000          | 20,000               | Same as all models                           |
| EMA               | Yes             | Yes                  | Keep — helps stability                       |
| weight_decay      | 1.0             | 1.0                  | Same across all models                       |

Apply to ALL 7 training scripts:

**Maze (2):**

- `models/SHREK-HRM/script/train/train_shrek_large_maze.sh`
- `models/SHREK-HRM/script/train/train_shrek_tiny_maze.sh`

**Ablation Sudoku (5):**

- `models/SHREK-HRM/script/train/AblationStudy/train_shrek_large_sudoku_vanilla_full.sh`
- `models/SHREK-HRM/script/train/AblationStudy/train_shrek_large_sudoku_vanilla_no_error.sh`
- `models/SHREK-HRM/script/train/AblationStudy/train_shrek_large_sudoku_vanilla_no_stagnation.sh`
- `models/SHREK-HRM/script/train/AblationStudy/train_shrek_large_sudoku_vanilla_no_both.sh`
- `models/SHREK-HRM/script/train/AblationStudy/train_shrek_tiny_sudoku_vanilla.sh`

### 3. Update W&B Project Names to V2

Rename `project_name` in all scripts to track V2 runs separately in W&B:

| Script | Current project_name | New project_name |
|---|---|---|
| `train_shrek_large_maze.sh` | HRM_Maze_Comparison | HRM_Maze_Comparison_V2 |
| `train_shrek_tiny_maze.sh` | HRM_Maze_Comparison | HRM_Maze_Comparison_V2 |
| `train_shrek_large_sudoku_vanilla_full.sh` | SHREK_Ablation_Sudoku | SHREK_Ablation_Sudoku_V2 |
| `train_shrek_large_sudoku_vanilla_no_error.sh` | SHREK_Ablation_Sudoku | SHREK_Ablation_Sudoku_V2 |
| `train_shrek_large_sudoku_vanilla_no_stagnation.sh` | SHREK_Ablation_Sudoku | SHREK_Ablation_Sudoku_V2 |
| `train_shrek_large_sudoku_vanilla_no_both.sh` | SHREK_Ablation_Sudoku | SHREK_Ablation_Sudoku_V2 |
| `train_shrek_tiny_sudoku_vanilla.sh` | SHREK_Ablation_Sudoku | SHREK_Ablation_Sudoku_V2 |

### 4. Additional Options

- Early stopping (manual: pick best checkpoint from W&B)
