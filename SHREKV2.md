# SHREK V2 — Self-Correction for TRM Attention (Best Model)

## Why TRM Attention as Base

From the TRM paper (Table 4):

| Model   | Params | Sudoku | Maze  |
| ------- | ------ | ------ | ----- |
| HRM     | 27M    | 55.0%  | 74.5% |
| TRM-Att | 7M     | 74.7%  | 85.3% |
| TRM-MLP | 5M/19M | 87.4%  | 0.0%  |

TRM-MLP scores 87.4% on Sudoku but **0% on Maze** (MLP can't handle variable-length sequences).
TRM Attention is the **best overall model** — works on both Sudoku (74.7%) and Maze (85.3%).

We build on **TRM Attention only** because:

- Best accuracy across both tasks
- Trains stably (no overfitting at 40-50k epochs)
- Small model (~7M params)
- Room to improve: W&B shows noisy Q-head halt decisions (~20-30% halt accuracy)
  and unstable training steps — exactly what SHREK's stagnation delta can fix

SHREK V1 proved error injection works (+12% on HRM). Applying it to TRM-Att could push both Sudoku and Maze higher.

## Goal

Add SHREK's self-correction (error injection + stagnation delta) to TRM Attention.
Keep TRM's hyperparameters unchanged — only add ~1K SHREK parameters.

Two model variants:

| Variant        | hidden_size | num_heads | Params | Compared to           |
| -------------- | ----------- | --------- | ------ | --------------------- |
| SHREK V2 Large | 512         | 8         | ~7M    | TRM-Att (7M)          |
| SHREK V2 Small | 256         | 4         | ~3.4M  | Smallest model tested |

SHREK V1 at hidden=256 collapsed on HRM architecture, but TRM is simpler (2 layers, no hierarchy)
and trains more stably — hidden=256 has a better chance of working here.
If 3.4M SHREK V2 Small beats 27M HRM, that's an 8x parameter reduction.

---

## Step-by-Step Implementation

### Step 1: Create config files ✅

- [x] **1a.** Create `models/SHREK-HRMV2/config/arch/trm_shrek.yaml`
  - Copy `trm.yaml`
  - Add SHREK flags: `enable_error_injection: true`, `enable_stagnation_delta: true`
  - Add: `alpha_max: 0.01`, `alpha_warmup_steps: 5000`
  - Keep: `hidden_size: 512`, `num_heads: 8` (same as TRM-Att)

- [x] **1b.** Create `models/SHREK-HRMV2/config/arch/trm_shrek_small.yaml`
  - Copy `trm_shrek.yaml`
  - Change: `hidden_size: 256`, `num_heads: 4`

### Step 2: Add config flags to TRM config class ✅

- [x] **2a.** Open `models/SHREK-HRMV2/models/recursive_reasoning/trm.py`
- [x] **2b.** Add to `TinyRecursiveReasoningModel_ACTV1Config`:
  ```python
  enable_error_injection: bool = False
  enable_stagnation_delta: bool = False
  alpha_max: float = 0.01
  alpha_warmup_steps: int = 5000
  ```

### Step 3: Add prev_pred to carry dataclass ✅

- [x] **3a.** Add `prev_pred: torch.Tensor` to `TinyRecursiveReasoningModel_ACTV1InnerCarry`
- [x] **3b.** Update `empty_carry()` — add `prev_pred=torch.zeros(batch_size, config.seq_len, dtype=torch.int32)`
- [x] **3c.** Update `reset_carry()` — reset `prev_pred` to zeros when halted

### Step 4: Add error components to Inner.**init**() ✅

- [x] **4a.** Add `error_encoder = nn.Linear(1, config.hidden_size)` (if error injection enabled)
- [x] **4b.** Add `error_estimator = nn.Linear(config.hidden_size, 1)` (if error injection enabled)
- [x] **4c.** Add `_alpha_step = nn.Buffer(torch.tensor(0.0), persistent=True)` (warmup counter)
- [x] **4d.** Update `q_head` input size: `hidden_size + 1` if stagnation delta enabled, else `hidden_size`

### Step 5: Add error injection to Inner.forward() ✅

- [x] **5a.** Compute flip rate from output vs carry.prev_pred
- [x] **5b.** Compute learned error from error_estimator(z_H_mean.detach())
- [x] **5c.** Combine: error = 0.5 _ flip_err + 0.5 _ learned_err
- [x] **5d.** Inject into z_H with alpha warmup: z_H += alpha \* error_emb / sqrt(hidden_size)
- [x] **5e.** Add stagnation delta to Q-head input (if enabled)
- [x] **5f.** Store current_pred in new carry
- [x] **5g.** Return learned_err via outputs dict in ACT wrapper
- [x] **5h.** Updated ACT wrapper to unpack 4th return value from inner.forward()
- [x] **5i.** Fixed no_ACT_continue=False path to unpack 4 values (not 5)

### Step 6: Add aux loss ✅

Added to `models/SHREK-HRMV2/models/losses.py` (not pretrain.py — loss is computed in losses.py):

- [x] **6a.** Compute per-sample LM loss and normalize to [0, 1]
- [x] **6b.** Add aux MSE loss: `0.1 * mse(learned_err, normalized_lm_loss)`
- [x] **6c.** Add to total loss: `lm_loss + 0.5 * (q_halt + q_continue) + aux_loss`
- [x] **6d.** Log `aux_loss` in metrics for W&B tracking
- [x] **6e.** All gated by `if "learned_err" in outputs` — no effect on vanilla TRM

### Step 7: Create ablation training scripts (Large, Sudoku) ✅

- [x] **7a.** Create `train_shrekv2_large_sudoku_full.sh` — error injection ON, stagnation delta ON
- [x] **7b.** Create `train_shrekv2_large_sudoku_no_error.sh` — error injection OFF, stagnation delta ON
- [x] **7c.** Create `train_shrekv2_large_sudoku_no_stagnation.sh` — error injection ON, stagnation delta OFF
- [x] **7d.** Create `train_shrekv2_large_sudoku_no_both.sh` — both OFF (= baseline TRM with EMA)

### Step 8: Create Maze training scripts (Large) ✅

- [x] **8a.** Create `train_shrekv2_large_maze.sh`
  - `arch=trm_shrek`, full SHREK, L_cycles=4 (Maze uses 4 not 6)
  - lr=1e-4, batch=128, epochs=20k, EMA=True
  - project_name=SHREKV2_Maze

### Step 9: Create Small model training scripts ✅

- [x] **9a.** Create `train_shrekv2_small_sudoku.sh`
  - `arch=trm_shrek_small`, full SHREK, H_cycles=3, L_cycles=6
  - lr=1e-4, batch=768, epochs=40k, EMA=True

- [x] **9b.** Create `train_shrekv2_small_maze.sh`
  - `arch=trm_shrek_small`, full SHREK, H_cycles=3, L_cycles=4
  - lr=1e-4, batch=128, epochs=20k, EMA=True

### Step 11: Push and train on cluster

- [ ] **11a.** Push SHREKV2 branch to GitHub
- [ ] **11b.** Pull on cluster: `git fetch origin && git checkout SHREKV2`
- [ ] **11c.** Submit all 7 jobs:
  ```bash
  module load slurm
  cd ~/HMR
  # Ablation (4 scripts)
  sbatch models/SHREK-HRMV2/script/train/train_shrekv2_large_sudoku_full.sh
  sbatch models/SHREK-HRMV2/script/train/train_shrekv2_large_sudoku_no_error.sh
  sbatch models/SHREK-HRMV2/script/train/train_shrekv2_large_sudoku_no_stagnation.sh
  sbatch models/SHREK-HRMV2/script/train/train_shrekv2_large_sudoku_no_both.sh
  # Maze
  sbatch models/SHREK-HRMV2/script/train/train_shrekv2_large_maze.sh
  # Small
  sbatch models/SHREK-HRMV2/script/train/train_shrekv2_small_sudoku.sh
  sbatch models/SHREK-HRMV2/script/train/train_shrekv2_small_maze.sh
  ```
- [ ] **11d.** Monitor: `squeue -u thheim` and check W&B

### Step 12: Evaluate results

- [ ] **12a.** Check W&B for peak `all.exact_accuracy` for each run
- [ ] **12b.** Compare ablation results (which component helps TRM most?)
- [ ] **12c.** Compare SHREK V2 Large vs baseline TRM-Att
- [ ] **12d.** Check if SHREK V2 Small (3.4M) beats Original HRM (27M)
- [ ] **12e.** Run FLOPs measurement on best checkpoints
- [ ] **12f.** Generate updated bubble charts

---

## Files to Modify

1. `models/SHREK-HRMV2/models/recursive_reasoning/trm.py` — Steps 2-5
2. `models/SHREK-HRMV2/pretrain.py` — Step 6

## Files to Create

**Configs:**

1. `models/SHREK-HRMV2/config/arch/trm_shrek.yaml` — Large (h=512)
2. `models/SHREK-HRMV2/config/arch/trm_shrek_small.yaml` — Small (h=256)

**Ablation scripts (Large, Sudoku):** 3. `models/SHREK-HRMV2/script/train/train_shrekv2_large_sudoku_full.sh` 4. `models/SHREK-HRMV2/script/train/train_shrekv2_large_sudoku_no_error.sh` 5. `models/SHREK-HRMV2/script/train/train_shrekv2_large_sudoku_no_stagnation.sh` 6. `models/SHREK-HRMV2/script/train/train_shrekv2_large_sudoku_no_both.sh`

**Maze + Small scripts:** 7. `models/SHREK-HRMV2/script/train/train_shrekv2_large_maze.sh` 8. `models/SHREK-HRMV2/script/train/train_shrekv2_small_sudoku.sh` 9. `models/SHREK-HRMV2/script/train/train_shrekv2_small_maze.sh`

## Estimated Time

- Implementation: ~2 hours (Steps 1-10)
- Training: ~70-100 hours on GH200 (~3-4 days sequential, faster if both nodes available)
- Evaluation: ~2 hours (FLOPs + charts)
- Total: ~4-5 days
