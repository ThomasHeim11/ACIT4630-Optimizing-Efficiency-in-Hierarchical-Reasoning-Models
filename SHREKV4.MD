# SHREK V4 — Stabilize V1 with Cosine Decay, Drop Stagnation Delta

Folder: `models/SHREK-HRMv4/` (copied from V1 `SHREK-HRM`)
Base: HRM architecture + EMA + stablemax CE + universal flip-rate signal
Goal: turn V1's collapsing 0.66 peak into a stable end-of-training result, **at the same epoch budget as V1** (so the comparison is apples-to-apples)

---

## What we learned from V1, V2, V3 (rationale for V4)

- **V1 (SHREK on HRM)** peaked at ~0.66 Sudoku but collapsed to ~0.13 by step 50k. The "+13 over published HRM" gain was almost entirely from the training stack (EMA + stablemax + hyperparameters), **not** the SHREK mechanisms.
- **V1 ablation**: error injection alone ≈ baseline (0.66), stagnation delta alone hurt by ~2pt (0.64), both combined matched stagnation alone — so **stagnation delta is harmful** and **error injection is neutral**.
- **V2 (SHREK on TRM)** showed mechanisms don't transfer to TRM either — pure TRM (No_Both) was best.
- **V3.4 (cosine decay on TRM)** showed that scheduled magnitude decay prevents late-training corruption of converged states.

V4 combines these findings: keep the parts that work, drop the parts that don't, add the schedule that fixes late collapse.

---

## V4 changes (3 mechanism changes — same training budget)

### Mechanism

1. **Drop `enable_stagnation_delta` entirely.** Remove the code path, the flag, the Q-head input concatenation. It cost 2pt in V1 ablation and adds complexity.
2. **Keep error injection** (universal flip-rate + learned `error_estimator` + `error_encoder`). It's neutral on average in V1, but it's the novel mechanism — and the cosine schedule below should turn it from neutral to positive.
3. **Replace alpha warmup with pure cosine decay schedule.**
   - V1 had: linear warmup `0 → 0.01` over 5k steps, then constant `0.01` forever
   - V4: pure cosine decay `0.01 → 0` over total training steps. No warmup phase.
   - Formula: `alpha(t) = alpha_max * 0.5 * (1 + cos(π * t / total_steps))`
   - Cosine is flat near both ends, so it gives gentle injection early without needing a separate warmup, and `α = 0` exactly at end-of-training.
   - Reason: V3.4 showed pure cosine prevents injection from corrupting converged states. Same approach used here.

### Training (UNCHANGED from V1 — apples-to-apples comparison)

- **Sudoku epochs: 40000** (same as V1)
- **Maze epochs: same as V1**
- All hyperparameters identical to V1: lr=1e-4, weight_decay=1.0, batch=768 (Sudoku) / 128 (Maze), EMA=True, stablemax CE
- The whole point of V4 is to show that cosine decay alone fixes the collapse — not that we shortened training to avoid it.

### Cleanup

4. **Remove dead code**: `prev_q_halt` / `prev_q_continue` in carry are leftover from a caching attempt and not actually used. Drop them.

---

## Why V4 should beat V1 — the failure mode in plain words

V1's chart shows all variants peak ~0.66 around step 25k, then collapse to 0.08–0.13 by step 50k. The collapse happens because:

1. Model converges around step 25k
2. Injection magnitude stays at 0.01 forever (constant)
3. Constant noise into a converged model corrupts the representations
4. lm_loss climbs back up, exact_accuracy crashes

V4's bet: **the late collapse is caused by injection persisting past convergence.** Cosine decay fixes this directly — strong injection while learning, zero at convergence.

| Step range | V1 behavior | **V4 expected behavior** |
|---|---|---|
| 0 → 5k | Model learns rapidly, alpha ramping up | Same learning, alpha at full strength (cosine flat near start) |
| 5k → 25k | Climbs to peak ~0.66 | Same climb, alpha decaying gently |
| 25k (peak) | Model converged | Model converged |
| 25k → 40k | **Constant α=0.01 keeps injecting noise → corrupts converged model → collapse to 0.13** | **Cosine α tapering to 0 → injection fades → converged model preserved → plateau at ~0.66** |

V3.4 already proved this works on TRM (eliminated the late-stage Maze cliff). Same fix applied here.

---

## What V4 is NOT changing (intentionally)

These all stay because they're working:

- HRM architecture (2 cycles each H/L, 4 layers each, hidden=512)
- Universal flip-rate error signal (`error_signals.py`)
- `error_estimator` + `error_encoder` heads
- Aux MSE loss training the estimator
- EMA, stablemax CE
- All hyperparameters: lr=1e-4, weight_decay=1.0, batch=768/128, 40k Sudoku epochs
- Commit-step protection (zero injection on the final ACT step)

We're making the **smallest possible change** that addresses the documented failure mode. Cleanest scientific story.

---

## Files to modify

1. `models/SHREK-HRMv4/models/hrm/hrm_act_v1.py`
   - Remove `enable_stagnation_delta` flag and all conditional code
   - Remove Q-head input expansion (`hidden_size + 1`) — back to `hidden_size`
   - Remove `prev_q_halt` / `prev_q_continue` from carry dataclass
   - Remove `_alpha_step` warmup buffer
   - Replace alpha computation with pure cosine decay (no warmup)
   - Add `_current_step` and `_total_steps` attributes (driven from `pretrain.py`)
   - Add `set_train_progress(current_step, total_steps)` method

2. `models/SHREK-HRMv4/pretrain.py`
   - Add `_set_train_progress` helper that walks `_orig_mod` → `model` → `inner`
   - Call it once per step before forward (drives the cosine schedule)

3. `models/SHREK-HRMv4/config/arch/shrek_large.yaml`
   - Remove `enable_stagnation_delta: true` line
   - Keep `enable_error_injection: true`
   - Keep `alpha_max: 0.01`
   - Remove `alpha_warmup_steps` — no warmup with pure cosine decay

4. `models/SHREK-HRMv4/config/arch/shrek_small.yaml` (renamed from `shrek_tiny.yaml`)
   - Same removal of stagnation delta
   - Bump architecture to **7M params** so it's directly comparable to TRM-Att (7M)

5. `models/SHREK-HRMv4/models/losses.py`
   - Keep aux MSE loss as-is (trains the estimator)
   - No changes needed — estimator is the contribution

6. Training scripts in `models/SHREK-HRMv4/script/train/`
   - Update `cd ~/HMR/models/SHREK-HRM` → `cd ~/HMR/models/SHREK-HRMv4`
   - Update SBATCH log paths and run names: `shrek_*` → `shrekv4_*`
   - Drop `arch.enable_stagnation_delta=...` flag (no longer exists)
   - **Keep `epochs=40000` for Sudoku and the original Maze epoch count** — same budget as V1

---

## Implementation order

### Phase 1 — code changes (~1 hour) ✅

- [x] **1a.** In `hrm_act_v1.py`: remove `enable_stagnation_delta` and all guarded code paths
- [x] **1b.** In `hrm_act_v1.py`: remove `prev_q_halt` / `prev_q_continue` fields from carry dataclass + their init/reset/return paths
- [x] **1c.** In `hrm_act_v1.py`: change alpha computation to pure cosine decay (no warmup)
- [x] **1d.** In `hrm_act_v1.py`: add `_current_step`, `_total_steps`, `set_train_progress(step, total)` method (also added on the ACT wrapper, which delegates to `_Inner`)
- [x] **1e.** In `pretrain.py`: add `_set_train_progress` helper and call it before each forward
- [x] **1f.** In `shrek_large.yaml`: remove `enable_stagnation_delta`. Rename `shrek_tiny.yaml` → `shrek_small.yaml`. Architecture already targets ~7M (hidden=512, 2+2 layers).

### Phase 2 — script updates (~30 min) ✅

- [x] **2a.** Created `train_shrekv4_large_sudoku_full.sh` — paths → `SHREK-HRMv4`, no stagnation flag, run name `SHREKV4_Large_Sudoku_Full`, epochs=40000.
- [x] **2b.** Created `train_shrekv4_large_maze_full.sh` — same updates, epochs=20000 (matches V1 Maze budget).
- [x] **2c.** Created `train_shrekv4_large_sudoku_no_error.sh` and `train_shrekv4_large_maze_no_error.sh` — HRM+EMA baseline (no error injection) on both tasks.
- [x] **2d.** Created `train_shrekv4_small_sudoku_full.sh` and `train_shrekv4_small_maze_full.sh` — V4 Small (~7M) on both tasks.

### Phase 3 — local sanity check (~10 min) ✅

- [x] **3a.** `python3 -c "import ast; ast.parse(...)"` passes on `hrm_act_v1.py` and `pretrain.py`.
- [x] **3b.** `grep -rn "enable_stagnation_delta\|prev_q_halt\|prev_q_continue\|alpha_warmup_steps\|_alpha_step"` returns nothing — clean.

### Phase 4 — push and train

- [ ] **4a.** Commit + push on `ShrekV4` branch
- [ ] **4b.** On cluster: `module load slurm`, `cd ~/HMR && git pull origin ShrekV4`
- [ ] **4c.** Submit Sudoku: `sbatch ~/HMR/models/SHREK-HRMv4/script/train/train_shrekv4_large_sudoku.sh`
- [ ] **4d.** Submit Maze: `sbatch ~/HMR/models/SHREK-HRMv4/script/train/train_shrekv4_large_maze.sh`
- [ ] **4e.** Monitor `squeue -u thheim` and W&B

### Phase 5 — evaluate

- [ ] **5a.** Compare V4 final accuracy to V1 final at same step count (target: V4 final ≥ V1 peak ≈ 0.66 Sudoku, 0.83 Maze)
- [ ] **5b.** Confirm no late-training collapse — the main thing we're fixing
- [ ] **5c.** Report peak + final + end-of-training stability in the paper
- [ ] **5d.** If Maze still shows late drift → tune cosine schedule shape (e.g. start decay earlier)

---

## What to watch on W&B

- `all.exact_accuracy` — should **plateau, not collapse**. Should hold its peak through end of training (40k for Sudoku).
- `all.lm_loss` — should bottom and stay there. V1 lm_loss climbed back up late — that should disappear.
- `all.q_halt_loss` — should be flat. V1 had spikes due to stagnation delta corrupting Q-head; without it, should be clean.
- `aux_loss` — should drop as model converges. Sanity check that estimator is calibrated.
- Compare cleanly against `Ablation_No_Both` from V1 (HRM + EMA + stablemax baseline). V4 needs to beat that **at end of training**, not just at peak.

---

## Expected outcome (same epoch budget as V1)

| Task | Published HRM | V1 No_Both peak | V1 Full final (40k) | **V4 target final (40k)** |
|------|---------------|-----------------|----------------------|---------------------------|
| Sudoku | 0.53 | 0.66 | ~0.08 (collapse) | **0.65–0.70 stable** |
| Maze   | 0.745 | ? | drift / drop | **0.83+ stable** |

The paper story if this works:
> "We identify late-training collapse as the dominant failure mode of HRM-based models on small datasets. Combining a flip-rate-driven error injection mechanism with a cosine-decayed magnitude schedule eliminates this collapse and stabilizes end-of-training accuracy at 13+ points above published HRM, **using the same training budget as V1**."

---

## Decisions made

- [x] **SHREK Small at 7M params** (renamed from "Tiny") — directly comparable to TRM-Att (7M).
- [x] **Ablation scope**: 2 variants only —
  1. **HRM + EMA** (no error injection)
  2. **HRM + EMA + error injection (with cosine decay)** ← V4 main result
- [x] **HRM baseline (53%)**: cited from TRM paper (Wang et al.) — no retraining needed.

### Ablation matrix

| Variant | EMA | Error Injection | Cosine α decay | Source |
|---|---|---|---|---|
| Published HRM | ✗ | ✗ | ✗ | TRM paper (cite, 53%) |
| **V4 No_Error** (Large + Small) | ✓ | ✗ | n/a | New runs |
| **V4 Full** (Large + Small) | ✓ | ✓ | ✓ | **New main result** |

Total cluster jobs: 4 Large (2 tasks × 2 variants) + 2 Small full (2 tasks) = **6 jobs**. If leaner first pass needed, run V4 Full only (Large + Small on both tasks = 4 jobs), add No_Error baseline after.

### Training scripts to create

In `models/SHREK-HRMv4/script/train/`:

1. `train_shrekv4_large_sudoku_full.sh` — V4 main, Sudoku, 40k epochs
2. `train_shrekv4_large_maze_full.sh` — V4 main, Maze, original epoch count
3. `train_shrekv4_large_sudoku_no_error.sh` — HRM+EMA baseline, Sudoku
4. `train_shrekv4_large_maze_no_error.sh` — HRM+EMA baseline, Maze
5. `train_shrekv4_small_sudoku_full.sh` — V4 Small (7M), Sudoku
6. `train_shrekv4_small_maze_full.sh` — V4 Small (7M), Maze

### Config files to update

- `config/arch/shrek_large.yaml` — drop stagnation flag, keep large
- `config/arch/shrek_small.yaml` — rename from `shrek_tiny.yaml`, target 7M (not 3M)