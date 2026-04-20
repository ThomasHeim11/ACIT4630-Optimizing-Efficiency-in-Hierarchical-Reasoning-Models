# SHREK V3 — Self-Gated Error Injection for TRM

> **One-liner for the group:** Where SHREK V1 (on HRM) injected a uniform, schedule-based error correction into every sample plus a stagnation-delta signal, SHREK V3 (on TRM) makes the correction per-sample and self-regulating — the model's own uncertainty decides how much to inject, so hard samples get full correction and confident ones get none — and drops the stagnation-delta component entirely after ablation showed it was destabilizing.

## Motivation

SHREK V2 ran a full ablation on Sudoku-Extreme (Large, 50k steps, lr=1e-4, batch=768, EMA):

| Variant                              | Peak exact_accuracy             | Final lm_loss |
| ------------------------------------ | ------------------------------- | ------------- |
| No_Both (baseline TRM + EMA)         | **0.75**                        | 0.22          |
| No_Stagnation (error injection only) | 0.73                            | 0.22          |
| No_Error (stagnation delta only)     | 0.73 (unstable, 0.2 dip at 34k) | 0.23          |
| Full (both components)               | 0.67                            | 0.28          |

Baseline TRM beat every SHREK V2 variant. The ablation tells us exactly why — and points to a principled fix.

---

## Diagnosis

### Why stagnation delta (blue) is unstable

1. **Distribution shift.** `||z_H_after − z_H_before||` is large early (state changes a lot) and near-zero late (state stabilizes). The Q-head sees a feature whose meaning shifts across training. It cannot learn a stable mapping.
2. **Redundant on Sudoku.** Q-head already hits 99% halt accuracy without help. Extra noisy information degrades a clean signal.
3. **Positive feedback.** Q-head → halt decision → z_H trajectory → delta → Q-head. Small perturbations amplify. The catastrophic dip at 34k (0.7 → 0.2) is this feedback loop.

### Why combining both (red) is worst

- Error injection writes into z_H: `z_H += alpha · error_emb / √h`
- Stagnation reads z_H movement: `delta = ||z_H_after − z_H_before||`
- **Stagnation is measuring the noise error injection just added.** The signal is corrupted by construction. Q-head learns garbage.

### Why error injection alone (green) plateaus below baseline

- Green **led from 10k to 44k steps** (0.70 at 30k vs baseline's 0.65) — error injection genuinely accelerates learning
- But alpha=0.01 never turns off. Constant perturbation prevents clean convergence. The model converges, but noise keeps it from settling. Baseline catches up at the end.

**Key insight:** error injection works as regularization but needs to back off as the model learns. A fixed alpha is the wrong interface.

---

## The SHREK V3 Update

Two changes, both simplifications:

1. **Drop `flip_err` from the error signal.** Use only the learned uncertainty estimate.
2. **Gate injection magnitude by the same learned estimate.** One signal drives both content and strength.

```python
# SHREK V2 (current — two-signal + scheduled alpha):
flip_err = (current_pred != prev_pred).float().mean(-1)
learned_err = sigmoid(error_estimator(z_H_mean.detach()))
error = 0.5 * flip_err + 0.5 * learned_err
alpha_t = alpha_max * min(1.0, step / warmup_steps)
z_H = z_H + alpha_t * error_encoder(error) / sqrt(hidden_size)

# SHREK V3 (new — single signal, per-sample gated):
learned_err = sigmoid(error_estimator(z_H_mean.detach()))          # (B,)
error_emb = error_encoder(learned_err.unsqueeze(-1))               # (B, h)
alpha_per_sample = alpha_max * learned_err.detach().clamp(0, 1)    # (B,)
z_H = z_H + alpha_per_sample.view(-1, 1, 1) * error_emb.unsqueeze(1) / sqrt(hidden_size)
```

### Why drop `flip_err`

`flip_err` measures how many tokens changed prediction between consecutive steps. Pros: works from step 1 without training, directly observable. Cons dominate in V3:

- **Discrete and coarse.** 1 token flipping vs 10 flipping are different realities; the averaged rate obscures this.
- **Late-training noise.** Once the model is near-converged, flips are rare and driven by numerical noise, not uncertainty. The signal starts injecting noise at random moments instead of at genuinely-uncertain moments.
- **First-step artifact.** On a reset sample, `prev_pred = 0` → flip_rate is artificially high for one step regardless of actual difficulty.
- **Contradicts the gate.** When `learned_err` is low (model confident), the gate closes. But a single numerical flip then spikes `flip_err`, and whatever injection _does_ get through is pure noise. The two signals work against each other late in training.
- **Redundant with the gate.** The gate already uses `learned_err`. Reusing it for the error signal makes the mechanism coherent — one uncertainty measure controls everything.

`learned_err` alone is continuous, calibrated to actual LM loss via `aux_loss`, trends smoothly to zero as training converges, and has an aligned training objective.

### Why gate by `learned_err`

- **Zero new hyperparameters.** Uses the existing estimator.
- **Self-regulating.** When uncertain, inject full. When confident, inject near zero. No schedule, no decay.
- **Per-sample adaptive.** Hard puzzles get more perturbation, easy ones get none — gating at batch-element granularity.
- **Works on any run length.** No dependency on epoch count or hand-tuned windows.
- **Theoretical interpretation.** Regularization strength is inversely proportional to the model's own estimate of how wrong it is.

### Why this addresses V2's failures

- **Fixes green's plateau:** `learned_err → 0` late in training, so injection vanishes automatically. Green's early lead + clean late convergence → should cross baseline's 0.75.
- **Removes the broken components:** stagnation delta dropped entirely. Warmup schedule replaced by self-gating. `flip_err` replaced by pure `learned_err`.

---

## What We Remove

| Removed                                                      | Why                                                                                                                                                                    |
| ------------------------------------------------------------ | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `enable_stagnation_delta` flag and code path in `trm.py`     | Fully removed. Ablation proved it harms stability (positive feedback). V2 reproducibility lives on the `SHREKV2` branch.                                               |
| Stagnation delta input to Q-head (`+1` on input dim)         | Q-head now always takes plain `z_H[:, 0]` of size `hidden_size`                                                                                                        |
| `flip_err` component of the error signal                     | Discrete, noisy late in training, redundant with the gating signal, contradicts the gate on random numerical flips                                                     |
| `alpha_warmup_steps` hyperparameter                          | Self-gating replaces it. `learned_err` starts ~0.5 under random init, so early alpha is already half-strength. Estimator trains within hundreds of steps via aux_loss. |
| 3× ablation scripts (`no_error`, `no_stagnation`, `no_both`) | Ablation is complete; keep the `full` script for reference only                                                                                                        |

V2 reproducibility: the stagnation-delta code still exists on the `SHREKV2` branch. Check out that branch to re-run the V2 ablation; V3 is deliberately clean.

---

## Implementation Plan

### Step 1 — Modify `trm.py` ✅

- [x] Remove `alpha_warmup_steps` from `TinyRecursiveReasoningModel_ACTV1Config`
- [x] Remove the `_alpha_step` buffer (warmup counter)
- [x] Drop `flip_err` computation — use only `learned_err` as the error signal:
  ```python
  learned_err = torch.sigmoid(self.error_estimator(z_H_mean.float())).squeeze(-1)  # (B,)
  error_emb = self.error_encoder(learned_err.unsqueeze(-1))                        # (B, h)
  ```
- [x] Replace scalar alpha with per-sample gated alpha (same signal as the error content):
  ```python
  alpha_per_sample = self.config.alpha_max * learned_err.detach().clamp(0, 1)
  z_H = z_H + alpha_per_sample.view(-1, 1, 1) * error_emb.unsqueeze(1) / math.sqrt(self.config.hidden_size)
  ```
- [x] Remove `enable_stagnation_delta` flag and the stagnation code path entirely (V2 reproducibility lives on the `SHREKV2` branch)
- [x] Leave the aux loss in `losses.py` unchanged — the error estimator still trains via MSE against normalized LM loss

### Step 2 — New configs ✅

- [x] `config/arch/trm_shrekv3_normal.yaml` — **SHREK Normal** (~7M, matches TRM-Att). hidden=512, num_heads=8, error injection on, `alpha_max=0.01`
- [x] `config/arch/trm_shrekv3_tiny.yaml` — **SHREK Tiny** (~2M, ~13× smaller than original HRM). hidden=256, num_heads=4, same SHREK V3 components

Parameter overhead from SHREK on top of vanilla TRM: `error_encoder` (1→h) + `error_estimator` (h→1) ≈ 1.5k params on Normal. Effectively a **free upgrade** in parameter-count terms.

### Step 3 — Training scripts (4 total: {Normal, Tiny} × {Sudoku, Maze}) ✅

Two model sizes × two tasks. Normal (~7M, hidden=512) matches TRM-Att's parameter count; Tiny (~2M, hidden=256) targets the "beats 27M HRM with ~13× fewer params" story.

- [x] `script/train/train_shrekv3_normal_sudoku.sh` — `arch=trm_shrekv3_normal`, Sudoku-Extreme, epochs=40000, batch=768, lr=1e-4, L_cycles=6, ema=True
- [x] `script/train/train_shrekv3_normal_maze.sh` — `arch=trm_shrekv3_normal`, Maze-30x30-hard, epochs=20000, batch=128, lr=1e-4, L_cycles=4, ema=True
- [x] `script/train/train_shrekv3_tiny_sudoku.sh` — `arch=trm_shrekv3_tiny`, Sudoku-Extreme, epochs=40000, batch=768, lr=1e-4, L_cycles=6, ema=True
- [x] `script/train/train_shrekv3_tiny_maze.sh` — `arch=trm_shrekv3_tiny`, Maze-30x30-hard, epochs=20000, batch=128, lr=1e-4, L_cycles=4, ema=True

Hyperparameters mirror the V2 ablation scripts so comparison is apples-to-apples. No V3 ablation variants — the V2 ablation already settled which components matter; V3 is the single clean config.

### Step 4 — Push and train

- [ ] Commit V3 changes on the `ShrekV3` branch (trm.py edits, 2 configs, 4 scripts, plan doc)
- [ ] Push `ShrekV3` branch to origin
- [ ] On cluster: `git fetch origin && git checkout ShrekV3`
- [ ] Submit 4 jobs:
  ```bash
  module load slurm
  cd ~/HMR
  sbatch models/SHREK-HRMV2/script/train/train_shrekv3_normal_sudoku.sh
  sbatch models/SHREK-HRMV2/script/train/train_shrekv3_normal_maze.sh
  sbatch models/SHREK-HRMV2/script/train/train_shrekv3_tiny_sudoku.sh
  sbatch models/SHREK-HRMV2/script/train/train_shrekv3_tiny_maze.sh
  ```
- [ ] Monitor on W&B (`SHREKV3_Sudoku` and `SHREKV3_Maze` projects) and via `squeue -u thheim`

### Step 5 — Evaluate

- [ ] Compare SHREK V3 Sudoku vs baseline TRM (0.75) and SHREK V2 Full (0.67)
- [ ] Compare SHREK V3 Maze vs baseline TRM (0.87) and SHREK V2 Full
- [ ] FLOPs measurement on best checkpoint
- [ ] Update paper with V3 results

---

## Expected Results

| Model        | Params | Sudoku baseline | Maze baseline   | SHREK V3 target   |
| ------------ | ------ | --------------- | --------------- | ----------------- |
| SHREK Normal | ~7M    | 0.75 (TRM-Att)  | 0.87 (TRM-Att)  | **>0.77 / >0.88** |
| SHREK Tiny   | ~2M    | 0.55 (HRM 27M)  | 0.745 (HRM 27M) | **>0.55 / >0.75** |

**Paper claims if targets hit:**

- Normal (~7M) beats TRM-Att (~7M) on both tasks — self-correction as a free upgrade
- Tiny (~2M) beats original HRM (27M) with **~13× fewer parameters**

The V2 ablation proved error injection can accelerate learning (green led for 34k/50k steps). Removing its ceiling via self-gating should turn that mid-training lead into a final-accuracy win.

---

## Files to Modify

1. `models/SHREK-HRMV2/models/recursive_reasoning/trm.py` — Step 1 (modified) ✅
2. `models/SHREK-HRMV2/config/arch/trm_shrekv3_normal.yaml` — Step 2 (new)
3. `models/SHREK-HRMV2/config/arch/trm_shrekv3_tiny.yaml` — Step 2 (new)
4. `models/SHREK-HRMV2/script/train/train_shrekv3_normal_sudoku.sh` — Step 3 (new)
5. `models/SHREK-HRMV2/script/train/train_shrekv3_normal_maze.sh` — Step 3 (new)
6. `models/SHREK-HRMV2/script/train/train_shrekv3_tiny_sudoku.sh` — Step 3 (new)
7. `models/SHREK-HRMV2/script/train/train_shrekv3_tiny_maze.sh` — Step 3 (new)

## Estimated Time

- Implementation: ~1 hour
- Training: ~24 hours on GH200 (Sudoku 50k + Maze 20k in parallel on two nodes)
- Evaluation: ~1 hour
- Total: ~1.5 days
