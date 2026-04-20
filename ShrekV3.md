# SHREK V3 — Self-Gated Error Injection for TRM

## Motivation

SHREK V2 ran a full ablation on Sudoku-Extreme (Large, 50k steps, lr=1e-4, batch=768, EMA):

| Variant | Peak exact_accuracy | Final lm_loss |
| --- | --- | --- |
| No_Both (baseline TRM + EMA) | **0.75** | 0.22 |
| No_Stagnation (error injection only) | 0.73 | 0.22 |
| No_Error (stagnation delta only) | 0.73 (unstable, 0.2 dip at 34k) | 0.23 |
| Full (both components) | 0.67 | 0.28 |

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
- **Contradicts the gate.** When `learned_err` is low (model confident), the gate closes. But a single numerical flip then spikes `flip_err`, and whatever injection *does* get through is pure noise. The two signals work against each other late in training.
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

| Removed | Why |
| --- | --- |
| `enable_stagnation_delta` path in `trm.py` | Ablation proved it harms stability (positive feedback) and corrupts combined signal |
| Stagnation delta input to Q-head | Same |
| `flip_err` component of the error signal | Discrete, noisy late in training, redundant with the gating signal, contradicts the gate on random numerical flips |
| `alpha_warmup_steps` hyperparameter | Self-gating replaces it. `learned_err` starts ~0.5 under random init, so early alpha is already half-strength. Estimator trains within hundreds of steps via aux_loss. |
| 3× ablation scripts (`no_error`, `no_stagnation`, `no_both`) | Ablation is complete; keep the `full` script for reference only |

We keep `enable_stagnation_delta` **as a config flag** (default false) so SHREK V2 can still be reproduced for the paper's ablation story.

---

## Implementation Plan

### Step 1 — Modify `trm.py`

- [ ] Remove `alpha_warmup_steps` from `TinyRecursiveReasoningModel_ACTV1Config`
- [ ] Remove the `_alpha_step` buffer (warmup counter)
- [ ] Drop `flip_err` computation — use only `learned_err` as the error signal:
  ```python
  learned_err = torch.sigmoid(self.error_estimator(z_H_mean.float())).squeeze(-1)  # (B,)
  error_emb = self.error_encoder(learned_err.unsqueeze(-1))                        # (B, h)
  ```
- [ ] Replace scalar alpha with per-sample gated alpha (same signal as the error content):
  ```python
  alpha_per_sample = self.config.alpha_max * learned_err.detach().clamp(0, 1)
  z_H = z_H + alpha_per_sample.view(-1, 1, 1) * error_emb.unsqueeze(1) / math.sqrt(self.config.hidden_size)
  ```
- [ ] Keep `enable_stagnation_delta` flag (default false) — gate the stagnation code path on it for backward compatibility with V2 ablation
- [ ] Leave the aux loss in `losses.py` unchanged — the error estimator still trains via MSE against normalized LM loss

### Step 2 — New configs

- [ ] `config/arch/trm_shrekv3.yaml` — hidden=512, error injection on, stagnation off, no warmup
- [ ] `config/arch/trm_shrekv3_small.yaml` — hidden=256

### Step 3 — Training scripts (2 total, clean)

- [ ] `script/train/train_shrekv3_sudoku.sh` — Sudoku-Extreme, 50k epochs, L_cycles=6, batch=768, lr=1e-4, ema=True
- [ ] `script/train/train_shrekv3_maze.sh` — Maze-30x30-hard, 20k epochs (or longer if time allows), L_cycles=4, batch=128, lr=1e-4, ema=True

One config per task. No ablation variants — the V2 ablation already settled which components matter.

### Step 4 — Push and train

- [ ] Push SHREKV2 branch updates (add V3 on top of V2 code)
- [ ] Pull on cluster
- [ ] Submit 2 jobs (Sudoku + Maze)
- [ ] Monitor on W&B

### Step 5 — Evaluate

- [ ] Compare SHREK V3 Sudoku vs baseline TRM (0.75) and SHREK V2 Full (0.67)
- [ ] Compare SHREK V3 Maze vs baseline TRM (0.87) and SHREK V2 Full
- [ ] FLOPs measurement on best checkpoint
- [ ] Update paper with V3 results

---

## Expected Results

| Task | Baseline TRM | SHREK V2 Full | SHREK V3 (target) |
| --- | --- | --- | --- |
| Sudoku-Extreme (1k) | 0.75 | 0.67 | **>0.77** |
| Maze-30x30-hard | 0.87 (paper) | TBD | **>0.88** |

The ablation proved error injection can accelerate learning (green led for 34k/50k steps). Removing its ceiling via self-gating should turn that mid-training lead into a final-accuracy win.

---

## Files to Modify

1. `models/SHREK-HRMV2/models/recursive_reasoning/trm.py` — Step 1
2. `models/SHREK-HRMV2/config/arch/trm_shrekv3.yaml` — Step 2 (new)
3. `models/SHREK-HRMV2/config/arch/trm_shrekv3_small.yaml` — Step 2 (new)
4. `models/SHREK-HRMV2/script/train/train_shrekv3_sudoku.sh` — Step 3 (new)
5. `models/SHREK-HRMV2/script/train/train_shrekv3_maze.sh` — Step 3 (new)

## Estimated Time

- Implementation: ~1 hour
- Training: ~24 hours on GH200 (Sudoku 50k + Maze 20k in parallel on two nodes)
- Evaluation: ~1 hour
- Total: ~1.5 days
