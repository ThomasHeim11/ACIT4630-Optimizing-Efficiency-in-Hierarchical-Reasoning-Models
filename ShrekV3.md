# SHREK V3 — Self-Gated Error Injection for TRM

> **One-liner for the group:** Where SHREK V1 (on HRM) injected a uniform, schedule-based error correction into every sample plus a stagnation-delta signal, SHREK V3 (on TRM) makes the correction per-sample and self-regulating — the model's own uncertainty decides how much to inject, so hard samples get full correction and confident ones get none — and drops the stagnation-delta component entirely after ablation showed it was destabilizing.

## Plain-English Explanation for the Group

### What SHREK does

Our model is TRM with one extra mechanism bolted on: **it predicts which cells it's unsure about, then nudges itself to reconsider those cells**. Think of it like a student re-reading the hardest question on an exam instead of all of them.

### What we changed (the patch)

**Problem 1 — the Maze model was collapsing.**
It got to 85.5% then crashed to 52%. Why? A bug in how gradients flowed: the model's "uncertainty meter" was accidentally being trained by the wrong signal, so it drifted until it started cranking up corrections on cells that were already correct, which broke the output.
→ **Fix:** we cut that wrong gradient path. The uncertainty meter now only learns from its proper training signal.

**Problem 2 — the correction was too broad.**
Before the patch, the model computed *one* uncertainty score per puzzle and applied the same correction to every cell. That's wasteful: in Sudoku, ~25 cells are already given (trivial), and in Maze, ~850 cells are just walls (trivial). They don't need correction.
→ **Fix:** the model now computes uncertainty **per cell**. Easy cells get zero correction, hard cells get full correction. Surgical instead of spray-and-pray.

### Why this should beat TRM

- **Stable training:** no more collapse, so it keeps improving past the point where it used to crash
- **Smarter regularization:** only the hard cells get touched, so the model doesn't keep disturbing cells it's already solved
- **Self-tuning:** no hyperparameters to tweak — the model decides per cell how much help it needs

### Expected outcome

- Sudoku: around 0.78–0.80 (TRM gets 0.75)
- Maze: around 0.88–0.90 (TRM gets 0.87)
- Tiny variant (~2M params) should beat original HRM (27M) — same quality, 13× smaller

Zero new configs, zero new hyperparameters — just smarter use of the mechanism we already had.

---

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

---

## V3.1 — Stability + Per-Token Gating (combined patch)

### What we observed in V3

The first SHREK V3 Normal Maze run reached a healthy peak of `exact_accuracy = 0.855` at step ~45-55k, then **collapsed to 0.52 by step 70k**. `lm_loss` rose from 0.028 → 0.031+ over the same window. The Q-head stayed healthy (`q_halt_accuracy` still climbing), so this is not a halting failure — the LM output itself degraded.

V3 also has a *structural* limitation independent of the collapse: the gate is **per-sample**, which means easy and hard tokens inside the same puzzle receive identical perturbation. On Sudoku this wastes injection on the ~25 given cells; on Maze it wastes it on the ~850 wall/empty cells. To decisively beat both baselines we need **per-token** gating.

V3.1 addresses both issues in a single patch: a stability fix (prevents collapse) plus a mechanism upgrade (targets regularization at the cell level).

### Fix 1 — Stability: detach `learned_err` before injection

**Root cause of the collapse.** The V3 forward pass detached `learned_err` only when computing `alpha`, not when computing `error_emb`. LM loss backpropagated through `error_encoder` → `learned_err` → `error_estimator.weight`, giving the estimator two training signals:

1. `aux_loss = 0.1 · MSE(learned_err, normalized_lm_loss)` — the intended objective
2. Main LM loss, flowing back through the injection path — unintended

Late in training both losses are tiny, so (1)'s gradient shrinks while (2) keeps nudging the estimator in unrelated directions. The estimator drifts → spikes → alpha spikes → z_H off-manifold → LM loss rises → estimator learns higher values → cascade.

**Fix:** detach `learned_err` fully before any injection use.

```python
learned_err = sigmoid(error_estimator(z_H_mean.detach()))   # keeps grad → aux_loss trains it
learned_err_det = learned_err.detach()                      # used for injection — no grad path to estimator
error_emb = error_encoder(learned_err_det.unsqueeze(-1))    # LM loss cannot reach estimator
alpha = alpha_max * learned_err_det.clamp(0, 1)             # no extra safety clamp; keep full range
z_H += alpha * error_emb / sqrt(h)
```

`alpha_max` stays at `0.01`; no schedule, no tighter clamp. The detach alone closes the cascade path while preserving early-training acceleration. (A tighter clamp or lower alpha would add safety at the cost of regularization strength — unnecessary once the gradient leak is gone.)

### Fix 2 — Accuracy: per-token gating instead of per-sample

**V3 (per-sample, current).** The estimator takes the *mean* z_H over all token positions and outputs one scalar per puzzle:

```
z_H: (B, 81, h)  →  mean over positions  →  z_H_mean: (B, h)  →  Linear(h, 1)  →  learned_err: (B,)
```

That single gate value applies equally to every cell. Given cells in Sudoku (trivial) get the same perturbation as empty cells (hard). Wall cells in Maze (trivial) get the same perturbation as branch-point path cells (hard).

**V3.1 (per-token).** Drop the mean. Apply the Linear at each position:

```
z_H: (B, L, h)  →  Linear(h, 1) per-position  →  learned_err: (B, L)
```

Each cell gets its own uncertainty score, which drives its own alpha. Easy cells → gate closes → near-zero injection. Hard cells → gate opens → full regularization. Same mechanism, finer-grained. Works on *both* tasks because both tasks have intra-puzzle difficulty variance.

### Combined V3.1 forward pass (full code for the injection block)

```python
# SHREK V3.1: per-token self-gated error injection, gradient-isolated estimator
if self.config.enable_error_injection:
    # Per-token uncertainty — drop the mean, apply estimator at each position.
    z_H_tokens = z_H[:, self.puzzle_emb_len:].detach()  # (B, L, h)
    learned_err = torch.sigmoid(self.error_estimator(z_H_tokens.float())).squeeze(-1)  # (B, L)

    # Fully detach before injection — estimator is trained only by aux_loss.
    learned_err_det = learned_err.detach()

    # Per-token error content and per-token gate.
    error_emb = self.error_encoder(learned_err_det.unsqueeze(-1))  # (B, L, h)
    alpha_per_token = self.config.alpha_max * learned_err_det.clamp(0, 1)  # (B, L)

    # Inject only into the non-puzzle-embedding positions (the actual tokens).
    scale = math.sqrt(self.config.hidden_size)
    z_H_tokens_new = z_H[:, self.puzzle_emb_len:] + alpha_per_token.unsqueeze(-1) * error_emb / scale
    z_H = torch.cat([z_H[:, :self.puzzle_emb_len], z_H_tokens_new], dim=1)
```

### Fix 3 — `losses.py`: per-token aux loss

The aux loss must also become per-token so the estimator's training target aligns with its per-position output.

```python
# V3 (per-sample):
per_sample_lm_loss = (self.loss_fn(...) / loss_divisor).sum(-1)   # (B,)
normalized_lm_loss = (per_sample_lm_loss - lm_min) / (lm_max - lm_min + 1e-8)
aux_loss = 0.1 * F.mse_loss(outputs["learned_err"], normalized_lm_loss.detach(), reduction="sum")

# V3.1 (per-token):
per_token_lm_loss = (self.loss_fn(...) / loss_divisor)             # (B, L)
with torch.no_grad():
    valid = (labels != IGNORE_LABEL_ID)                            # (B, L)
    lm_max = per_token_lm_loss[valid].max().clamp(min=1e-8)
    normalized_lm_loss = (per_token_lm_loss / lm_max).clamp(0, 1)
    normalized_lm_loss = torch.where(valid, normalized_lm_loss, torch.zeros_like(normalized_lm_loss))
aux_loss = 0.1 * F.mse_loss(outputs["learned_err"], normalized_lm_loss.detach(), reduction="sum")
```

Invalid positions (padding) get target 0 so the estimator learns to output near-zero on those tokens and stay out of the way.

### Why this combination addresses "generalize well on both tasks"

| Task | V3 limitation | V3.1 remedy |
|------|---------------|-------------|
| Sudoku | Per-sample gate applies uniform noise to all 81 cells including ~25 given cells that are trivial. Over-regularizes; Sudoku plateaus. | Per-token gate closes on given cells (zero injection), opens on hardest empty cells. Precision regularization. |
| Maze | Per-sample gate applies uniform noise to all 900 cells including ~850 walls/empty space. Wastes >90% of the injection budget. Plus the cascade causes late collapse. | Per-token gate closes on walls, opens on path branch points. Detach fix eliminates the collapse. |

Same mechanism, same hyperparameters, adapts per-task automatically via the estimator. This is the generalization property we want.

### Why we do NOT re-add `flip_err`

- The V3 collapse was caused by gradient bleed-through, not by `flip_err`'s absence. Adding it back does not fix the cascade.
- `flip_err` is discrete and noisy late in training (a single numerical flip spikes it) — the exact reason we dropped it for V3.
- V3.1's per-token `learned_err` already gives us what `flip_err` tried to provide (observable per-position uncertainty), but smooth and continuous.

### Expected behavior

- **Stability:** No late collapse. Estimator cannot be dragged around by LM loss; aux loss is its only trainer.
- **Sudoku:** Gate closes on given cells immediately (they are trivially predicted). Regularization concentrates on empty cells with real uncertainty. Expected peak: **>0.77** (vs baseline 0.75, V2-error-only 0.73).
- **Maze:** Gate closes on walls/free space. Regularization concentrates on path branch points. With no collapse, training continues past the 55k peak. Expected peak: **>0.87** (vs baseline 0.87 at 150k steps; V3 hit 0.855 at 55k steps with per-sample gate).
- **Tiny variants:** Same mechanism, smaller backbone. Targets: beat 27M HRM (0.55 Sudoku / 0.745 Maze) with ~2M params.

### Files changed in V3.1

1. `models/SHREK-HRMV2/models/recursive_reasoning/trm.py` — per-token estimator, full detach before injection, per-token injection into token positions only
2. `models/SHREK-HRMV2/models/losses.py` — per-token normalized LM loss target for aux loss
3. Configs **unchanged** (`alpha_max=0.01` preserved in both normal and tiny). Training scripts unchanged.

### Action plan

- [ ] Cancel any running V3 jobs: `scancel 1042873 1042874 1042875 1042876`
- [ ] Apply the code changes in `trm.py` and `losses.py`
- [ ] Local smoke-check (YAML parses, no syntax errors)
- [ ] Commit + push on the `ShrekV3` branch
- [ ] On cluster: `git pull` and resubmit the 4 training scripts
- [ ] Watch `exact_accuracy` through late training — should climb and hold (not peak+collapse)

---

## V3.2 — Explore-then-Commit (skip injection on the final ACT step)

### What we observed in V3.1

V3.1 fixed the catastrophic V3 collapse (Maze 0.855→0.52). Stability improved drastically. But on the final 50k-step Sudoku and 150k-step Maze runs, two patterns remain:

| Run | Final exact_accuracy | TRM baseline | Gap | Late-training symptom |
|---|---|---|---|---|
| **Normal Maze** | 0.86 (peak 0.88) | 0.87 | +0.01 / −0.01 | One large dip to 0.31 at step ~70k; smaller oscillation throughout |
| **Normal Sudoku** | 0.67 | 0.71 | −0.04 | `all.q_halt_loss` rising 0.005 → 0.06 in last 20k steps; exact_accuracy plateau |

Both runs share the same diagnosis: **`q_halt_loss` rises late in training**. This is residual Q-learning instability — smaller than the V3 cascade, but still drags down the final number. When Q-targets drift, the halt head sometimes stops too early (delivering a wrong-but-confident answer) or runs the full ACT budget when it shouldn't, and both cases hurt eval accuracy.

### Root cause: Q-head sees a moving target at decision time

Each ACT step ends with `z_H` *plus* a per-token injection `alpha · error_emb / √h`. The Q-head reads `z_H[:, 0]` to decide halt vs continue. When `learned_err > 0` on uncertain cells, the injected noise propagates through the network and reaches `z_H[:, 0]`. The Q-head is trying to learn:

> "Given this state, should I commit or keep reasoning?"

But the *state itself* is being perturbed at the moment of decision. The Q-target Q-learning bootstraps from is built from a noisy `z_H` and a noisy next-step `z_H`. With no target network and no replay buffer, those noisy targets compound — the q_halt_loss rises slowly even after V3.1's gradient fix.

The Q-head needs a clean, stationary signal at the decision step.

### Mechanism: explore early, commit late

Inside one inner forward call (one ACT step), the model runs `H_cycles × L_cycles` of reasoning, then injects noise into `z_H`, then the LM head and Q-head read the result. We want to keep injection during exploration but disable it on the *final* ACT step — the step where the model commits its answer.

Concretely:

```
# Pseudocode (placement inside _Inner.forward, after the reasoning loops)
if enable_error_injection:
    learned_err   = sigmoid(error_estimator(z_H_tokens.detach()))   # (B, L) — for aux loss
    alpha         = alpha_max * learned_err.detach().clamp(0, 1)    # (B, L)
    if is_last_step is not None:
        # Zero alpha for samples on their commit step → no injection on those samples.
        alpha = alpha * (~is_last_step).to(alpha.dtype).view(-1, 1)
    z_H = z_H + alpha · error_encoder(learned_err) / sqrt(h)
```

`is_last_step: (B,) bool` is computed in the outer ACT wrapper (where the step counter lives) and passed into the inner forward. It's true for any sample whose step counter is about to reach `halt_max_steps`. During eval, all samples run the full budget, so it fires on the literal final inner call. During training, it fires only for samples that have used their full ACT budget without halting.

### Why this works (three reasons)

1. **Q-head decisions become Q-head trainable.** On the commit step, `z_H[:, 0]` is the unmodified output of the reasoning trunk. Q-target = Q on that clean state. With consistent, low-variance Q-targets, `q_halt_loss` stops rising, halting decisions become reliable, and the late-training dip vanishes.
2. **No accuracy is lost.** All exploration steps (1 .. halt_max_steps − 1) still get full per-token injection — the regularization mechanism that drove V3.1's early gains continues to operate. We're only removing the *one* perturbation that was directly contaminating the decision input.
3. **It's intrinsic to the design, not a hyperparameter band-aid.** "Explore with noise, commit without it" is a classic stochastic-search pattern (e.g., simulated annealing, ε-greedy). It composes naturally with self-gating: easy cells have alpha ≈ 0 throughout (gate closed), hard cells get full alpha during exploration (gate open) but alpha = 0 on commit (override).

### Why this addresses BOTH Maze and Sudoku

| Task | V3.1 issue | V3.2 expected effect |
|------|-------------|----------------------|
| **Maze** | Big dip at 70k tied to `q_halt_loss` spike to 0.95. Q-head temporarily makes wrong halt decisions, accuracy crashes to 0.31 before recovering. | Halt decisions trained on clean states stop spiking. Single-shot dips disappear. Final accuracy converges near peak (~0.88) instead of oscillating. Clean win over TRM 0.87. |
| **Sudoku** | `q_halt_loss` rises 0.005→0.06 over last 20k steps. Halt timing degrades on hardest puzzles. Plateau at 0.67. | Stable halt timing late in training → model uses correct ACT budget per puzzle → harder puzzles get the full 16 steps without injection contamination on the final step → exact_accuracy keeps climbing past 0.67 toward TRM's 0.71+. |

Same compute budget. Same epochs. One mechanism change.

### Why we are NOT doing other things

- **NOT changing alpha_max** — self-gating already does the right per-cell modulation; the issue isn't injection magnitude on hard cells, it's injection *timing* relative to the halt decision.
- **NOT changing `epochs`** — fair-comparison constraint stands. SHREK must beat TRM at TRM's compute budget to be a clean claim.
- **NOT adding a target network or replay buffer** — those are big architectural changes; the explore-then-commit fix is a 3-line change that addresses the same instability with much less code.
- **NOT decaying alpha across ACT steps within a single forward** — coarser version of the same idea; clean on/off at the commit step is simpler and more interpretable.
- **NOT reintroducing flip_err** — orthogonal to the Q-stability problem.

### Files to change

1. `models/SHREK-HRMV2/models/recursive_reasoning/trm.py`
   - `_Inner.forward` signature: add `is_last_step: Optional[torch.Tensor] = None`
   - Inside the injection block: zero `alpha_per_token` for samples where `is_last_step` is true
   - Outer `forward`: compute `will_be_last_step = (new_steps + 1) >= halt_max_steps` and pass to inner

No config changes. No script changes. No new hyperparameters. Same 4 training scripts resubmit as-is.

### Expected outcome

| Run | V3.1 result | V3.2 target | TRM | Verdict if hit |
|-----|-------------|-------------|-----|----------------|
| Normal Maze | 0.86 (peak 0.88) | **0.88-0.90 stable** | 0.87 | Clean win |
| Normal Sudoku | 0.67 | **0.71-0.74** | 0.71 | At parity or win |
| Tiny variants | underperform HRM | (out of scope for V3.2) | HRM 0.745/0.55 | Move to limitations |

The Tiny size class is left as a documented limitation — V3.2 targets the Normal-vs-TRM comparison only.

### Action plan

- [ ] Apply the trm.py changes (add `is_last_step` parameter + gate alpha)
- [ ] Local syntax check
- [ ] Commit + push on `ShrekV3` branch
- [ ] On cluster: `git pull` and resubmit Normal Sudoku + Normal Maze
- [ ] Watch `q_halt_loss` — should stay flat or decrease, not rise
- [ ] Watch `all.exact_accuracy` — should climb monotonically without single-shot dips
