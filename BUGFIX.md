# SHREK-HRM Bug Fixes and Improvements

Five bugs were found and fixed, plus four stability improvements added. Bug 4 went through multiple iterations before arriving at the final solution (Bug 4d).

---

## Bug 1: Model Too Small (expansion=2 → 4)

**Problem:** SHREK used `expansion=2` in the FFN layers, halving the model's capacity compared to HRM's `expansion=4`. The model didn't have enough "brain power" to learn Sudoku.

**Result:** lm_loss plateaued near random guessing. 0% puzzles solved.

**Fix:** Changed `expansion` to `4` in both `shrek_large.yaml` and `shrek_tiny.yaml` to match HRM baseline.

---

## Bug 2: Q-Target Used Wrong Step (Same-Step → Previous-Step) — REVERTED IN BUG 4

**Problem:** The Q-head learns "should I keep thinking or stop?" To learn this, it needs a target from the *next* state. Original HRM ran the model twice to get this. SHREK was supposed to cache Q-values from the *previous* step as a cheaper alternative (like DQN). But the code accidentally used Q-values from the *current* step — the model was chasing its own output.

**Result:** `q_continue_loss` rose forever instead of decreasing. The Q-head never learned when to halt.

**Fix:** Save Q-values *before* the inner forward pass, then use those saved values as the target. Now the delay works as intended.

**NOTE:** This fix was itself incorrect — it made the Q-target look 1 step *backwards* (step T-1) instead of *forwards* (step T+1). See Bug 4 for the correct fix that reverts this change.

---

## Bug 3: Error Estimator Target Always 1.0 (Max Normalization → Min-Max)

**Problem:** The error estimator learns to predict how wrong the model is. Its target was normalized by dividing by `max(loss)`. When all samples had similar loss (common early in training), every target ≈ 1.0. The estimator just learned to always output 1.0 — useless.

**Result:** The error signal gave no useful feedback early in training when the model needed it most.

**Fix:** Changed to min-max normalization: `(loss - min) / (max - min)`. When all losses are the same, target = 0 (correct). When they vary, targets spread across 0-1.

---

## Improvement 1: Scale Error Injection by Hidden Size

**Problem:** SHREK injects an error signal into `z_H` after each reasoning step. The injection has the same absolute magnitude for both Large (`hidden_size=512`) and Tiny (`hidden_size=256`). For Tiny, this is proportionally twice as large — like adding a bucket of water to a small pool vs a large pool. This caused Tiny to learn up to 55% accuracy, then catastrophically forget everything and collapse back to random (11%).

**Result:** SHREK-Tiny's accuracy collapsed at step ~4k every time. SHREK-Large was stable but plateaued at ~53%.

**Fix:** Scale the injection by `1/sqrt(hidden_size)`, following standard transformer scaling conventions. Both models now receive the same *relative* amount of error injection.

**Inspiration:** Standard practice in transformer architectures — attention scores are scaled by `1/sqrt(d_k)` for the same reason (Vaswani et al., "Attention Is All You Need", 2017).

---

## Improvement 2: EMA (Exponential Moving Average)

**Problem:** Small recursive models are sensitive to training noise. A single bad gradient update can cause accuracy to drop suddenly, and the model may never recover ("catastrophic forgetting").

**Result:** SHREK-Tiny showed sudden accuracy collapse. SHREK-Large showed noisy oscillations around its plateau.

**Fix:** Keep a smoothed shadow copy of all model weights, updated slowly each step: `shadow = 0.999 × shadow + 0.001 × real_weights`. Use the shadow weights for evaluation and checkpoints. This prevents sudden collapses because the shadow copy is resistant to individual bad updates.

**Inspiration:** Taken directly from the TRM paper (Jolicoeur-Martineau, "Less is More: Recursive Reasoning with Tiny Networks", 2025), which uses `ema=True` for all experiments. EMA is also widely used in diffusion models, GANs, and modern ML training pipelines.

---

## Improvement 3: Alpha Warmup for Error Injection

**Problem:** SHREK's error injection uses `alpha` to control injection strength. Previously `alpha` started at `0.01` from step 0. But the error estimator is untrained at step 0 — it outputs random values. The model receives random nudges that destabilize small models. SHREK-Tiny consistently collapsed at step ~3k even with scaled injection (Improvement 1), because the error estimator was still inaccurate when it started affecting training.

**Result:** SHREK-Tiny's accuracy rose to 55% then collapsed to 11% at step ~3k, every run.

**Fix:** Replace the fixed `alpha` with a linear warmup schedule: `alpha` ramps from `0.0` to `0.01` over 5000 training steps. During warmup, the error estimator trains via aux_loss (learning what "wrong" looks like) but its output doesn't affect `z_H`. By the time `alpha` reaches full strength, the estimator is accurate and the model is stable.

```
Step 0:      alpha = 0.00   (no injection, model learns normally)
Step 1000:   alpha = 0.002  (tiny nudge, estimator starting to learn)
Step 2500:   alpha = 0.005  (medium nudge, estimator getting accurate)
Step 5000+:  alpha = 0.01   (full strength, estimator is trained)
```

**Inspiration:** Same principle as learning rate warmup, used in all transformer training since Vaswani et al. ("Attention Is All You Need", 2017). Also similar to ReZero (Bachlechner et al., 2020), where residual connections start with a gate at 0 and grow during training.

---

## Improvement 4: Redesign SHREK-Tiny (Wide and Shallow Instead of Thin and Deep)

**Problem:** SHREK-Tiny originally used `hidden_size=256` with 8 layers (4H + 4L) to reach ~7M parameters. Despite all fixes (scaled injection, EMA, alpha warmup), Tiny consistently collapsed at step ~3k — accuracy rose to 55% then crashed to random (11%). The root cause was not the error injection but `hidden_size=256` being too small for each token to represent Sudoku constraints.

**Key insight from TRM:** The TRM paper (Jolicoeur-Martineau, 2025) achieves 7M parameters with `hidden_size=512` and only 2 layers. TRM keeps the hidden state wide (512 dims per token) and saves parameters by using fewer layers, compensating with more reasoning cycles. This works because cycles reuse layers — but nothing can compensate for a hidden_size that's too small.

**Before (thin and deep — collapsed):**
```
hidden_size=256, H_layers=4, L_layers=4  →  ~8M params, collapses at step 3k
```

**After (wide and shallow — stable):**
```
hidden_size=512, H_layers=2, L_layers=2  →  ~8M params, stable training
```

**Fix:** Changed SHREK-Tiny to `hidden_size=512` (same as Large) with `H_layers=2, L_layers=2`. Same parameter count, but each token has enough representation capacity for stable training.

**Inspiration:** TRM architecture design (Jolicoeur-Martineau, "Less is More: Recursive Reasoning with Tiny Networks", 2025).

---

## Bug 4: Q-Target Looked Backwards Instead of Forwards

This bug went through four iterations (4a → 4b → 4c → 4d) before arriving at the correct solution.

### Bug 4a: Root Cause — Q-Target Used Step T-1 Instead of T+1

**Problem:** Bug 2's "fix" introduced a worse bug. The Q-head needs to know: "what's the value of thinking *one more step*?" This requires Q-values from the *next* state (step T+1). The original HRM gets this by running `inner()` a second time (double forward pass). Bug 2's fix changed SHREK to use Q-values cached from the *previous* step (step T-1) — looking backwards instead of forwards, a 2-step discrepancy from the correct target.

```
Original HRM:  target = Q(step T+1)  ← "what if I keep going?"     ✓ correct
Bug 2 "fix":   target = Q(step T-1)  ← "what was it before?"       ✗ backwards
```

**Result:** `q_continue_loss` rose continuously throughout training (0.05 → 0.25 by 50k steps) because the target was nonsensical. The diverging loss sent conflicting gradients through the shared `z_H` representation, causing `lm_loss` to plateau at 2.0 (near random guessing). The model achieved only 38% per-cell accuracy on test and 0% exact puzzle accuracy.

### Bug 4b: First Fix Attempt — Double Forward Pass (Caused New Issues)

**Fix attempt:** Reverted to the original HRM's double forward pass — run `inner()` a second time to get step T+1 Q-values.

```python
next_q_halt, next_q_continue = self.inner(new_inner_carry, new_current_data)[-2]
target = sigmoid(max(next_q_halt, next_q_continue))
```

**New problem:** The second `inner()` call incremented `_alpha_step` a second time per training step, making the alpha warmup complete at step ~2500 instead of 5000. Error injection reached full strength before the error estimator was trained, destabilizing training after step ~6k.

**Evidence from wandb:** `q_continue_loss` initially decreased (steps 0-6k) then rose sharply. `lm_loss` spiked from 1.35 back to 1.8 around step 8k.

**Additional problem:** Removing the now-unused `prev_q_halt`/`prev_q_continue` fields from the carry dataclass caused a `torch.compile` regression — the compiled model produced worse results with a 3-field vs 5-field dataclass. The fields must be kept.

### Bug 4c: Failed Fix — Current-Step Cached Q-Values (Step T)

**Fix attempt:** Use the Q-values that `inner()` writes into `new_inner_carry` after the forward pass. These are from the *current* step (step T) — saves 50% compute by avoiding the second `inner()` call.

```python
inner()  # writes Q-values into new_inner_carry
target = sigmoid(max(new_inner_carry.prev_q_halt, new_inner_carry.prev_q_continue))
```

**Problem:** Step T Q-values are the SAME values as the prediction (one with gradients, one detached). The loss becomes `BCE(q_continue, sigmoid(q_continue.detach()))` — the model trains against itself. This creates a degenerate signal that pushes Q-values toward maximum uncertainty instead of learning useful halt decisions.

**Result:** `q_continue_loss` rose and `lm_loss` stuck at 2.0 — same broken behavior as Bug 4a despite using different Q-values.

### Bug 4d: Final Fix — Double Forward Pass + Alpha Step Save/Restore

**Fix:** Revert to the double forward pass (step T+1, same as original HRM) but fix the `_alpha_step` double increment by saving and restoring the counter around the second `inner()` call.

```python
# Save alpha warmup counter before second inner() call
saved_alpha_step = self.inner._alpha_step.clone()

# Run inner() a second time to get step T+1 Q-values (correct Q-learning target)
next_q_halt, next_q_continue = self.inner(new_inner_carry, new_current_data)[-2]

# Restore counter so warmup advances once per training step, not twice
self.inner._alpha_step.copy_(saved_alpha_step)

target = sigmoid(max(next_q_halt, next_q_continue))
```

**Why this works:**
- Step T+1 Q-values come from a DIFFERENT state than the prediction — proper Q-learning
- `_alpha_step` save/restore prevents double increment — alpha warmup takes 5000 steps as intended
- `prev_q` fields stay in carry (written but not read) — required for `torch.compile` compatibility
- Matches original HRM's Q-target computation exactly

**Summary of all attempts:**
```
Bug 2 "fix":  target = Q(step T-1)  ✗ backwards
Bug 4b:       target = Q(step T+1)  ✓ correct, but _alpha_step double increment
Bug 4c:       target = Q(step T)    ✗ self-target (degenerate learning signal)
Bug 4d:       target = Q(step T+1)  ✓ correct, with _alpha_step save/restore
```

---

## Bug 5: Parasitic Gradient Leakage Through z_H (Two Paths)

**Problem:** After fixing Bugs 1-4, SHREK-HRM was stuck at ~54% token accuracy / 0% exact accuracy after 50k+ training steps. The original AugmentedHRM solves sudoku with the same architecture. Root cause: two gradient paths leaked conflicting optimization signals into the reasoning layers (`H_level`/`L_level`), preventing the model from learning beyond basic cell copying.

### Path A: `aux_loss` → `error_estimator` → `z_H` → reasoning layers

The error estimator reads `z_H_mean = z_H[:, puzzle_emb_len:].mean(dim=1)` and predicts how wrong the model is. Its auxiliary loss trains the estimator to match real `lm_loss`. But because `z_H_mean` was not detached, gradients from `aux_loss` flowed backwards through `z_H` into `H_level` and `L_level`. This told the reasoning layers "predict your own loss" instead of "predict correct tokens" — a parasitic objective that conflicted with `lm_loss`.

**Fix:** Detach `z_H_mean` before the error estimator:
```python
z_H_mean = z_H[:, self.puzzle_emb_len:].mean(dim=1).detach()
```

`aux_loss` still trains the `error_estimator` weights — it just can't corrupt the main model anymore.

### Path B: `q_halt_loss` → stagnation `delta` → `z_H` → reasoning layers

The stagnation delta measures how much `z_H` changed during the current reasoning step: `delta = norm(z_H_end - z_H_start) / norm(z_H_start)`. This delta is fed to the Q-head, and Q-head loss backpropagates through it. Because `z_H` was not detached before computing delta, Q-head gradients flowed through `norm()` over ALL 82 positions of `z_H`. The original HRM only sends Q-head gradients through position 0 (CLS token). This 82x gradient amplification drowned out the `lm_loss` signal.

**Fix:** Detach `z_H` before computing stagnation delta:
```python
z_H_f = z_H.detach().float()
```

Delta remains an informational-only input to the Q-head — it can read the stagnation signal but can't push gradients back through it.

### Why both paths matter

Either path alone is enough to stall training. Path A pulls reasoning layers toward "predict loss" instead of "predict tokens". Path B amplifies Q-head gradients 82x over the correct amount. Together they created a gradient soup where `lm_loss` couldn't make progress. With both paths detached, `lm_loss` is the only signal reaching `H_level`/`L_level` — matching the original HRM's gradient structure exactly.

### Dead code cleanup (same commit)

- Removed `use_default` parameter and duplicate else branch from `reset_carry` — both branches were identical, parameter was never passed as `False`.
- Removed commented-out debug line `# is_last_step = new_steps >= 32`.
