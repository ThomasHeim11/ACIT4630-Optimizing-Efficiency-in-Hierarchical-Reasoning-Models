# SHREK-HRM Bug Fixes and Improvements

Three bugs were found and fixed, plus three stability improvements added.

---

## Bug 1: Model Too Small (expansion=2 → 4)

**Problem:** SHREK used `expansion=2` in the FFN layers, halving the model's capacity compared to HRM's `expansion=4`. The model didn't have enough "brain power" to learn Sudoku.

**Result:** lm_loss plateaued near random guessing. 0% puzzles solved.

**Fix:** Changed `expansion` to `4` in both `shrek_large.yaml` and `shrek_tiny.yaml` to match HRM baseline.

---

## Bug 2: Q-Target Used Wrong Step (Same-Step → Previous-Step)

**Problem:** The Q-head learns "should I keep thinking or stop?" To learn this, it needs a target from the *next* state. Original HRM ran the model twice to get this. SHREK was supposed to cache Q-values from the *previous* step as a cheaper alternative (like DQN). But the code accidentally used Q-values from the *current* step — the model was chasing its own output.

**Result:** `q_continue_loss` rose forever instead of decreasing. The Q-head never learned when to halt.

**Fix:** Save Q-values *before* the inner forward pass, then use those saved values as the target. Now the delay works as intended.

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
