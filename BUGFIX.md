# SHREK-HRM Bug Fixes

Three bugs were found and fixed before retraining.

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
