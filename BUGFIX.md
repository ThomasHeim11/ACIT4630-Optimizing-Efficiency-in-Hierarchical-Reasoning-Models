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
