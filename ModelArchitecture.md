# SHREK-HRM Architecture

**Stagnation Halting Reasoning Error Kernel — Hierarchical Reasoning Model**

SHREK-HRM is a recursive reasoning model built on top of HRM (Hierarchical Reasoning Model). It adds learned self-correction so the model knows when its answer is wrong and when it is stuck. All other components are inherited from HRM.

---

## Background

HRM is a small (27M parameter) model that solves puzzles like Sudoku, Mazes, and ARC-AGI by thinking in a loop. At each step it updates its hidden state `z_H` and tries to improve its answer. A Q-head decides when to stop thinking and output the final answer.

The problem: HRM reasons "in the dark" — it never knows if its current answer is right or wrong. It can get stuck on a wrong answer and never escape.

```
HRM            → reasons in a loop, but gets stuck in wrong answers
AugmentedHRM   → adds data augmentation and random noise on reset
SHREK          → replaces random noise with learned error feedback
```

---

## What SHREK Adds (Two New Components)

### 1. Error-Conditioned Input Injection

After each reasoning step, SHREK asks "how wrong am I?" and feeds the answer back into the model. This is called **error injection**:

```
HRM:    z_H stays the same after the inner loop (no feedback)
SHREK:  z_H = z_H + alpha × error_encoder(error) / sqrt(hidden_size)
```

The error signal combines two complementary signals:

**Signal A — Flip rate** (works from step 1, no learning needed):

```
flip_rate = fraction of output tokens that changed vs previous step
```

- High flip rate = model is oscillating between different answers
- Low flip rate = model has settled (correctly or stuck on wrong answer)

**Signal B — Learned error estimator** (catches what flip rate misses):

```
learned_err = sigmoid(error_estimator(mean(z_H).detach()))
```

- A small neural network that reads z_H (detached) and predicts how wrong the model is
- Trained with an auxiliary loss: predicted error should match real loss (min-max normalized)
- The detach ensures aux_loss only trains error_estimator weights, not the reasoning layers (see Bug 5)
- Catches "confident but wrong" — the model stopped changing its answer but it's still incorrect

**Combined:**

```
error = 0.5 × flip_rate + 0.5 × learned_err
```

The error is mapped to a vector by `error_encoder` (a linear layer, `1 → hidden_size`) and scaled by:
- `alpha` — follows a linear warmup schedule from 0 to 0.01 over 5000 steps. This lets the error estimator train before its signal affects the model.
- `1/sqrt(hidden_size)` — makes the injection proportional to model size, preventing small models from being overwhelmed.

### 2. Stagnation-Aware ACT (Q-head)

The Q-head decides "should I halt or continue thinking?" SHREK gives it extra information — a stagnation signal measuring how much `z_H` changed during this reasoning step:

```
delta = ‖z_H_end.detach() − z_H_start‖ / (‖z_H_start‖ + ε)
```

The detach ensures Q-head loss only sends gradients through position 0 (CLS token), matching HRM. Without detach, delta would route Q-head gradients through all 82 positions of z_H, drowning out lm_loss (see Bug 5).

This scalar is appended to the Q-head input:

```
HRM:    q_logits = q_head(z_H[:, 0])                    # hidden_size → 2
SHREK:  q_logits = q_head(concat(z_H[:, 0], delta))     # hidden_size+1 → 2
```

The Q-head learns: *low delta + low error = done, halt. Low delta + high error = stuck in wrong answer, keep going.*

### Q-target (Same as HRM)

SHREK uses the same Q-target computation as the original HRM — a double forward pass. `inner()` runs a second time inside `torch.no_grad()` to get step T+1 Q-values:

```
target = sigmoid(max(Q_halt(step T+1), Q_continue(step T+1)))
```

The `_alpha_step` warmup counter is saved and restored around this call to prevent double-incrementing (see Bug 4d in BUGFIX.md). An earlier approach cached Q-values in the carry for a 50% compute saving, but this produced degenerate Q-targets (see Bug 4 in BUGFIX.md).

---

## What's Identical to HRM

Everything structural is inherited unchanged:

- **Reasoning loop**: H_cycles × L_cycles iterations, `no_grad` except last step (1-step gradient trick)
- **Block architecture**: Attention + SwiGLU + RMSNorm (post-norm), non-causal
- **Token decoding**: `lm_head(z_H)` skipping puzzle embedding prefix
- **Q-target**: Double forward pass for step T+1 Q-values
- **ACT halt logic**: `q_halt > q_continue` with exploration
- **Gradient structure**: `lm_loss` trains all positions of z_H, Q-losses train position 0 only
- **Reset**: Halted sequences reset to learned `H_init`/`L_init` (SHREK removed AugmentedHRM's random noise on reset — error injection replaces it)

---

## Full Forward Pass (One ACT Step)

```
1. Reset carry for halted sequences (clean initial state)
2. Inner reasoning loop (with gradient on last step only):
     for each H-step:
         for each L-step:
             z_L = L_level(z_L, z_H + input_embeddings)
         z_H = H_level(z_H, z_L)
3. Decode output: logits = lm_head(z_H)
4. Compute error signal (flip rate + learned estimator, both detached from z_H)
5. Inject error into z_H:  z_H = z_H + alpha × error_encoder(error) / sqrt(hidden_size)
6. Compute stagnation delta from z_H change (detached from z_H)
7. ACT halt decision: q_head(concat(z_H[:,0], delta)) → halt or continue
8. Double forward pass for Q-target: run inner() again to get step T+1 Q-values
9. Store z_H and Q-values in carry for next step
```

Steps 4-5 happen AFTER the inner loop — the error injection modifies z_H for the next ACT step, not the current one's output.

**Gradient isolation:** The error estimator input (`z_H_mean`) and the stagnation delta (`z_H_f`) are both detached from `z_H` before use. This ensures that only `lm_loss` sends gradients through the reasoning layers (`H_level`/`L_level`). Without detaching, the auxiliary loss and Q-head loss would send parasitic gradients through `z_H` that conflict with the main language modeling objective. See Bug 5 in BUGFIX.md.

---

## Training Stability Features

### Alpha Warmup

The error injection strength (`alpha`) ramps linearly from 0 to 0.01 over 5000 training steps. This gives the error estimator time to train via the auxiliary loss before its predictions affect the model. Without warmup, the untrained estimator injects random noise that can destabilize small models.

```
Step 0:      alpha = 0.00   (no injection, model learns normally)
Step 2500:   alpha = 0.005  (half strength, estimator getting accurate)
Step 5000+:  alpha = 0.01   (full strength, estimator is trained)
```

### EMA (Exponential Moving Average)

A smoothed shadow copy of all model weights is maintained throughout training:

```
shadow_weights = 0.999 × shadow_weights + 0.001 × real_weights
```

The shadow weights are used for evaluation and saved in checkpoints. This prevents sudden accuracy drops from individual bad gradient updates. Same technique used by TRM for all their experiments.

### Scaled Error Injection

The error injection is divided by `sqrt(hidden_size)` so both Large (512-dim) and Tiny (512-dim, fewer layers) receive the same relative injection strength. This follows standard transformer scaling conventions.

---

## Model Variants

| Config      | Hidden | Heads | H_layers | L_layers | FFN exp | Params |
| ----------- | ------ | ----- | -------- | -------- | ------- | ------ |
| SHREK-Large | 512    | 8     | 4        | 4        | 4       | ~27M   |
| SHREK-Tiny  | 512    | 8     | 2        | 2        | 4       | ~8M    |

Both use `H_cycles=2, L_cycles=2, halt_max_steps=16, expansion=4, hidden_size=512`.

SHREK-Large matches HRM's architecture exactly (same layers, same hidden size) with SHREK components added on top.

SHREK-Tiny saves parameters by reducing layers (2+2 vs 4+4) while keeping `hidden_size=512`. This follows TRM's design principle: small recursive models need wide hidden states for stable training. Reducing hidden_size (instead of layers) causes training collapse.

---

## Parameter Overhead

| Component                               | Size       |
| --------------------------------------- | ---------- |
| Base HRM layers (SHREK-Large)           | ~27M       |
| Error encoder (`1 → hidden_size`)       | 513        |
| Error estimator (`hidden_size → 1`)     | 513        |
| Stagnation scalar added to Q-head input | +2 weights |
| **Total SHREK-Large**                   | **~27M + ~1K** |

Under 0.01% overhead. Same model class and size as HRM for all practical purposes.

---

## Training Setup

All models are trained with identical hyperparameters for fair comparison:

| Setting            | Value                                      |
| ------------------ | ------------------------------------------ |
| Dataset            | Sudoku-Extreme 1K (with hint augmentation) |
| Epochs             | 40,000                                     |
| Eval interval      | 1,000                                      |
| Learning rate      | 1e-4                                       |
| LR warmup steps    | 2,000                                      |
| Alpha warmup steps | 5,000                                      |
| Weight decay       | 1.0                                        |
| Global batch size  | 768                                        |
| Optimizer          | AdamATan2                                  |
| EMA rate           | 0.999                                      |
| GPU                | 1x NVIDIA GH200 (96GB)                     |

Evaluation uses 10-checkpoint ensemble with 9 token permutations, matching the AugmentedHRM methodology.
