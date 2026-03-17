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

## What SHREK Adds (Three New Components)

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
learned_err = sigmoid(error_estimator(mean(z_H)))
```

- A small neural network that reads z_H and predicts how wrong the model is
- Trained with an auxiliary loss: predicted error should match real loss (min-max normalized)
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
delta = ‖z_H_end − z_H_start‖ / (‖z_H_start‖ + ε)
```

This scalar is appended to the Q-head input:

```
HRM:    q_logits = q_head(z_H[:, 0])                    # hidden_size → 2
SHREK:  q_logits = q_head(concat(z_H[:, 0], delta))     # hidden_size+1 → 2
```

The Q-head learns: *low delta + low error = done, halt. Low delta + high error = stuck in wrong answer, keep going.*

### 3. Q-target Caching (50% Training Compute Saving)

In Q-learning, the Q-head needs a "target" to learn from. HRM computes this by running the full model **twice** per training step — once for the actual output, and once just to get next-step Q-values for the target. This doubles training compute.

SHREK removes the second forward pass. Instead, it caches the Q-values from each step in the carry state. At the next ACT step, the cached Q-values from the **previous** step are used as the target — a 1-step delayed target, similar to how DQN (Deep Q-Network) uses a delayed target network for stable training.

```
HRM:    target = sigmoid(Q_values from second forward pass)     # expensive
SHREK:  target = sigmoid(Q_values cached from previous step)    # free
```

This saves ~50% training compute with no cost at inference time.

---

## Full Forward Pass (One ACT Step)

```
1. Reset carry for halted sequences (clean initial state)
2. Save previous step's cached Q-values (for delayed Q-target)
3. Inner reasoning loop (with gradient on last step only):
     for each H-step:
         for each L-step:
             z_L = L_level(z_L, z_H + input_embeddings)
         z_H = H_level(z_H, z_L)
4. Decode output: logits = lm_head(z_H)
5. Compute error signal (flip rate + learned estimator)
6. Inject error into z_H:  z_H = z_H + alpha × error_encoder(error) / sqrt(hidden_size)
7. Compute stagnation delta from z_H change
8. ACT halt decision: q_head(concat(z_H[:,0], delta)) → halt or continue
9. Store z_H and Q-values in carry for next step
```

Steps 5-6 happen AFTER the inner loop — the error injection modifies z_H for the next ACT step, not the current one's output.

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
