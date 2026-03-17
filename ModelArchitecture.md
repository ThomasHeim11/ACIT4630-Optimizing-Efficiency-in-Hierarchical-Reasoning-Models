# SHREK-HRM Architecture

**Self-Correcting Hierarchical Reasoning Compact Model** — ~27M parameters, built on top of AugmentedHRM.

---

## What SHREK Is

SHREK extends AugmentedHRM with two learned components that give the model explicit feedback about whether its current answer is correct and whether it is making progress. Everything else is inherited unchanged from HRM.

```
HRM            → reasons in the dark, gets stuck in wrong fixed points
AugmentedHRM   → adds data augmentation, bootstrapping, random perturbation
SHREK          → keeps aug + bootstrapping, replaces random perturbation with learned self-correction
```

---

## Core Problem

HRM and AugmentedHRM see the same frozen input at every reasoning step, regardless of how wrong the current answer is. This causes:
- **Wrong fixed-point traps** — converges to a stable wrong answer and can't escape
- **Easy-task paradox** — fails on trivially easy inputs more than hard ones
- **Aimless cycling** — loops through the same states without converging

---

## The Two New Components

### 1. Error-Conditioned Input Injection

After each reasoning step, compute a combined error signal and inject it directly into `z_H`:

```
HRM:    z_H unchanged after inner loop
SHREK:  z_H = z_H + alpha × error_encoder(error)
```

The error signal is **universal** — no task-specific rules. It combines two complementary signals:

**Signal A — Flip rate** (works from step 1, no learning needed):
```
flip_rate = fraction of output tokens that changed vs previous step
```
- High → model is oscillating, still searching
- Low → model has settled (correctly or stuck)

**Signal B — Learned error estimator** (catches what flip rate misses):
```
learned_err = sigmoid(error_estimator(z_H_mean))
```
- Reads `z_H` directly and predicts how wrong the model is
- Trained via auxiliary loss using real `lm_loss` as target
- Catches "confident but wrong" — flip rate is low but answer is still incorrect

**Combined 50/50:**
```
error = 0.5 × flip_rate + 0.5 × learned_err
```

The error encoder is a single linear layer (`1 → hidden_size`) with a learned gate scalar `alpha` initialised at `0.01`. Training starts as standard HRM; `alpha` grows as the model learns to use the signal.

### 2. Stagnation-Aware ACT (Q-head)

Measure how much `z_H` changed during this full outer step:

```
delta = ‖z_H_end − z_H_start‖ / (‖z_H_start‖ + ε)
```

Append this scalar to the Q-head input:

```
HRM:    q_logits = q_head(z_H[:, 0])                    # hidden_size → 2
SHREK:  q_logits = q_head(concat(z_H[:, 0], delta))     # hidden_size+1 → 2
```

The model learns: *low delta + low error = halt correctly*, *low delta + high error = fixed-point trap, keep going*.

### 3. Q-target Caching

HRM ran the full model **twice** per outer step to get next-step Q-values for training. SHREK caches Q-values from the current step and reuses them as the target for the next step — same as a DQN target network. This removes one full forward pass per training step (~50% compute saving during training).

---

## Full Forward Pass

```
1. Reset carry to clean initial state (no random noise)
2. For each outer ACT step:
   a. Compute error signal from current prediction → error_embedding_t
   b. Compute stagnation scalar from z_H change
   c. Inner H/L loop:
      for each H-step:
          for each L-step:
              z_L = L_level(z_L, z_H + input_embeddings + error_embedding_t)
          z_H = H_level(z_H, z_L)
   d. Decode output: logits = lm_head(z_H)
   e. ACT decision: q_head(concat(z_H[:,0], stagnation_t)) → halt or continue
```

First step: `stagnation_0 = 1.0` (no prior state), `error_embedding_0 = 0` (no prior prediction).

---

## Model Variants

| Config | Hidden | Heads | FFN exp | Params |
|---|---|---|---|---|
| SHREK-Large | 512 | 8 | 4 | ~27M |
| SHREK-Tiny  | 256 | 4 | 4 | ~8M  |

Both use `H_cycles=2, L_cycles=2, H_layers=4, L_layers=4, halt_max_steps=16, expansion=4` — same as HRM for fair comparison.

---

## Parameter Overhead

| Component | Params |
|---|---|
| Base HRM | ~27M |
| Error encoder (900 → 512) | ~461K |
| Stagnation scalar to Q-head | 2 |
| Learned gate `alpha` | 1 |
| **Total SHREK** | **~27.5M** |

Under 2% overhead. Same model class and size as HRM for all practical purposes.

---

## Computational Complexity (FLOPs)

All four models (HRM, TRM, AugmentedHRM, SHREK) are compared on **GFLOPs per puzzle** — the number of floating point operations for one inference pass on one test input.

**Methodology:** Analytical formula following Kaplan et al. (2020), Table 1. Each multiply-add counts as 2 FLOPs (1 multiply + 1 add). Computed from model config — no runtime measurement needed.

```
FLOPs per token per layer:
  Attention QKV:       2 × d_model × 3d_attn
  Attention scores:    2 × n_ctx × d_attn
  Attention out proj:  2 × d_attn × d_model
  FFN:                 2 × 2 × d_model × d_ff

Total per puzzle = avg_act_steps × H_cycles × (L_cycles × L_block + H_block) × seq_len
```

`avg_act_steps` is the only runtime value — measured as the mean outer ACT steps per puzzle during evaluation.

**Why full formula over the `C ≈ 2N` shortcut:** Although our models satisfy `d_model >> n_ctx/12` (so the shortcut would hold), we use the full per-operation breakdown because `seq_len` differs per benchmark (81 for Sudoku, 900 for ARC/Maze), making the explicit formula more transparent and directly comparable across tasks.

**Source:** Kaplan et al., *Scaling Laws for Neural Language Models*, arXiv:2001.08361, 2020.
