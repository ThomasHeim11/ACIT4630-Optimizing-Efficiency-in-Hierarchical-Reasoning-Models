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

After each outer ACT step, decode the current prediction, compute an error signal, and inject it into the next inner loop:

```
HRM:    z_L = L_level(z_L,  z_H + input_embeddings)
SHREK:  z_L = L_level(z_L,  z_H + input_embeddings + error_embedding_t)
```

**Error signal per task:**
| Task | Signal |
|---|---|
| Sudoku | Per-cell conflict indicator (row/col/box violations) |
| Maze | Per-cell path validity (wall hits, disconnected path) |
| ARC-AGI | Per-cell output entropy (model uncertainty) |

The error encoder is a single linear layer (`seq_len → hidden_size`) with a learned gate scalar `alpha` initialised near zero. Training starts as standard HRM; `alpha` grows as the model learns to use feedback.

### 2. Stagnation-Aware ACT (Q-head)

Measure how much the model's state changed between outer steps:

```
stagnation_t = ‖z_H_t − z_H_{t−1}‖ / (‖z_H_{t−1}‖ + ε)
```

Append this scalar to the Q-head input:

```
HRM:    q_logits = q_head(z_H[:, 0])                         # hidden_size → 2
SHREK:  q_logits = q_head(concat(z_H[:, 0], stagnation_t))   # hidden_size+1 → 2
```

The model learns: *low stagnation + low error = halt correctly*, *low stagnation + high error = fixed-point trap, keep going*.

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
| SHREK-Large | 512 | 8 | 2 | ~27M |
| SHREK-Tiny  | 256 | 4 | 2 | ~7M  |

Both use `H_cycles=2, L_cycles=2, H_layers=4, L_layers=4, halt_max_steps=16` — same as HRM for fair comparison.

`expansion=2` (vs HRM's 4) halves FFN FLOPs.

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

## Green AI Metrics

All four models (HRM, TRM, AugmentedHRM, SHREK) are evaluated on:
- **Accuracy** per benchmark
- **Energy (kWh)** via CodeCarbon
- **FLOPs/puzzle** computed analytically from model config
- **Time/puzzle** wall-clock, averaged over 100 batches

Key metric: **energy per correct solution** — efficiency normalised by usefulness.
