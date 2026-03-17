# SHREK-HRM Architecture

**Stagnation Halting Reasoning Error Kernel — Hierarchical Reasoning Model**

SHREK-HRM is a ~27M parameter recursive reasoning model built on top of HRM (Hierarchical Reasoning Model). It adds learned self-correction so the model knows when its answer is wrong and when it is stuck.

---

## Background

HRM is a small (27M parameter) recursive reasoning model that solves puzzles like Sudoku, Mazes, and ARC-AGI by thinking in a loop. At each step it updates its hidden state and tries to improve its answer. A Q-head decides when to stop thinking and output the final answer.

```
HRM            → reasons in a loop, but gets stuck in wrong answers
AugmentedHRM   → adds data augmentation and random noise on reset
SHREK          → replaces random noise with learned error feedback
```

---

## The Problem SHREK Solves

HRM and AugmentedHRM have no way to know if their current answer is wrong. They see the same frozen input at every step. This causes:

- **Wrong fixed-point traps** — the model converges to a wrong answer and stays there
- **Aimless cycling** — the model loops through states without converging

SHREK fixes this by giving the model two new signals: "how wrong am I?" and "am I making progress?"

---

## SHREK's Three New Components

### 1. Error-Conditioned Input Injection

After each reasoning step, SHREK computes an error score and injects it into the hidden state `z_H`:

```
HRM:    z_H stays the same after the inner loop
SHREK:  z_H = z_H + alpha × error_encoder(error)
```

The error signal combines two complementary signals:

**Signal A — Flip rate** (no learning needed, works from step 1):

```
flip_rate = fraction of output tokens that changed vs previous step
```

- High flip rate → model is oscillating between different answers
- Low flip rate → model has settled (but might be stuck on the wrong answer)

**Signal B — Learned error estimator** (catches what flip rate misses):

```
learned_err = sigmoid(error_estimator(mean(z_H)))
```

- A small neural network that reads z_H and predicts how wrong the model is
- Trained with an auxiliary loss: predicted error should match real loss
- Catches "confident but wrong" — the model is not changing its answer but it is still incorrect

**Combined:**

```
error = 0.5 × flip_rate + 0.5 × learned_err
```

The error is mapped to a vector by `error_encoder` (a linear layer, `1 → hidden_size`) and scaled by a learned gate `alpha` (starts at 0.01, grows during training). The auxiliary loss uses min-max normalization so the error estimator gets a useful 0-1 target signal even when all samples have similar loss.

### 2. Stagnation-Aware ACT (Q-head)

The Q-head decides "should I halt or continue?" SHREK gives it extra information — how much did `z_H` change this step:

```
delta = ‖z_H_end − z_H_start‖ / (‖z_H_start‖ + ε)
```

This scalar is appended to the Q-head input:

```
HRM:    q_logits = q_head(z_H[:, 0])                    # hidden_size → 2
SHREK:  q_logits = q_head(concat(z_H[:, 0], delta))     # hidden_size+1 → 2
```

The Q-head learns: _low delta + low error = done, halt. Low delta + high error = stuck, keep going._

### 3. Q-target Caching (50% Training Compute Saving)

In Q-learning, the Q-head needs a "target" to learn from. HRM computes this by running the full model **twice** per training step — once for the actual output, and once just to get next-step Q-values for the target. This doubles training compute.

SHREK removes the second forward pass. Instead, it caches the Q-values from each step in the carry state. At the next ACT step, the cached Q-values from the **previous** step are used as the target — a 1-step delayed target, similar to how DQN (Deep Q-Network) uses a delayed target network for stable training.

```
HRM:    target = sigmoid(Q_values from second forward pass)     # expensive
SHREK:  target = sigmoid(Q_values cached from previous step)    # free
```

This saves ~50% training compute with no cost at inference time (inference only runs one forward pass regardless).

---

## Full Forward Pass (One ACT Step)

```
1. Reset carry for halted sequences (clean initial state)
2. Save previous step's cached Q-values (for Q-target)
3. Inner reasoning loop (with gradient on last step only):
     for each H-step:
         for each L-step:
             z_L = L_level(z_L, z_H + input_embeddings)
         z_H = H_level(z_H, z_L)
4. Decode output: logits = lm_head(z_H)
5. Compute error signal (flip rate + learned estimator)
6. Inject error into z_H:  z_H = z_H + alpha × error_encoder(error)
7. Compute stagnation delta from z_H change
8. ACT halt decision: q_head(concat(z_H[:,0], delta)) → halt or continue
9. Store z_H and Q-values in carry for next step
```

Steps 5-6 happen AFTER the inner loop — the error injection modifies z_H for the next ACT step, not the current one's output.

---

## Model Variants

| Config      | Hidden | Heads | FFN exp | Params |
| ----------- | ------ | ----- | ------- | ------ |
| SHREK-Large | 512    | 8     | 4       | ~27M   |
| SHREK-Tiny  | 256    | 4     | 4       | ~8M    |

Both use `H_cycles=2, L_cycles=2, H_layers=4, L_layers=4, halt_max_steps=16, expansion=4` — identical base config to HRM for fair comparison.

---

## Parameter Overhead

| Component                               | Size           |
| --------------------------------------- | -------------- |
| Base HRM layers                         | ~27M           |
| Error encoder (`1 → hidden_size`)       | 513            |
| Error estimator (`hidden_size → 1`)     | 513            |
| Stagnation scalar added to Q-head input | +2 weights     |
| Learned gate `alpha`                    | 1              |
| **Total SHREK-Large**                   | **~27M + ~1K** |

Under 0.01% overhead. Same model class and size as HRM for all practical purposes.

---

## Training Setup

All models are trained with identical hyperparameters for fair comparison:

| Setting           | Value                                      |
| ----------------- | ------------------------------------------ |
| Dataset           | Sudoku-Extreme 1K (with hint augmentation) |
| Epochs            | 40,000                                     |
| Eval interval     | 1,000                                      |
| Learning rate     | 1e-4                                       |
| Weight decay      | 1.0                                        |
| Global batch size | 768                                        |
| Optimizer         | AdamATan2                                  |
| GPU               | 1× NVIDIA GH200 (96GB)                     |

Evaluation uses 10-checkpoint ensemble with 9 token permutations, matching the AugmentedHRM methodology.
