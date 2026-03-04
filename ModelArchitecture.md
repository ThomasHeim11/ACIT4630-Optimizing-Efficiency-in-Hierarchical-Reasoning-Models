# SHREK Model Architecture
## Self-Correcting Hierarchical Reasoning Compact Model

---

## Abstract

SHREK (Self-Correcting Hierarchical Reasoning Compact Model) is a ~27M parameter reasoning model that extends AugmentedHRM by adding two targeted components to the inner reasoning loop: **Error-Conditioned Input Injection** and a **Stagnation-Aware ACT mechanism**.

The core problem SHREK addresses is that HRM and AugmentedHRM reason blindly — at every step the model processes only the original frozen input, with no explicit feedback about whether its current prediction is correct or whether it is making progress. This causes the well-documented failure modes identified by the mechanistic analysis paper: wrong fixed-point traps, the easy-task paradox, and aimless cycling.

SHREK introduces two additions. First, after each outer reasoning step the model decodes its current prediction, computes a differentiable error signal (conflict violations for Sudoku, path validity for Maze, output entropy for ARC-AGI), and injects this signal back into the reasoning loop for the next step — giving the model explicit knowledge of what is wrong with its current answer. Second, a normalised stagnation scalar measuring the change in the model's hidden state between steps is fed to the ACT halting mechanism, allowing the model to learn the difference between productive convergence and a fixed-point trap.

Both components are active during training, not bolted on at inference time. The base architecture — the H-level, L-level transformer blocks, embeddings, and Q-learning ACT — is inherited unchanged from AugmentedHRM. Parameter overhead is under 0.5M regardless of benchmark. SHREK is trained from scratch on all four benchmarks (ARC-AGI-1, ARC-AGI-2, Sudoku-Extreme, Maze-Hard) alongside HRM, TRM, and AugmentedHRM for a fair cross-benchmark comparison.

---

## Glossary

The following terms appear throughout this document. Read these definitions before continuing.

| Term | Plain-language definition |
|---|---|
| **Parameters / Weights** | The numbers inside a neural network that are learned during training. HRM has ~27 million. TRM has ~7 million. |
| **Forward pass** | One run of the model on an input — feeding data in, getting a prediction out. |
| **Embedding** | Converting a discrete token (e.g. the digit "5") into a vector of numbers the network can process. HRM uses vectors of length 512. |
| **Hidden size** | The length of internal computation vectors. HRM uses 512. Larger = more expressive but slower. |
| **Logits** | Raw unnormalised output numbers before converting to probabilities. The highest logit per cell becomes the model's predicted digit. |
| **Entropy** | A measure of uncertainty. High entropy = the model is spread across many answers (unsure). Low entropy = the model is confident in one answer. |
| **z_H** | The high-level hidden state. The model's current abstract understanding of the problem. Used to produce the output prediction at each step. Shape: `(batch, seq_len, hidden_size)`. |
| **z_L** | The low-level hidden state. Detailed working memory that resets every time z_H updates. |
| **Input injection** | At each reasoning step the original problem embedding is added to z_H before processing: `z_H + input_embeddings`. Prevents the model from forgetting the problem. |
| **Fixed point** | A state where the model's output stops changing — it has settled. Can be a *correct* fixed point (right answer) or a *wrong* fixed point (stuck on a wrong answer). |
| **Fixed-point trap** | When the model gets stuck at a wrong fixed point and cannot escape. The central failure mode identified in the mechanistic analysis paper. |
| **Hierarchical convergence** | HRM's two-level loop: the L-level runs several times per one H-level update. The L-level resets after each H update, keeping it computationally active. |
| **ACT (Adaptive Computation Time)** | A mechanism that lets the model decide how many reasoning steps to take. Easy problems halt early; hard problems run longer. Avoids wasting compute. |
| **Q-head** | A small network on top of z_H that predicts `q_halt` and `q_continue`. The model halts when `q_halt > q_continue`. Trained with Q-learning (a reinforcement learning method). |
| **1-step gradient approximation** | HRM's training trick: only the final reasoning step is differentiated (gradients computed). All prior steps run without gradient tracking to save memory. |
| **Data augmentation** | Creating extra training examples by transforming existing ones. For Sudoku: shuffling rows, columns, digit mappings. Improves generalisation. |
| **Model bootstrapping** | Mixing easier hint-augmented puzzles into training so the model learns the task structure before tackling the hardest inputs. Used in AugmentedHRM. |
| **Conflict loss** | For Sudoku: a measure of how many rule violations exist in the current predicted board (duplicate digits in rows/columns/boxes). Zero = valid solution. Already implemented in `visualization/landscape.py`. |
| **PCA (Principal Component Analysis)** | Compresses the high-dimensional z_H vector (512 dims) down to 2D for plotting. Used in the mechanistic analysis to visualise the model's reasoning trajectory. |
| **Stagnation** | When z_H stops changing between consecutive outer steps — the model has converged but may be at a wrong fixed point. Measured as `‖z_H_t − z_H_{t-1}‖ / ‖z_H_{t-1}‖`. |
| **Error-conditioned injection** | SHREK's first new component. An encoded error signal added to the input injection so the model sees what is wrong with its current prediction at each step. |
| **Stagnation-aware ACT** | SHREK's second new component. The stagnation scalar is fed to the Q-head so the model learns whether low change means correct convergence or a trap. |

---

## 1. Baseline: What AugmentedHRM Does

Before describing SHREK, it is important to understand exactly what the preceding model does, because SHREK is a targeted extension, not a redesign.

**HRM inner forward pass (unchanged in AugmentedHRM):**
```
for each H-step:
    for each L-step:
        z_L = L_level(z_L, z_H + input_embeddings)
    z_H = H_level(z_H, z_L)

output = lm_head(z_H)
q_halt, q_continue = q_head(z_H[:, 0])
```

The input injection at every L-step is `z_H + input_embeddings`. The `input_embeddings` are the **frozen original problem** — they never change across reasoning steps. The model can only update its belief state `z_H` by recurrently processing the same static input.

**AugmentedHRM's additions:**
1. Data augmentation during training (shuffle/permute the puzzle)
2. Random perturbation on reset: `z_H_init += random_noise` when ACT halts
3. Model bootstrapping: training on easier hint-augmented puzzles

The perturbation is applied **only on reset** (between ACT outer steps), it is **random** (not conditioned on the current state), and it happens **outside the reasoning loop** (not during active inference). The model never learns to use a stagnation signal — it just gets randomly nudged at the start of each new outer pass.

---

## 2. The Core Problem This Architecture Is Solving

The mechanistic analysis (paper [3]) found three failure modes:

1. **Wrong fixed-point traps**: the model converges to a stable but incorrect state
2. **Easy-task paradox**: the model fails on puzzles that should be trivial (e.g., Sudoku with one cell missing)
3. **Aimless cycling**: the model loops without making progress ("grokking" dynamics)

All three share a root cause: **the model has no feedback about the quality of its current prediction during reasoning**. It processes the same frozen input at every step regardless of whether its current answer has 0 or 50 errors. It cannot tell whether it is making progress.

---

## 3. SHREK Architecture

SHREK adds two components to the HRM inner loop. Everything else — the H_level, L_level, ACT Q-head, training procedure, weight initialisation — stays identical.

### 3.1 Component 1: Error-Conditioned Input Injection

After each outer ACT step, decode the current prediction from `z_H` and compute a differentiable **error signal**. Encode this signal into `hidden_size` space and add it to the input injection at each subsequent inner step.

**Modified injection:**
```
Current HRM:   z_L = L_level(z_L,  z_H + input_embeddings)
SHREK:         z_L = L_level(z_L,  z_H + input_embeddings + error_embedding_t)
```

The error embedding is computed once per outer ACT step, before the inner H/L loop begins. It is fixed throughout that outer step's inner iterations.

**Error signal definition per benchmark:**

| Benchmark | Error signal | Notes |
|---|---|---|
| Sudoku-Extreme | Per-cell conflict indicator (81-dim binary vector: 1 if cell participates in any row/col/box conflict) | `differentiable_conflict_loss` already exists in `landscape.py` |
| Maze-Hard (30×30) | Per-cell path validity (900-dim: 1 if cell on predicted path is a wall or the path is discontinuous) | Requires path decoding from prediction |
| ARC-AGI-1 and ARC-AGI-2 | Per-cell output entropy (900-dim: entropy of output logit distribution at each position) | No domain knowledge required; high entropy = uncertain = likely wrong |

The error encoder is a single linear projection:
```
error_encoder: R^{seq_len} → R^{hidden_size}
```

For Sudoku: `81 → hidden_size`
For Maze: `900 → hidden_size`
For ARC-AGI: `900 → hidden_size`

This is followed by RMS norm to keep the magnitude consistent with the existing input injection. A learned scalar gate (`alpha`, initialised near zero) controls how much the error signal contributes:

```
error_embedding_t = alpha * rms_norm(error_encoder(error_signal_t))
```

Initialising `alpha ≈ 0` means the model starts training as standard HRM and gradually learns to use the error signal. This prevents training instability from a noisy error signal at initialisation.

### 3.2 Component 2: Stagnation-Aware ACT

Compute a normalised stagnation scalar at each outer ACT step:
```
stagnation_t = ||z_H_t - z_H_{t-1}|| / (||z_H_{t-1}|| + eps)
```

A high value means the model is actively changing state. A low value means the model has converged (either to a correct answer or a fixed-point trap).

This scalar is fed as an additional feature to the ACT Q-head:
```
Current Q-head:   q_logits = q_head(z_H[:, 0])              # hidden_size → 2
SHREK Q-head:     q_logits = q_head(concat(z_H[:, 0], s_t)) # hidden_size+1 → 2
```

The Q-head learns during training when stagnation signals a correct convergence (halt) versus a trap (continue, which may trigger the error correction loop).

No perturbation is applied. Unlike AugmentedHRM, SHREK does not randomly reset the state. The model learns to handle stagnation through the error injection and the trained Q-head.

---

## 4. Full Forward Pass

```
Outer ACT loop:
    1. If halted: reset carry to initial state (standard HRM reset, no noise)
    2. Compute error_signal_t from current prediction (decoded from z_H)
    3. Encode: error_embedding_t = alpha * rms_norm(error_encoder(error_signal_t))
    4. Compute stagnation_t = ||z_H_t - z_H_{t-1}|| / (||z_H_{t-1}|| + eps)
    5. Inner H/L loop:
          for each H-step:
              for each L-step:
                  z_L = L_level(z_L, z_H + input_embeddings + error_embedding_t)
              z_H = H_level(z_H, z_L)
    6. output = lm_head(z_H)
    7. q_logits = q_head(concat(z_H[:, 0], stagnation_t))
    8. halt/continue decision via ACT
```

On the first outer step, `z_H_{t-1}` is undefined. Set `stagnation_0 = 1.0` (maximum change, model is "actively searching") and `error_embedding_0 = 0` (no prior prediction to evaluate).

---

## 5. Model Size

SHREK is intentionally close in size to HRM to keep the comparison fair under Green AI principles.

| Component | Parameters |
|---|---|
| Base HRM (H_level + L_level + embeddings) | ~27M |
| Error encoder (e.g., 81 → 512 for Sudoku) | ~42K |
| Error encoder (900 → 512 for ARC/Maze) | ~461K |
| Stagnation feature in Q-head (1 extra input) | ~2 |
| Learned gate scalar `alpha` | 1 |
| **Total SHREK (Sudoku)** | **~27.04M** |
| **Total SHREK (ARC/Maze)** | **~27.5M** |

The overhead is negligible. SHREK is effectively 27M parameters — the same as HRM and AugmentedHRM.

**Do not increase the base model size.** The purpose of this work is efficient reasoning. Adding more hidden dimensions or layers to chase accuracy would undermine the Green AI argument and make it harder to attribute improvements to the architectural changes rather than raw capacity.

---

## 6. Pros

**Principled, not a heuristic**: The error injection directly addresses the mechanistic finding that models reason without feedback. It is grounded in the PCA trajectory analysis.

**Universal across benchmarks**: The error signal definition adapts per task (structured for Sudoku/Maze, entropy-based for ARC-AGI), so the architecture applies to all four benchmarks without special-casing the model itself.

**Trained into the model**: Both components are active during training, not bolted on at inference. The model learns to use the error signal and stagnation feature. This makes the comparison with HRM/AugmentedHRM clean and fair.

**Minimal parameter overhead**: ~0.1-2% parameter increase depending on benchmark. FLOPs increase only by the small error encoder projection per outer step.

**Replaces random perturbation with learned response**: AugmentedHRM's random noise is crude. SHREK gives the model information and lets it learn what to do with it — a scientifically stronger approach.

**Clean mechanistic story**: The PCA trajectory plots should visually show SHREK trajectories correcting course when error is high, whereas AugmentedHRM trajectories cycle randomly before finding the answer.

---

## 7. Cons and Risks

### 7.1 Early Training Instability

At the start of training, predictions are essentially random. The error signal at step 0 is therefore a noisy vector that carries no useful information. If `alpha` is not initialised near zero, this noise is injected directly into the reasoning loop and can destabilise early training. The gated initialisation (`alpha ≈ 0`) is critical and must be carefully tuned.

### 7.2 Chicken-and-Egg Learning Problem

The model must simultaneously learn (a) how to make good predictions and (b) how to use the error feedback from those predictions. These two objectives are coupled. If the error signal improves only after the model improves, it may provide little benefit early in training when it is most needed. The model may converge to a local optimum where it ignores the error signal entirely.

### 7.3 Shortcut Risk for Sudoku

This is a significant concern. For Sudoku, the structured conflict loss is an extremely informative signal — arguably more informative than the reasoning itself. The model might learn to use the error feedback as a direct constraint propagation shortcut rather than developing genuine hierarchical reasoning. If this happens, SHREK's accuracy improvement would not reflect better reasoning but better exploitation of a task-specific signal. This would be an interesting finding in itself, but it would undermine the reasoning narrative.

**How to test for this**: Ablate the error signal at inference time on a trained SHREK. If accuracy drops to AugmentedHRM levels, the model genuinely uses it. If accuracy is unchanged, the model learned to ignore it and the base training is doing the work.

### 7.4 ARC-AGI Error Signal Is Weak

For ARC-AGI, the entropy-based error signal is much weaker than the structured signals for Sudoku and Maze. Output entropy captures "how uncertain is the model" but not "what kind of error is the model making". On ARC-AGI, a model can be confident and wrong — especially early in training. The improvement on ARC-AGI from error injection is likely to be modest or zero.

### 7.5 Path Validity for Maze Is Ambiguous

For Maze-Hard (30×30), defining a per-cell error signal requires decoding the predicted path and checking its validity. The prediction from `lm_head(z_H)` is a token distribution over `{#, space, S, G, o}` at each cell. A valid solution path must be continuous and reach the goal. Computing a differentiable validity measure from this is non-trivial and may introduce errors in the signal itself.

### 7.6 No Guarantee of Escaping Fixed Points

Even with error feedback, there is no formal guarantee that injecting the error signal will cause the model to escape a fixed point. If the model is stuck in a wrong stable state, the error embedding is appended to an already-converged `z_H`. The recurrent loop may be strong enough to return to the same state regardless. The stagnation-aware Q-head mitigates this by learning to detect convergence, but it cannot force the inner loop to explore different states.

### 7.7 Increased Inference Complexity

Each outer ACT step now requires:
1. A `lm_head` decode of `z_H` (already computed for output, can be reused)
2. A structured error computation (new)
3. A stagnation norm computation (new)
4. An error encoder forward pass (new)

Steps 2 and 3 add small but non-zero inference cost. For the Green AI comparison (CodeCarbon, FLOPs), SHREK will be marginally more expensive per step than HRM. This must be reported honestly.

---

## 8. Honest Assessment: Will This Improve Results?

### On Sudoku-Extreme
**Likely yes, but headroom is limited.** AugmentedHRM already achieves ~96.9% with bootstrapping. The structured conflict signal is clean and meaningful. SHREK should improve on easy-puzzle failures (the easy-task paradox) since the conflict signal directly pinpoints the problem cell. Expect 1-3 percentage points improvement, mostly from easy puzzles. This is a meaningful but not dramatic improvement.

### On Maze-Hard
**Uncertain.** The path validity signal is harder to define differentiably. If implemented well, it should help with the same fixed-point trap problem. If the signal is noisy or ambiguous, it may have no effect. AugmentedHRM's improvement on Maze is also unknown (Gap 3), so even matching AugmentedHRM on Maze with SHREK on top is interesting.

### On ARC-AGI-1 and ARC-AGI-2
**Unlikely to show large improvement from error injection alone.** ARC-AGI requires learning transformation rules from very few examples. The entropy signal is weak. The primary improvement on ARC-AGI will likely come from whether AugmentedHRM's augmentation techniques generalise to this benchmark — not from SHREK's error injection. SHREK's stagnation-aware ACT may help marginally.

### Biggest Risk
The most likely outcome is: SHREK improves on Sudoku, shows mixed results on Maze and ARC-AGI, and the story is that the improvement is benchmark-dependent on how clean the error signal is. **This is actually a publishable and honest result** — it shows that structured feedback helps when the error signal is well-defined, and calls for future work on better error representations for abstract tasks like ARC-AGI.

### What Is More Likely to Be the Paper's Strongest Contribution
Gap 3 — the first cross-benchmark evaluation of AugmentedHRM — is likely to produce more interesting findings than SHREK beating AugmentedHRM everywhere. If AugmentedHRM's techniques fail to generalise to ARC-AGI or Maze (plausible, since they were designed for Sudoku), and SHREK partially recovers this, that is a strong result. If AugmentedHRM generalises well, then SHREK is an incremental improvement on an already strong baseline.

**The paper's value does not depend entirely on SHREK outperforming everything.** An honest analysis of what works where, with mechanistic PCA trajectories to explain why, is a strong contribution.

---

## 9. Implementation Checklist

The following changes are needed relative to AugmentedHRM:

- [ ] Add `error_encoder` linear layer to `HierarchicalReasoningModel_ACTV1_Inner.__init__`
- [ ] Add learned gate scalar `alpha` (initialised to 0.01)
- [ ] Add per-benchmark error signal computation function (conflict loss, path validity, entropy)
- [ ] Modify `HierarchicalReasoningModel_ACTV1_Inner.forward` to compute error embedding and inject it
- [ ] Store previous `z_H` across outer steps for stagnation computation
- [ ] Modify Q-head input dimension from `hidden_size` to `hidden_size + 1`
- [ ] Add stagnation scalar to Q-head input in ACT wrapper forward
- [ ] Update `require_trace` path to also return error signal values per step (for mechanistic analysis)
- [ ] Implement ablation flag: `use_error_injection=True/False` for ablation study
- [ ] Add CodeCarbon tracker around evaluation loops

---

## 10. Relation to the Mechanistic Analysis

The mechanistic analysis paper visualises `z_H` trajectories using PCA. For SHREK, the expected trajectory behaviour is:

- When error is high and stagnation is detected: the error injection should cause `z_H` to shift away from the wrong fixed point (visible as a direction change in PCA)
- When error decreases: the trajectory should converge more directly toward the correct fixed point
- On easy Sudoku: the conflict signal is a precise pointer to the one wrong cell, so the model should correct in 1-2 outer steps rather than cycling

Comparing SHREK PCA trajectories against HRM and AugmentedHRM trajectories on the same puzzle samples is the mechanistic analysis section of your paper. This requires storing `error_signal_t` values alongside `z_H_trace` in the `require_trace` return path.
