# SHREK Model Architecture
## Self-Correcting Hierarchical Reasoning Compact Model

---

## Abstract

SHREK (Self-Correcting Hierarchical Reasoning Compact Model) is a ~27 million parameter neural network for structured reasoning tasks. It is built directly on top of AugmentedHRM — the model from the mechanistic analysis paper — and extends it with two new trained components that give the model explicit feedback about whether its current answer is correct and whether it is making progress.

**What we inherit from AugmentedHRM and why.** AugmentedHRM introduced three improvements over the original HRM: (1) *data augmentation* — creating extra training examples by randomly shuffling puzzle rows, columns, and digit labels so the model sees more variety and does not memorise specific patterns; (2) *model bootstrapping* — mixing easier hint-filled puzzles into training so the model first learns the general structure of the task before tackling the hardest examples; and (3) *random perturbation* — adding random noise to the model's starting state each time it restarts reasoning, giving it a different starting point to hopefully avoid getting stuck. SHREK keeps the first two because they are proven to work and do not conflict with anything we add. We remove and replace the third.

**What SHREK replaces and adds.** AugmentedHRM's random perturbation is crude — the model gets randomly nudged but never learns when or how to correct itself. The root problem is that HRM and AugmentedHRM reason in the dark: at every step the model processes only the original frozen problem input, with no feedback about whether its current predicted answer is right or wrong. This causes the three failure modes identified by the mechanistic analysis paper: *wrong fixed-point traps* (the model converges to a stable but incorrect answer and cannot move away), the *easy-task paradox* (the model fails on trivially easy inputs like a Sudoku with only one blank cell), and *aimless cycling* (the model loops through the same states without making progress).

SHREK fixes this by replacing random perturbation with two learned components. The first is **Error-Conditioned Input Injection**: after each reasoning step, the model decodes its current best answer, computes a differentiable *error signal* measuring how wrong that answer is (for example, how many rule violations exist in a Sudoku board), and feeds this signal back into the reasoning loop as an additional input. The model now knows what is wrong and can learn to correct it. The second is **Stagnation-Aware ACT** (*Adaptive Computation Time* — the mechanism that decides when the model should stop reasoning): a normalised measure of how much the model's internal state changed between steps is added as an input to the halting decision. The model learns the difference between productive convergence (correctly settling on an answer) and a fixed-point trap (incorrectly getting stuck).

Both new components are active during training, not added after the fact at test time. The model is trained with these signals from the start and learns to use them. The base architecture — the hierarchical transformer blocks, the embedding layers, the Q-learning halting mechanism — is inherited completely unchanged from AugmentedHRM. The parameter overhead is under 0.5 million regardless of benchmark, making SHREK effectively the same size as HRM and AugmentedHRM (~27M parameters).

SHREK is trained from scratch and evaluated on all four benchmarks: ARC-AGI-1, ARC-AGI-2, Sudoku-Extreme, and Maze-Hard. The same is done for HRM, TRM, and AugmentedHRM to produce a fair cross-benchmark comparison. The three-step progression is:

```
HRM           →  gets stuck, no escape mechanism
AugmentedHRM  →  adds training tricks and random escape
SHREK         →  keeps training tricks, replaces random escape with learned self-correction
```

---

## Glossary

Every technical term used in this document is defined below. Read this before continuing.

| Term | Plain-language definition |
|---|---|
| **Parameters / Weights** | The numbers inside a neural network that are learned during training. More parameters = more expressive model. HRM has ~27 million. TRM has ~7 million. |
| **Training** | The process of adjusting a model's parameters by showing it many examples and measuring how wrong its predictions are, then nudging the parameters to reduce those errors. |
| **Inference** | Using a trained model to make predictions on new inputs. No parameter updates happen during inference. |
| **Forward pass** | One complete run of the model on a single input — data goes in, a prediction comes out. |
| **Embedding** | A way of converting discrete symbols (e.g. the digit "5") into a vector of numbers that the network can compute with. HRM uses vectors of length 512. |
| **Hidden size** | The length of the internal computation vectors the model uses at each layer. HRM uses 512. Larger = more expressive but slower and more expensive. |
| **Logits** | The raw output numbers from the model before they are converted into probabilities. For Sudoku, the model outputs 11 logits per cell (one per digit 0–9 plus a padding token) and picks the highest one as its prediction. |
| **Entropy** | A measure of uncertainty. High entropy means the model's probability is spread across many possible answers (unsure). Low entropy means it is confident in one answer. Formula: `H = -Σ p log p`. |
| **z_H (High-level state)** | The model's main internal memory — its current high-level understanding of the problem. Shaped as `(batch_size, sequence_length, hidden_size)`. The output prediction is decoded from z_H at each step. Think of it as the model's current best hypothesis. |
| **z_L (Low-level state)** | A secondary working-memory state used for detailed computation inside each reasoning cycle. It resets every time z_H is updated (see hierarchical convergence). |
| **Input injection** | At every inner reasoning step, the original problem embedding is added to the current state: `z_H + input_embeddings`. This prevents the model from forgetting the problem while it reasons. |
| **Frozen input** | The original problem representation never changes across reasoning steps — it is computed once and reused. The model updates z_H but always sees the same original input added to it. |
| **Fixed point** | A state where the model's output stops changing between reasoning steps — it has "settled". A *correct* fixed point = right answer found. A *wrong* fixed point = stuck on an incorrect answer. |
| **Fixed-point trap** | When the model converges to a wrong answer and cannot escape it. The main failure mode found by the mechanistic analysis paper. |
| **Hierarchical convergence** | HRM's two-level loop structure. The L-level (low-level) runs several times for each one update of the H-level (high-level). The L-level resets after each H-level update, keeping it computationally active. This mimics fast low-level thinking guided by slower high-level planning. |
| **ACT (Adaptive Computation Time)** | A mechanism that lets the model decide how many reasoning steps to take rather than always running a fixed number. Easier problems halt early (saving compute); harder ones run longer. |
| **Q-head** | A small neural network on top of z_H that outputs two values — `q_halt` (value of stopping now) and `q_continue` (value of doing another step). The model halts when `q_halt > q_continue`. Trained using Q-learning (a reinforcement learning technique where the model learns the long-term value of each action). |
| **Q-learning** | A reinforcement learning method where a model learns to estimate the value of taking a given action in a given state. Here, the actions are "halt" and "continue" and the model learns their values from training outcomes. |
| **1-step gradient approximation** | HRM's memory-saving training trick: only the final reasoning step has gradients computed (backpropagation). All earlier steps run without gradient tracking to avoid storing the entire reasoning history in memory. |
| **Data augmentation** | Generating extra training examples by applying valid transformations to existing ones. For Sudoku: randomly shuffling rows, columns, and digit labels. Helps the model generalise rather than memorise. |
| **Model bootstrapping** | A curriculum learning technique: start training by mixing in easier examples (Sudoku with more cells pre-filled) so the model learns the basic task structure before seeing the hardest inputs. Used in AugmentedHRM and kept in SHREK. |
| **Random perturbation** | AugmentedHRM's escape mechanism: when the model restarts a reasoning sequence, random noise is added to the initial state so it starts from a slightly different position and might avoid the same wrong fixed point. SHREK replaces this with a learned mechanism. |
| **Conflict loss** | For Sudoku: a differentiable measure of rule violations in the current predicted board — how many rows, columns, or 3×3 boxes contain duplicate digits. Zero = valid solution. Already implemented in `visualization/landscape.py`. |
| **Error signal** | A per-cell measure of how wrong the model's current prediction is. Structured (conflict violations for Sudoku, path validity for Maze) or unstructured (output entropy for ARC-AGI). SHREK's first new component feeds this back to the model. |
| **Stagnation** | When z_H stops changing significantly between consecutive outer reasoning steps — the model has converged. May be a correct convergence or a fixed-point trap. Measured as `‖z_H_t − z_H_{t-1}‖ / (‖z_H_{t-1}‖ + ε)`. |
| **Error-conditioned injection** | SHREK's first new component. The error signal is encoded and added to the input injection so the model explicitly sees what is wrong with its current answer at each step. |
| **Stagnation-aware ACT** | SHREK's second new component. The stagnation scalar is added to the Q-head's input so the model learns to distinguish productive convergence from a fixed-point trap. |
| **PCA (Principal Component Analysis)** | A technique for compressing high-dimensional data into 2D for visualisation. Used in the mechanistic analysis to plot z_H trajectories — each dot in the plot is a reasoning step, and the path shows how the model's internal state evolves. |
| **Differentiable** | A mathematical property meaning gradients can be computed through a function during training. Error signals must be differentiable (or at least not block gradients) to be usable in the training process. |
| **Ablation** | A controlled experiment where one component is removed to measure its individual contribution. For example: train SHREK, then re-evaluate with error injection disabled to see how much accuracy it contributes. |
| **FLOPs (Floating Point Operations)** | A hardware-agnostic measure of computational cost — how many arithmetic operations the model performs. Used in Green AI comparisons alongside energy consumption. |

---

## 1. Baseline: What AugmentedHRM Does

SHREK is a direct extension of AugmentedHRM, not a redesign. To understand what SHREK adds, you first need to understand exactly what AugmentedHRM does and what it does not do.

**The HRM inner forward pass — unchanged in both AugmentedHRM and SHREK:**
```
for each H-step:
    for each L-step:
        z_L = L_level(z_L, z_H + input_embeddings)
    z_H = H_level(z_H, z_L)

output = lm_head(z_H)
q_halt, q_continue = q_head(z_H[:, 0])
```

At every L-step, the only things the model sees are its own current state (`z_H`) and the original frozen problem (`input_embeddings`). The `input_embeddings` never change — they are the same at step 1 and step 50. The model has no way of knowing whether its current answer is close to correct or completely wrong.

**AugmentedHRM's three additions over HRM:**

| Technique | What it does | Kept in SHREK? |
|---|---|---|
| Data augmentation | Shuffles puzzle rows, columns, digit labels during training to increase variety | ✅ Yes — proven effective, no conflicts |
| Model bootstrapping | Mixes in easier hint-filled puzzles during training for curriculum learning | ✅ Yes — proven effective, no conflicts |
| Random perturbation | Adds random noise to the initial z_H state at the start of each new reasoning pass | ❌ No — replaced by stagnation-aware ACT |

The random perturbation operates at reset time — between full inference passes — not during active reasoning. It gives the model a random starting point hoping it finds a different fixed point by chance. It is not conditioned on anything meaningful: the model does not learn when to perturb or in what direction.

SHREK keeps what works (data augmentation and bootstrapping) and replaces what is unprincipled (random perturbation) with two learned components that address the underlying cause.

---

## 2. The Core Problem This Architecture Is Solving

The mechanistic analysis paper (paper [3]) identified three failure modes across all models up to and including AugmentedHRM:

1. **Wrong fixed-point traps**: the model converges to a stable but incorrect state. Once there, no amount of additional reasoning steps can move it — the same input keeps producing the same wrong answer.
2. **Easy-task paradox**: the model fails on puzzles that should be trivially easy. A Sudoku with only one blank cell (where the answer is mathematically forced) trips the model more often than hard puzzles with many blanks.
3. **Aimless cycling**: the model loops through the same set of states without converging — neither finding the answer nor halting cleanly. The paper calls this "grokking dynamics".

All three share the same root cause: **the model processes the same frozen input at every reasoning step regardless of how wrong its current answer is**. It has no feedback signal. It cannot tell whether it is at 5 conflicts or 0 conflicts, whether it is improving or cycling. It reasons entirely in the dark.

AugmentedHRM's random perturbation partially escapes traps by chance. SHREK directly informs the model about its error and progress so it can correct itself deliberately.

---

## 3. SHREK Architecture

SHREK adds two components to the HRM inner loop. The H_level, L_level transformer blocks, embeddings, Q-learning ACT, data augmentation pipeline, and bootstrapping training procedure are all inherited unchanged.

### 3.1 What SHREK Inherits (Unchanged from AugmentedHRM)

- Full HRM hierarchical architecture (H_level + L_level transformer blocks)
- All embedding layers (`embed_tokens`, `embed_pos`/`rotary_emb`, `puzzle_emb`)
- LM head and Q-head structure
- 1-step gradient approximation for training
- Data augmentation during training
- Model bootstrapping (hint-augmented easier examples) during training
- Standard clean reset on restart (`reset_carry(use_default=True)`) — no random noise

### 3.2 What SHREK Removes

- Random perturbation on reset (`reset_carry(use_default=False)`) — replaced entirely by stagnation-aware ACT

### 3.3 Component 1: Error-Conditioned Input Injection

After each outer ACT step, the model's current prediction is decoded from `z_H` using the existing `lm_head`. An error signal is computed from that prediction and encoded into a vector of the same size as the hidden state. This vector is added to the input injection for every inner step of the next reasoning cycle.

**Modified input injection:**
```
AugmentedHRM:   z_L = L_level(z_L,  z_H + input_embeddings)
SHREK:          z_L = L_level(z_L,  z_H + input_embeddings + error_embedding_t)
```

The error embedding is computed once per outer ACT step (before the inner H/L loop begins) and stays fixed throughout that cycle's inner iterations.

**Error signal per benchmark:**

| Benchmark | Error signal | Dimension | Notes |
|---|---|---|---|
| Sudoku-Extreme | Per-cell conflict indicator: 1 if a cell participates in any row, column, or box rule violation | 81-dim | `differentiable_conflict_loss` already in `visualization/landscape.py` |
| Maze-Hard (30×30) | Per-cell path validity: 1 if a cell on the predicted path is a wall or the path is discontinuous | 900-dim | Requires path decoding from prediction |
| ARC-AGI-1 and ARC-AGI-2 | Per-cell output entropy: uncertainty at each output position | 900-dim | No domain knowledge required — works from existing logits |

The error encoder is a single linear projection followed by RMS normalisation:
```
error_encoder:  R^{seq_len}  →  R^{hidden_size}
error_embedding_t = alpha * rms_norm(error_encoder(error_signal_t))
```

`alpha` is a single learned scalar initialised near zero (e.g. 0.01). This means SHREK starts training behaving exactly like standard HRM — the error signal has no effect initially. As training progresses, `alpha` grows and the model learns to use the signal. This avoids instability from noisy error signals early in training.

### 3.4 Component 2: Stagnation-Aware ACT

At each outer ACT step, compute how much the model's main state changed since the previous step:

```
stagnation_t = ||z_H_t - z_H_{t-1}|| / (||z_H_{t-1}|| + eps)
```

- High value → model is actively changing → likely still making progress
- Low value → model has converged → could be correct (good) or a trap (bad)

This scalar is appended to the Q-head's input:
```
AugmentedHRM Q-head:   q_logits = q_head(z_H[:, 0])                    # hidden_size → 2
SHREK Q-head:          q_logits = q_head(concat(z_H[:, 0], stagnation_t))  # hidden_size+1 → 2
```

The Q-head now receives both the model's internal state and explicit information about whether that state is changing. Over training, it learns that low stagnation combined with low error = correct convergence (halt), while low stagnation combined with high error = fixed-point trap (continue, which allows the error injection in the next step to attempt correction).

---

## 4. Full Forward Pass

```
Training data pipeline (inherited from AugmentedHRM):
  - Data augmentation: shuffle puzzle structure
  - Bootstrapping: mix in easier hint-augmented examples

Outer ACT loop (one iteration = one outer step):
  1. If restarting: reset carry to clean initial state
       z_H = z_H_init  (no random noise — perturbation removed)
       z_L = z_L_init

  2. Compute error signal from current prediction:
       current_pred = argmax(lm_head(z_H), dim=-1)
       error_signal_t = compute_error(current_pred, benchmark_type)
       error_embedding_t = alpha * rms_norm(error_encoder(error_signal_t))

  3. Compute stagnation scalar:
       stagnation_t = ||z_H - z_H_prev|| / (||z_H_prev|| + eps)
       z_H_prev = z_H  (store for next step)

  4. Inner H/L loop:
       for each H-step:
           for each L-step:
               z_L = L_level(z_L, z_H + input_embeddings + error_embedding_t)
           z_H = H_level(z_H, z_L)

  5. Decode output:
       output = lm_head(z_H)

  6. ACT halting decision:
       q_logits = q_head(concat(z_H[:, 0], stagnation_t))
       halt if q_halt > q_continue, or if max steps reached
```

**First step initialisation:** On the first outer step, there is no previous z_H to compare. Set `stagnation_0 = 1.0` (treat as maximum change — model is freshly searching) and `error_embedding_0 = 0` (no prior prediction exists yet).

---

## 5. Model Size

SHREK's parameter count is kept as close to HRM and AugmentedHRM as possible. Increasing size to chase accuracy would undermine the Green AI argument and make it impossible to attribute improvements to architecture rather than raw capacity.

| Component | Parameters |
|---|---|
| Base HRM (H_level + L_level + all embeddings + heads) | ~27M |
| Error encoder — Sudoku (81 → 512) | ~42K |
| Error encoder — Maze / ARC-AGI (900 → 512) | ~461K |
| Stagnation feature added to Q-head input (1 scalar) | 2 |
| Learned gate scalar `alpha` | 1 |
| **Total SHREK on Sudoku** | **~27.04M** |
| **Total SHREK on Maze / ARC-AGI** | **~27.5M** |

The overhead is negligible — under 2% in the worst case. SHREK is the same model class and size as HRM and AugmentedHRM for all practical purposes.

---

## 6. Pros

**Builds on the strongest available baseline.** By keeping data augmentation and bootstrapping from AugmentedHRM, SHREK starts from proven improvements rather than reimplementing them. Any further gain over AugmentedHRM is directly attributable to the two new components.

**Cleanest possible ablation story.** The comparison between AugmentedHRM and SHREK isolates exactly one change: random perturbation replaced by error injection + stagnation ACT. Same data pipeline, same base architecture, same training procedure — only the escape mechanism differs.

**Principled, not a heuristic.** The error injection directly addresses the mechanistic finding that models reason without feedback. It is grounded in the PCA trajectory analysis from paper [3].

**Universal across all four benchmarks.** The error signal adapts per task (structured conflict loss for Sudoku, path validity for Maze, entropy for ARC-AGI) without changing the model architecture. The same two components run on all benchmarks.

**Trained into the model.** Both components are active during training. The model learns to use the error signal and stagnation feature — they are not bolted on at test time. This makes comparisons with other retrained models fair and honest.

**Minimal parameter overhead.** Under 0.5M additional parameters. FLOPs increase only by the error encoder forward pass (one matrix multiplication per outer step).

**Replaces random with learned.** AugmentedHRM's perturbation is random and unguided. SHREK gives the model explicit information and trains it to respond. This is scientifically stronger and more likely to generalise across benchmarks.

---

## 7. Cons and Risks

### 7.1 Early Training Instability

At the start of training, the model's predictions are essentially random noise. The error signal computed from a random prediction is therefore also noise. If `alpha` is not initialised near zero, this noisy error embedding is injected directly into the reasoning loop from step one and can destabilise the entire training run. The near-zero initialisation of `alpha` is not optional — it is critical.

### 7.2 Chicken-and-Egg Learning Problem

The model must simultaneously learn (a) how to make reasonable predictions and (b) how to use the error feedback from those predictions to improve them. These two goals are coupled. Early in training, predictions are bad, so the error signal is uninformative, so the model cannot benefit from it, so there is no pressure to improve the error signal's usefulness. The model may converge to a state where it ignores the error signal entirely. The near-zero `alpha` initialisation gives some protection by making the model first learn the base task before relying on error feedback, but this cannot be fully guaranteed.

### 7.3 Shortcut Risk for Sudoku

This is the most significant risk. For Sudoku, the conflict loss is an extremely informative, clean signal — arguably more informative than the recurrent reasoning itself. The model might learn to treat the error feedback as a direct constraint-solving shortcut: look at the conflict vector, fill in the conflicting cells, repeat. This would improve accuracy but would not reflect better hierarchical reasoning — it would reflect better exploitation of the error signal. If this happens, the paper's reasoning narrative is undermined.

**How to detect this:** Run a trained SHREK model in evaluation mode with `error_injection` disabled (`alpha = 0`). If accuracy drops to AugmentedHRM levels, the model is genuinely using the error signal as intended. If accuracy is unchanged, the model learned to ignore it and the base training is doing all the work.

### 7.4 ARC-AGI Error Signal Is Weak

For ARC-AGI, output entropy is used as the error signal because no structured task-specific signal is available. Entropy captures how uncertain the model is, but not what kind of error it is making. A model can be confidently wrong on ARC-AGI — high confidence in the wrong transformation. In this case the entropy signal is low (confident) even though the answer is incorrect, and SHREK's error injection provides no useful information. Improvements on ARC-AGI from error injection alone are likely to be modest or zero.

### 7.5 Path Validity for Maze Is Non-Trivial

The Maze-Hard error signal requires decoding the predicted token sequence into a path and checking whether that path is valid (continuous, no wall collisions, reaches the goal). The token vocabulary is `{#, space, S, G, o}` — decoding a valid path from this and computing a differentiable validity measure introduces implementation complexity and potential signal noise that does not exist in the Sudoku case.

### 7.6 No Guarantee of Escaping Fixed Points

Even with error feedback, there is no mathematical guarantee that the error injection will move the model out of a wrong fixed point. If `z_H` has converged to a stable wrong state, the recurrent loop may be strong enough to return to that same state regardless of what is added to the input. The stagnation-aware Q-head mitigates this by learning to detect stagnation and potentially continuing to reason, but it cannot force the inner loop to explore new states — it can only keep the outer loop running longer.

### 7.7 Slightly Higher Inference Cost

Each outer ACT step now additionally requires:
1. Decoding the current prediction via `lm_head(z_H)` — already computed for output, reusable at no extra cost
2. Computing the structured error signal from the prediction (new cost)
3. Running the error encoder forward pass (one matrix multiplication, new cost)
4. Computing the stagnation norm (two norm operations, negligible cost)

For the Green AI FLOPs and CodeCarbon energy comparisons, SHREK will be marginally more expensive per outer step than HRM and AugmentedHRM. This must be reported transparently.

---

## 8. Honest Assessment: Will This Improve Results?

### On Sudoku-Extreme
**Likely yes, but limited headroom.** AugmentedHRM with bootstrapping already achieves ~96.9%. The structured conflict signal is clean and precisely informative. SHREK should particularly improve easy-puzzle accuracy (the easy-task paradox) because the conflict signal directly identifies the conflicting cell. Expect 1–3 percentage points improvement concentrated on easy puzzles. Meaningful but not dramatic.

### On Maze-Hard
**Uncertain.** The path validity signal is harder to define cleanly. If the implementation is robust, it should help with fixed-point traps in the same way as Sudoku. If the signal is noisy or ambiguous, it may have no effect. Note that AugmentedHRM's performance on Maze is itself unknown — this is part of the paper's Gap 3 contribution. Even a clean baseline AugmentedHRM result on Maze is new information.

### On ARC-AGI-1 and ARC-AGI-2
**Unlikely to show large improvement from error injection.** ARC-AGI fundamentally requires inducing transformation rules from a handful of examples. The entropy signal is too weak to help with this. The main story on ARC-AGI will be whether AugmentedHRM's augmentation and bootstrapping generalise to this benchmark — that is itself a new result. SHREK's stagnation-aware ACT may help marginally.

### Biggest Risk
SHREK may improve clearly on Sudoku, show mixed results on Maze, and show little improvement on ARC-AGI. This is a publishable and intellectually honest result: structured error feedback helps when the error signal is clean and task-specific, and calls for future work on stronger error representations for abstract reasoning tasks like ARC-AGI.

### Strongest Contribution in the Paper
The first cross-benchmark evaluation of AugmentedHRM (Gap 3) is likely to generate the most novel findings. If AugmentedHRM's Sudoku-tuned techniques fail to generalise to ARC-AGI or Maze, and SHREK partially recovers this, that is a strong and clear result. The paper's value does not depend on SHREK beating every model on every benchmark — an honest mechanistic analysis of what works where, supported by PCA trajectory comparisons, is a strong contribution.

---

## 9. Implementation Checklist

Changes needed relative to AugmentedHRM code:

**Inherited — no changes needed:**
- [ ] Data augmentation pipeline (already in `dataset/build_sudoku_dataset.py`, `build_maze_dataset.py`, `build_arc_dataset.py`)
- [ ] Model bootstrapping / hint augmentation (already in training config)

**Removed:**
- [ ] Delete `use_default=False` path in `reset_carry` — replace all perturbation calls with standard clean reset

**Added:**
- [ ] Add `error_encoder` linear layer (`seq_len → hidden_size`) to `HierarchicalReasoningModel_ACTV1_Inner.__init__`
- [ ] Add learned scalar gate `alpha` initialised to `0.01`
- [ ] Implement per-benchmark error signal function: conflict loss (Sudoku), path validity (Maze), entropy (ARC-AGI)
- [ ] Modify `HierarchicalReasoningModel_ACTV1_Inner.forward` to compute and inject error embedding before inner loop
- [ ] Store `z_H_prev` across outer steps for stagnation computation
- [ ] Extend Q-head input dimension from `hidden_size` to `hidden_size + 1`
- [ ] Pass stagnation scalar to Q-head in ACT wrapper forward
- [ ] Extend `require_trace` return to include error signal values per outer step
- [ ] Add ablation flag `use_error_injection: bool` (default True) to config for ablation studies
- [ ] Add CodeCarbon energy tracker around evaluation loops for Green AI measurements

---

## 10. Relation to the Mechanistic Analysis

The mechanistic analysis paper visualises `z_H` reasoning trajectories using PCA. For SHREK, the expected changes compared to HRM and AugmentedHRM are:

- **When error is high and stagnation is low**: error injection should push `z_H` away from the wrong fixed point, visible as a directional shift in the PCA trajectory rather than a cycle.
- **When error decreases step-by-step**: the trajectory should converge more directly and smoothly toward the correct fixed point — less cycling, more purposeful movement.
- **On easy Sudoku puzzles**: the conflict signal is a precise pointer to the one wrong cell. The model should correct in 1–2 outer steps rather than the many cycles seen in AugmentedHRM.
- **AugmentedHRM trajectories** by comparison will show more random-walk behaviour before convergence — reflective of the random perturbation giving it different starting points rather than guided correction.

To produce these plots, `error_signal_t` values must be stored alongside `z_H_trace` in the `require_trace` return path, then both are passed to the existing `pca_trajectory.py` and `landscape.py` visualisation tools.
