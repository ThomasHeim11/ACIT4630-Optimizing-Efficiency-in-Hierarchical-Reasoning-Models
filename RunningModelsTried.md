# Training Runs Attempted — HRM Sudoku Reproduction

All runs on single GH200 GPU (102GB VRAM), Simula HPC cluster.
Paper targets: Original HRM ~55% test exact accuracy, Augmented HRM ~60% single ckpt / ~97% ensemble.

---

## Run 1: Original HRM — `amethyst-asp`

| Parameter | Value |
|---|---|
| Codebase | HRM-main (Original) |
| Dataset | sudoku-extreme-1k-aug-1000 (vanilla) |
| lr | 1e-4 |
| global_batch_size | 768 |
| epochs | 40000 |
| DISABLE_COMPILE | 1 |

**Result: DIVERGED**
- Accuracy crashed from 0.4 → 0.1 (random)
- Steps stuck at 16 (never learned to halt)
- lm_loss rising
- Killed manually

**Why it failed:** 8-GPU hyperparameters (lr=1e-4, batch=768) on 1 GPU without distributed gradient averaging. Original HRM codebase lacks divergence scheduling, making Q-learning unstable.

---

## Run 2: Augmented HRM — `giga-moose`

| Parameter | Value |
|---|---|
| Codebase | hrm-mechanistic-analysis-main (Augmented) |
| Dataset | sudoku-extreme-1k-aug-1000-hint |
| lr | 1e-4 |
| global_batch_size | 768 |
| epochs | 40000 |
| DISABLE_COMPILE | 1 |

**Result: KILLED EARLY (only reached step ~11718)**
- Accuracy ~0.55 (per-cell), exact_accuracy spiking
- Steps ~8 (healthy)
- q_continue_loss rising (0.04 → 0.064)
- Killed to restart with "safer" 1-GPU recipe

**Why it was killed:** Thought rising q_continue_loss was dangerous. In hindsight, this was the most promising run — exact_accuracy was spiking. Should have let it continue.

---

## Run 3: Original HRM — `honorable-capuchin`

| Parameter | Value |
|---|---|
| Codebase | HRM-main (Original) |
| Dataset | sudoku-extreme-1k-aug-1000 (vanilla) |
| lr | 7e-5 |
| global_batch_size | 384 |
| epochs | 20000 |
| DISABLE_COMPILE | 1 |

**Result: DIVERGED (again)**
- Accuracy spiked to ~0.4 at step ~5k, then crashed to ~0.1
- Steps stuck at 15 entire run
- q_continue_loss reached 0.19 (past kill threshold)
- lm_loss plateaued at ~2.2 (near random)
- exact_accuracy: flat zero
- Ran to completion (20k epochs)

**Why it failed:** Even with reduced lr (7e-5) matching the authors' 1-GPU recipe, the Original HRM codebase is too unstable on 1 GPU. Lacks the divergence scheduling mask that the Augmented HRM has.

---

## Run 4: Augmented HRM — `automatic-harrier`

| Parameter | Value |
|---|---|
| Codebase | hrm-mechanistic-analysis-main (Augmented) |
| Dataset | sudoku-extreme-1k-aug-1000-hint |
| lr | 7e-5 |
| global_batch_size | 384 |
| epochs | 40000 |
| DISABLE_COMPILE | 1 |

**Result: TRAINED BUT 0% TEST EXACT ACCURACY**
- train/accuracy ~0.55 (per-cell only)
- train/exact_accuracy ~0.01 (almost never solved a full puzzle)
- Steps ~8 (healthy)
- q_continue_loss diverged to 0.35 (way past 0.15 threshold)
- lm_loss plateaued at ~2.0
- Ran to ~93k steps

**Evaluation (10 ckpts, 9 permutations, 1000 test examples):** 0/1000 = 0.0% exact accuracy

**Why it failed:** Lower lr (7e-5) and smaller batch (384) led to slower learning. The model learned per-cell accuracy (~55%) but never achieved per-puzzle accuracy. q_continue_loss diverged far worse than giga-moose (0.35 vs 0.064), suggesting the reduced batch size hurt Q-learning stability.

---

## Key Lessons Learned

1. **Original HRM codebase cannot train on 1 GPU** — tried twice (lr=1e-4 and lr=7e-5), both diverged. Lacks divergence scheduling mask.
2. **Augmented HRM codebase is more stable** — divergence scheduling prevents total collapse, but Q-learning still drifts.
3. **lr=1e-4 + batch=768 > lr=7e-5 + batch=384** — giga-moose (1e-4/768) had spiking exact_accuracy; automatic-harrier (7e-5/384) had 0%. Larger batch smooths Q-learning gradients.
4. **Don't kill promising runs early** — giga-moose was the best run but was killed at step 11718 out of concern about q_continue_loss.
5. **Per-cell accuracy != puzzle-solving** — 55% per-cell accuracy means 0% exact puzzle accuracy. The model needs much higher per-cell accuracy to solve full puzzles.
6. **torch.compile matters for speed** — DISABLE_COMPILE=1 was used in all runs. Removing it should speed up training.

---

## What Has NOT Been Tried Yet

1. **Vanilla dataset through Augmented HRM codebase** — uses divergence scheduling for stable training of the Original HRM baseline
2. **lr=1e-4 + batch=768 with Augmented HRM running to completion** — giga-moose was killed early; needs full run
3. **torch.compile enabled** — all runs used DISABLE_COMPILE=1
4. **Multi-node training (2 GPUs across gh001 + gh002)** — cluster has 1 GPU per node, would need multi-node torchrun
5. **Lower lr (5e-5)** — fallback option from plan, never attempted
6. **Evaluating giga-moose checkpoints** — only has steps up to 11718, probably too early

---

## Next Planned Runs

Both using Augmented HRM codebase, lr=1e-4, batch=768, torch.compile enabled:

1. **Original HRM baseline**: vanilla dataset, 20k epochs
2. **Augmented HRM**: hint dataset, 40k epochs
