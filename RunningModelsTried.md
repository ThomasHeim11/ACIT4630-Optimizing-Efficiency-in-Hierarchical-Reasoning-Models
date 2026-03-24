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

1. **CRITICAL: The adam_atan2 optimizer was broken in ALL previous runs.** Our pure-Python patch applied weight decay to the gradient (`grad.add(p, alpha=wd)`) instead of directly to parameters (`p.mul_(1 - lr * wd)`). This warped the atan2 update computation. Fixed on 2026-03-24 by using the real `adam-atan2-pytorch` v0.2.8 via import shim.
2. **Original HRM codebase could not train on 1 GPU** — tried twice (lr=1e-4 and lr=7e-5), both diverged. But this was with the broken optimizer.
3. **Augmented HRM codebase is more stable** — divergence scheduling prevents total collapse, but Q-learning still drifts with broken optimizer.
4. **Don't kill promising runs early** — giga-moose was killed at step 11718 out of concern about q_continue_loss.
5. **Per-cell accuracy != puzzle-solving** — 55% per-cell accuracy means 0% exact puzzle accuracy.
6. **v0.0.3 of adam-atan2 requires C++ CUDA backend** that won't compile on our cluster. v0.2.8 (pure Python) works but has `assert lr > 0` — fixed with `lr=1e-7` placeholder in pretrain.py (scheduler overwrites immediately).

---

## Run 5: Original HRM — `liberal-bee` (RUNNING)

| Parameter | Value |
|---|---|
| Codebase | HRM-main (Original) |
| Dataset | sudoku-extreme-1k-aug-1000 (vanilla) |
| lr | 7e-5 |
| global_batch_size | 384 |
| epochs | 20000 |
| Optimizer | **adam-atan2-pytorch v0.2.8 (FIXED)** |
| torch.compile | enabled |

**Status:** Running on gh001, job 997039. First run with correct optimizer.

---

## Run 6: Augmented HRM — `hopeful-quetzal` (RUNNING)

| Parameter | Value |
|---|---|
| Codebase | hrm-mechanistic-analysis-main (Augmented) |
| Dataset | sudoku-extreme-1k-aug-1000-hint |
| lr | 1e-4 |
| global_batch_size | 768 |
| epochs | 40000 |
| Optimizer | **adam-atan2-pytorch v0.2.8 (FIXED)** |
| torch.compile | enabled |

**Status:** Running on gh002, job 997040. First run with correct optimizer.

---

## Optimizer Fix Summary

| | Old (broken patch) | New (v0.2.8 via shim) |
|---|---|---|
| Weight decay | `grad.add(p, alpha=wd)` (injected into gradient) | `p.mul_(1 - lr * wd)` (applied to params directly) |
| Bias correction | None | Yes (`1 - beta^t`) |
| Update scaling | `atan2(m, sqrt(v))` | `atan2(m/bc1, sqrt(v/bc2)) * a` where a=1.27 |
| Source | Our reimplementation | Real lucidrains library |

---

## What Has NOT Been Tried Yet

1. **Multi-node training (2 GPUs across gh001 + gh002)** — cluster has 1 GPU per node
2. **Lower lr (5e-5)** — fallback if current runs fail
3. **Vanilla dataset through Augmented HRM codebase** — fallback option
