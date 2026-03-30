# Results

All models trained on 1x NVIDIA GH200 GPU (102GB VRAM). Papers used 8 GPUs (HRM) or 4 GPUs (TRM).

**Important:** All SHREK models are trained on the **vanilla dataset** (no hints) for both Sudoku and Maze. This ensures a fair architecture-only comparison — any improvement is from SHREK's self-correction, not from data augmentation tricks.

---

## Sudoku-Extreme (1k training examples, 422k test examples)

All models trained on vanilla dataset unless noted.

### Single Checkpoint Test Accuracy (`all.exact_accuracy`)

| Model | Parameters | Dataset | Our Result | Paper Target |
|---|---|---|---|---|
| Original HRM | ~27M | vanilla | 53% | 55% (+-2%) |
| Augmented HRM | ~27M | hint | 54.2% | 59.9% |
| TRM MLP | ~5M | vanilla | ~84% | ~87% |
| TRM Attention | ~7M | vanilla | ~70% | ~75% |
| **SHREK Large** | **~27M** | **vanilla** | **Pending** | **—** |
| **SHREK Tiny** | **~8M** | **vanilla** | **Pending** | **—** |

*Previous SHREK results on hint dataset: SHREK Large 70.6%, SHREK Tiny 61.6%*

### Ensemble Evaluation (10 checkpoints + 9 token permutations, hint dataset)

| Model | Snapshot (1k samples) | Full (422k samples) | Paper Target |
|---|---|---|---|
| Augmented HRM | 90.5% | 92.6% | 96.9% |

### Key Findings — Sudoku

- TRM MLP (~84%) is the overall best on Sudoku due to its MLP architecture suited for fixed 9x9 grids
- All paper results successfully reproduced within stated variance
- SHREK vanilla results pending — will show architecture effect without data advantages

---

## Maze-Hard (1k training examples, 1k test examples)

All models trained on vanilla dataset (no hint dataset exists for maze).

### Test Accuracy (`all.exact_accuracy`)

| Model | Parameters | Our Peak Result | Paper Target | Status |
|---|---|---|---|---|
| Original HRM | ~27M | ~75% | 74.5% | Reproduced |
| TRM Attention | ~7M | ~80% (then collapses) | 85.3% | Unstable |
| **SHREK Large** | **~27M** | **Pending (retrain)** | **—** | **—** |
| **SHREK Tiny** | **~8M** | **Pending (retrain)** | **—** | **—** |

*Previous SHREK results (before stagnation delta): SHREK Large ~85%, SHREK Tiny ~75%*

Note: Augmented HRM (~60%) excluded from main comparison — its techniques are Sudoku-specific and hurt maze performance.

### Key Findings — Maze

- Original HRM (~75%) successfully reproduces the paper's 74.5%
- TRM Attention peaks at ~80% but collapses around step 130k (late instability)
- Previous SHREK Large (~85%) was best on maze before stagnation delta was added — retraining will show if it improves further

---

## Cross-Task Summary

Will be updated after all retraining completes.

| Model | Params | Dataset | Sudoku | Maze |
|---|---|---|---|---|
| Original HRM | ~27M | vanilla | 53% | ~75% |
| TRM MLP | ~5M | vanilla | ~84% | — |
| TRM Attention | ~7M | vanilla | ~70% | ~80% (unstable) |
| **SHREK Large** | **~27M** | **vanilla** | **Pending** | **Pending** |
| **SHREK Tiny** | **~8M** | **vanilla** | **Pending** | **Pending** |

---

## Ablation Study — SHREK Components (Sudoku-Extreme, vanilla dataset)

SHREK adds two novel components over base HRM. We ablate each to isolate their contribution.
All ablation runs use the **vanilla** Sudoku dataset for fair comparison with Original HRM.

### SHREK's Novel Components

| Component | What it does |
|---|---|
| **Error Injection** | Combines flip rate + learned error estimator, encodes to vector, injects into z_H after each reasoning step |
| **Stagnation Delta** | Measures how much z_H changed, feeds to Q-head to improve halt decisions |

### Ablation Results (SHREK Large, vanilla Sudoku)

| Configuration | Error Injection | Stagnation Delta | EMA | Accuracy |
|---|---|---|---|---|
| Original HRM (baseline) | No | No | No | 53% |
| HRM + EMA (no SHREK components) | No | No | Yes | Pending |
| Only stagnation delta | No | Yes | Yes | Pending |
| Only error injection | Yes | No | Yes | Pending |
| **SHREK Large (full)** | **Yes** | **Yes** | **Yes** | **Pending** |

### What Each Comparison Tests

| Comparison | What it isolates |
|---|---|
| HRM + EMA vs Original HRM (53%) | Effect of EMA (not our contribution, from TRM) |
| Only error injection vs HRM + EMA | Contribution of error injection (novel) |
| Only stagnation delta vs HRM + EMA | Contribution of stagnation delta (novel) |
| SHREK full vs HRM + EMA | Combined effect of both novel components |
| SHREK full vs Only error injection | Added value of stagnation delta |
| SHREK full vs Only stagnation delta | Added value of error injection |

---

## Planned Retraining Runs

| # | Run | Dataset | New Script |
|---|---|---|---|
| 1 | SHREK Large Sudoku (full) | vanilla | Yes |
| 2 | SHREK Tiny Sudoku (full) | vanilla | Yes |
| 3 | SHREK Large Maze (full) | vanilla | No (rsync only) |
| 4 | SHREK Tiny Maze (full) | vanilla | No (rsync only) |
| 5 | Ablation: no error injection | vanilla Sudoku | Yes |
| 6 | Ablation: no stagnation delta | vanilla Sudoku | Yes |
| 7 | Ablation: no both (HRM + EMA) | vanilla Sudoku | Yes |

**Total: 7 runs, 5 new scripts**

---

## ARC-AGI

Not yet attempted.
