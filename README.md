# SHREK-HRM: Self-Corrective Hierarchical Reasoning

**SHREK-HRM** (**S**tagnation **H**alting **R**easoning **E**rror **K**ernel) extends the Hierarchical Reasoning Model with learned self-correction. Instead of relying on expensive inference-time techniques (checkpoint ensembles, token permutations), SHREK injects an error signal directly into the reasoning process, enabling the model to detect and correct its own mistakes during inference.

This repository benchmarks SHREK-HRM against HRM, Augmented HRM, and Tiny Recursive Models on Sudoku-Extreme and Maze-Hard.

---

## Model Architectures

| Model | Parameters | Layers | Key Feature |
|---|---|---|---|
| Original HRM | ~27M | 4H + 4L, hidden=512 | Baseline recursive reasoning |
| Augmented HRM | ~27M | 4H + 4L, hidden=512 | + Data mixing, bootstrap, relabeling |
| TRM Attention | ~7M | 2 layers, hidden=512 | Simplified recursive reasoning |
| TRM MLP | ~7M | 2 layers, hidden=512 | MLP-only variant (no attention) |
| **SHREK Large** | **~27M** | **4H + 4L, hidden=512** | **+ Error estimator, error injection, EMA** |
| **SHREK Tiny** | **~8M** | **2H + 2L, hidden=512** | **+ Error estimator, error injection, EMA** |

All HRM-family models share: `H_cycles=2, L_cycles=2, halt_max_steps=16, expansion=4`.
TRM uses: `H_cycles=3, L_cycles=6 (Sudoku) / L_cycles=4 (Maze)`.

---

## SHREK Architecture

SHREK adds two components to the base HRM architecture:

### 1. Error-Conditioned Input Injection

After each reasoning step, SHREK computes an error signal and injects it back into `z_H`:

```
z_H = z_H + alpha * error_encoder(error) / sqrt(hidden_size)
```

The error signal combines:
- **Flip rate** — fraction of output tokens that changed vs previous step (task-agnostic, no learning needed)
- **Learned error estimator** — neural network predicting model's error from detached `z_H`

Alpha follows a linear warmup from 0 to 0.01 over 5000 steps.

### 2. Stagnation-Aware Q-Head

The Q-head receives a stagnation signal measuring how much `z_H` changed:

```
delta = ||z_H_end - z_H_start|| / (||z_H_start|| + epsilon)
q_logits = q_head(concat(z_H[:, 0], delta))
```

This helps the Q-head distinguish "converged correctly" from "stuck on wrong answer."

### Parameter Overhead

| Component | Parameters |
|---|---|
| Error encoder (1 -> hidden_size) | 513 |
| Error estimator (hidden_size -> 1) | 513 |
| Stagnation scalar in Q-head | +2 |
| **Total overhead** | **~1K (<0.01%)** |

---

## Results

### Sudoku-Extreme (1000 training examples)

All models trained on 1x NVIDIA GH200 GPU (102GB VRAM).

**Single Checkpoint Test Accuracy** (`all.exact_accuracy`):

| Model | Parameters | Exact Accuracy | Paper Target |
|---|---|---|---|
| Original HRM | ~27M | 53% | 55% (+-2%) |
| Augmented HRM | ~27M | 54.2% | 59.9% |
| TRM MLP | ~7M | Pending | ~87% |
| TRM Attention | ~7M | Pending | ~75% |
| **SHREK Large** | **~27M** | **70.6%** | **—** |
| **SHREK Tiny** | **~8M** | **61.6%** | **—** |

**Ensemble Evaluation** (10 checkpoints + 9 token permutations, 1000 test samples):

| Model | Exact Accuracy | Paper Target |
|---|---|---|
| Augmented HRM | 90.5% | 96.9% |
| **SHREK Large** | **90.2%** | **—** |
| **SHREK Tiny** | **80.5%** | **—** |

**Key findings:**
- SHREK Large (70.6%) beats all baselines by 16+ points on single checkpoint
- SHREK Tiny (61.6%) outperforms 27M-parameter models with only 8M parameters
- Ensemble gives diminishing returns for SHREK — it's already good without inference tricks

### Maze-Hard (1000 training examples)

Training in progress.

### ARC-AGI

Not yet attempted.

---

## Reproducing Experiments

### Prerequisites

- NVIDIA GPU with CUDA 12.6 (tested on GH200, 102GB VRAM)
- Python 3.10
- PyTorch 2.10+

### Setup

```bash
pip install torch flash-attn einops tqdm coolname pydantic argdantic wandb omegaconf hydra-core
pip install adam-atan2-pytorch==0.2.8

# Create import shim for backward compatibility with adam-atan2 v0.0.3
echo "from adam_atan2_pytorch.adam_atan2 import AdamAtan2 as AdamATan2" > \
    $(python -c "import adam_atan2; print(adam_atan2.__path__[0])")/adam_atan2.py

wandb login
```

### Dataset Preparation

```bash
# Sudoku-Extreme (vanilla)
python dataset/build_sudoku_dataset.py \
    --output-dir data/sudoku-extreme-1k-aug-1000 \
    --subsample-size 1000 --num-aug 1000

# Sudoku-Extreme (with hints)
python dataset/build_sudoku_dataset.py \
    --output-dir data/sudoku-extreme-1k-aug-1000-hint \
    --subsample-size 1000 --num-aug 1000 --hint

# Maze-Hard
python dataset/build_maze_dataset.py
```

### Training Commands

All training scripts are in each model's `script/train/` directory and can be submitted via SLURM:

```bash
module load slurm

# Sudoku-Extreme
sbatch models/HRM\(Original\)/HRM-main/script/train/train_hrm_sudoku_1gpu.sh
sbatch models/hrm-mechanistic-analysis-main/scripts/train_augmented_hrm_sudoku_1gpu.sh
sbatch models/SHREK-HRM/script/train/train_shrek_large_sudoku.sh
sbatch models/SHREK-HRM/script/train/train_shrek_tiny_sudoku.sh
sbatch models/TinyRecursiveModels/script/train/train_trm_mlp_sudoku.sh
sbatch models/TinyRecursiveModels/script/train/train_trm_att_sudoku.sh

# Maze-Hard
sbatch models/HRM\(Original\)/HRM-main/script/train/train_hrm_maze_1gpu.sh
sbatch models/hrm-mechanistic-analysis-main/scripts/train_augmented_hrm_maze_1gpu.sh
sbatch models/SHREK-HRM/script/train/train_shrek_large_maze.sh
sbatch models/SHREK-HRM/script/train/train_shrek_tiny_maze.sh
sbatch models/TinyRecursiveModels/script/train/train_trm_att_maze.sh
```

**Hyperparameter summary:**

| Parameter | Original HRM (Sudoku) | All others |
|---|---|---|
| lr | 7e-5 | 1e-4 |
| global_batch_size | 384 | 768 |
| epochs | 20,000 | 40,000 (Sudoku) / 20,000 (Maze) |
| weight_decay | 1.0 | 1.0 |
| eval_interval | 1,000 | 1,000 |

Original HRM uses the paper's 1-GPU recipe. All others use the 8-GPU recipe since GH200 has enough VRAM.

### Evaluation

**Single checkpoint:** Check `all.exact_accuracy` in Weights & Biases.

**Ensemble evaluation** (10 checkpoints + 9 token permutations):

```bash
cd models/hrm-mechanistic-analysis-main  # or SHREK-HRM
DISABLE_COMPILE=1 python3 batch_inference.py \
    --checkpoints "step1,step2,...,step10" \
    --permutes 9 \
    --num_batch 10 --batch_size 100
```

### FLOPs Measurement

```bash
sbatch flops/measure_all_flops.sh
```

---

## Repository Structure

```
HMR/
├── models/
│   ├── HRM(Original)/HRM-main/       # Original HRM (Wang et al., 2025)
│   ├── hrm-mechanistic-analysis-main/ # Augmented HRM (Ren & Liu, 2026)
│   ├── TinyRecursiveModels/           # TRM (Jolicoeur-Martineau, 2025)
│   └── SHREK-HRM/                     # SHREK-HRM (ours)
├── dataset/data/                      # Training and test datasets
├── checkpoints/                       # Trained model checkpoints
├── flops/                             # FLOPs measurement scripts
├── papers/                            # Reference papers
├── BUGFIX.md                          # SHREK bug fix documentation
└── RunningModelsTried.md              # Training run history and lessons
```

---

## Hardware

| Spec | Value |
|---|---|
| GPU | NVIDIA GH200 480GB |
| VRAM | 102 GB |
| CPU Cores | 72 |
| RAM | 573 GB |
| CUDA | 12.6 |
| Cluster | Simula Research Laboratory HPC (2 nodes: gh001, gh002) |

---

## Computational Complexity (FLOPs)

All models compared on **GFLOPs per puzzle** using analytical formula following Kaplan et al. (2020):

```
FLOPs per token per layer:
  Attention QKV:       2 * d_model * 3 * d_attn
  Attention scores:    2 * n_ctx * d_attn
  Attention out proj:  2 * d_attn * d_model
  FFN:                 2 * 2 * d_model * d_ff

Total per puzzle = avg_act_steps * H_cycles * (L_cycles * L_block + H_block) * seq_len
```

`avg_act_steps` is the only runtime value, measured from W&B. Full formula used (not `C ~ 2N` shortcut) because `seq_len` differs per benchmark.

---

## Pre-trained Checkpoints & Dataset

- **Checkpoints**: [HRM-Reproduction-Checkpoints](https://huggingface.co/ThomasHeim/HRM-Reproduction-Checkpoints/tree/main) — All trained model weights (Original HRM, SHREK Large/Tiny, TRM Attention/MLP)
- **Dataset**: [HRM-dataset](https://huggingface.co/datasets/ThomasHeim/HRM-dataset) — Sudoku-Extreme and Maze-Hard datasets

---

## Source Papers

- **HRM**: [Hierarchical Reasoning Model](https://arxiv.org/abs/2506.21734) (Wang et al., 2025)
- **Augmented HRM**: [Are Your Reasoning Models Reasoning or Guessing?](https://arxiv.org/abs/2601.10679) (Ren & Liu, 2026)
- **TRM**: [Less is More: Recursive Reasoning with Tiny Networks](https://arxiv.org/abs/2510.04871) (Jolicoeur-Martineau, 2025)

## Acknowledgements

Built on code from [sapientinc/HRM](https://github.com/sapientinc/HRM), [renrua52/hrm-mechanistic-analysis](https://github.com/renrua52/hrm-mechanistic-analysis), and [TRM](https://github.com/AlexiaJM/TinyRecursiveModels).
