Work in progress... This work will be finished summer 2026.

# SHREK-HRM

**SHREK-HRM** stands for **Stagnation Halting Reasoning Error Kernel - Hierarchical Reasoning Model**.

This repository implements SHREK-HRM, an improved HRM variant with learned self-correction. It benchmarks SHREK-HRM against HRM, AugmentedHRM, and Tiny Recursive Models for reasoning efficiency and model dynamics.

## Training Results

All models trained on Sudoku-Extreme (1k examples, 1000 augmentations) for 40,000 epochs on 1x NVIDIA GH200 GPU (Simula Research Laboratory).

| Model | Parameters | Training Time | Hardware |
|---|---|---|---|
| SHREK-Large | 17,839,622 | 2h 54min | 1x GH200 |
| SHREK-Tiny | 5,249,798 | 1h 24min | 1x GH200 |

## Evaluation

Test set exact accuracy (Sudoku-Extreme, 9 permutations):

| Model | Exact Accuracy |
|---|---|
| SHREK-Large | TBD |
| SHREK-Tiny | TBD |
