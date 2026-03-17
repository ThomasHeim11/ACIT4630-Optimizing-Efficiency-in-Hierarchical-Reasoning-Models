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

**Source:** Kaplan et al., _Scaling Laws for Neural Language Models_, arXiv:2001.08361, 2020.
