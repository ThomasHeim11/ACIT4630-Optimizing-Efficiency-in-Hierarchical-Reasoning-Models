# Measuring FLOPs for HRM-Family Models

## Approach: Analytical Formula

We compute GFLOPs analytically from each model's config YAML — the same method used in standard deep learning papers for comparing model efficiency.

**Why analytical:**
- Hardware-independent — same number regardless of GPU
- Reproducible — anyone can verify from the config files alone
- Fair — identical formula applied to all four models
- Standard practice for transformer architecture comparisons

**We do not use:** PyTorch profiler, CodeCarbon, or energy measurements (out of scope).

---

## Unit: GFLOPs

**1 GFLOPs = 1 billion floating point operations**

We report **GFLOPs per puzzle** (one inference pass on one test input). This is the standard unit used in efficiency-focused deep learning papers — large enough to be readable for small models (~7M–27M parameters), small enough to avoid scientific notation.

Following Bianco et al. (IEEE 2018): **1 multiply-add = 2 FLOPs** (one multiply + one add counted separately). Each matrix multiplication of shape `(m, k) × (k, n)` costs `2 × m × k × n` FLOPs.

---

## Formula

Directly from Kaplan et al. (2020) Table 1 — FLOPs per token per layer:

```
Attention QKV:       2 × n_layers × d_model × 3d_attn
Attention scores:    2 × n_layers × n_ctx   × d_attn
Attention out proj:  2 × n_layers × d_attn  × d_model
FFN:                 2 × n_layers × 2       × d_model × d_ff

where d_ff = d_model × expansion  (ffn hidden size)
      d_attn = d_model             (standard setting)
```

For our models `d_model >> n_ctx/12` holds (e.g. 512 >> 75), so the attention score term is small — but we keep the full formula since `seq_len` differs per benchmark (81 for Sudoku, 900 for ARC/Maze), making the explicit breakdown more transparent.

FFN uses SwiGLU (3 linear layers: gate, up, down) so the FFN term is:
```
gate + up:  2 × n_layers × 2 × d_model × d_ff
down:       2 × n_layers × d_ff × d_model
```

Scaled to the full model:

```
flops_per_L_block    = transformer_block(L_layers)
flops_per_H_block    = transformer_block(H_layers)
flops_per_outer_step = H_cycles × (L_cycles × flops_per_L_block + flops_per_H_block)

total_GFLOPs = avg_act_steps × flops_per_outer_step / 1e9
```

`avg_act_steps` is the only runtime value — measured during evaluation.

---

## Model Config Values

| Model | hidden | heads | expansion | H/L layers | H/L cycles |
|---|---|---|---|---|---|
| HRM | 512 | 8 | 4 | 4/4 | 2/2 |
| TRM | 256 | 4 | 4 | 2/2 | 3/6 |
| AugmentedHRM | 512 | 8 | 4 | 4/4 | 2/2 |
| SHREK-Large | 512 | 8 | 2 | 4/4 | 2/2 |
| SHREK-Tiny | 256 | 4 | 2 | 4/4 | 2/2 |

SHREK adds one extra linear layer per outer step (error encoder: `seq_len → hidden_size`) — included in total.

---

## avg_act_steps

Because ACT lets the model decide how many steps to take, the step count varies per puzzle. We measure:

```
avg_act_steps = mean outer ACT steps taken per puzzle across the full test set
```

Recorded per model per benchmark during `evaluate.py`. This is the only number that comes from actually running the model.

---

## Workflow

1. Train all models on cluster
2. Run `evaluate.py` on each checkpoint — records accuracy + avg_act_steps
3. Plug avg_act_steps into formula → GFLOPs/puzzle
4. Report in results table

---

## Results Table (to fill in after training)

| Model | Benchmark | Accuracy | avg_act_steps | GFLOPs/puzzle |
|---|---|---|---|---|
| HRM | Sudoku | | | |
| HRM | ARC-AGI-1 | | | |
| HRM | ARC-AGI-2 | | | |
| TRM | Sudoku | | | |
| TRM | ARC-AGI-1 | | | |
| TRM | ARC-AGI-2 | | | |
| AugmentedHRM | Sudoku | | | |
| AugmentedHRM | ARC-AGI-1 | | | |
| AugmentedHRM | ARC-AGI-2 | | | |
| SHREK-Large | Sudoku | | | |
| SHREK-Large | ARC-AGI-1 | | | |
| SHREK-Large | ARC-AGI-2 | | | |
| SHREK-Tiny | Sudoku | | | |
| SHREK-Tiny | ARC-AGI-1 | | | |
| SHREK-Tiny | ARC-AGI-2 | | | |
