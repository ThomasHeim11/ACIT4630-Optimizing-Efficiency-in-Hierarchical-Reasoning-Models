# SHREK — Coding Implementation TODO

> Strategy: copy AugmentedHRM into `models/OurMODEL/` and make SHREK changes there.
> Base source: `models/hrm-mechanistic-analysis-main/`

---

## Step 1 — Copy AugmentedHRM into OurMODEL✅

```bash
cp -r models/hrm-mechanistic-analysis-main/ models/OurMODEL/
```

After this, all SHREK changes go in `models/OurMODEL/` — never touch the original.

---

## Step 2 — Create Error Signal Module✅

Create new file: `models/OurMODEL/models/hrm/error_signals.py`

Implement three functions:

### `compute_sudoku_error(logits, seq_len=82)`

- Decode argmax prediction from logits → 9×9 board
- Reuse logic from `visualization/landscape.py:differentiable_conflict_loss()`
- Return scalar float (number of row/col/box violations)

### `compute_maze_error(logits)`

- Decode predicted path from logits
- Count broken steps (cells that are not adjacent or are walls)
- Return scalar float

### `compute_arc_error(logits)`

- Compute entropy of softmax output: `-sum(p * log(p + 1e-8))`
- High entropy = uncertain = high error
- Return scalar float (mean over all output positions)

### `get_error_signal(logits, task_type: str) -> torch.Tensor`

- Dispatcher: routes to the right function based on `task_type`
- `task_type` values: `"sudoku"`, `"maze"`, `"arc"`
- Returns a (1,) shaped tensor on the same device as logits

---

## Step 3 — Modify `hrm_act_v1.py` for SHREK

File: `models/OurMODEL/models/hrm/hrm_act_v1.py`

### 3a — Add new parameters to `__init__`

```python
from .error_signals import get_error_signal

# Inside __init__, after existing layers:
self.error_encoder = nn.Linear(1, config.hidden_size)
self.alpha = nn.Parameter(torch.tensor(0.01))   # learned gate, starts near 0
# Q-head now takes hidden + 1 stagnation scalar:
self.q_head = CastedLinear(config.hidden_size + 1, 2)  # was (hidden_size, 2)
```

> Remove the old `self.q_head = CastedLinear(hidden_size, 2)` line.

### 3b — Remove random perturbation on reset

Find `reset_carry(use_default=False)` — it adds random noise to `z_H`.
Replace with clean zero init (no noise). SHREK uses error injection instead.

```python
# BEFORE (AugmentedHRM):
z_H = torch.randn_like(z_H) * perturbation_scale

# AFTER (SHREK): remove this line entirely
```

### 3c — Modify the outer ACT loop

Find the outer loop that updates `z_H` across H-level steps.

After each H-level update, before the Q-head halt decision:

```python
# --- SHREK: Error-Conditioned Injection ---
logits = self.model.inner.lm_head(z_H)[1:]       # decode current state
error = get_error_signal(logits, task_type)        # scalar error
error_emb = self.error_encoder(error.view(1, 1))  # (1, hidden_size)
z_H = z_H + self.alpha * error_emb               # inject (alpha starts ~0)

# --- SHREK: Stagnation-Aware ACT ---
if z_H_prev is not None:
    delta = torch.norm(z_H - z_H_prev) / (torch.norm(z_H_prev) + 1e-6)
else:
    delta = torch.zeros(1, device=z_H.device)

z_H_prev = z_H.detach().clone()

# Feed z_H_cls + delta into Q-head
cls_token = z_H[0]                                  # (hidden_size,)
q_input = torch.cat([cls_token, delta.unsqueeze(0)], dim=-1)  # (hidden_size+1,)
q_values = self.q_head(q_input)                     # (2,) -> halt or continue
```

### 3d — Add `task_type` parameter to `forward()`

```python
def forward(self, inputs, puzzle_identifiers, task_type: str = "sudoku"):
```

Pass `task_type` down to wherever the outer ACT loop lives.

### 3e — Initialize `z_H_prev = None` before the outer loop starts

---

## Step 4 — Pass `task_type` from Training Loop

File: `models/OurMODEL/pretrain.py`

- Read `task_type` from dataset metadata (or derive from dataset path name)
- Pass it to `model.forward(inputs, puzzle_identifiers, task_type=task_type)`

---

## Step 5 — Smoke Test

```bash
cd models/OurMODEL/
python3 pretrain.py epochs=2 eval_interval=1 global_batch_size=32
```

Check:

- [ ] Loss goes down (not NaN)
- [ ] `alpha` parameter is included in optimizer
- [ ] No shape errors in Q-head (hidden+1 dimension)
- [ ] Error signal returns a tensor, not a Python float

---

## File Checklist

| File                                          | Action                                           |
| --------------------------------------------- | ------------------------------------------------ |
| `models/OurMODEL/`                            | Copy from hrm-mechanistic-analysis-main (Step 1) |
| `models/OurMODEL/models/hrm/error_signals.py` | Create new (Step 2)                              |
| `models/OurMODEL/models/hrm/hrm_act_v1.py`    | Modify (Step 3)                                  |
| `models/OurMODEL/pretrain.py`                 | Modify (Step 4)                                  |
