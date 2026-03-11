# SHREK — Coding Implementation TODO

> Strategy: copy AugmentedHRM into `models/OurMODEL/` and make SHREK changes there.
> Base source: `models/hrm-mechanistic-analysis-main/`

---

## Step 1 — Copy AugmentedHRM into OurMODEL ✅

```bash
cp -r models/hrm-mechanistic-analysis-main/ models/OurMODEL/
```

After this, all SHREK changes go in `models/OurMODEL/` — never touch the original.

---

## Step 2 — Create Error Signal Module ✅ (needs update — see Step 2b)

~~Task-specific approach (implemented but being replaced):~~
~~Three functions: `compute_sudoku_error`, `compute_maze_error`, `compute_arc_error`~~
~~Dispatcher: `get_error_signal(logits, task_type: str)`~~

**Why we replaced it:** hardcoding rules per dataset is fragile. New dataset = new function.

---

## Step 2b — Replace Error Signal with Universal Flip Rate

File: `models/OurMODEL/models/hrm/error_singals.py`

**Core idea:** instead of checking task-specific rules, compare predictions *across time*.

At every reasoning step, ask: "Did I change my answer compared to last step?"

```
flip_rate = fraction of output tokens that changed from previous step
```

- `flip_rate = 1.0` → model keeps changing its mind → still oscillating → keep thinking
- `flip_rate = 0.0` → predictions stabilized → converged (possibly correct)

**Why this is better than entropy:**
- Entropy measures uncertainty at a single point in time
- Entropy misses "confident but wrong" (high confidence on the wrong answer)
- Flip rate catches oscillating wrong answers: even if the model is confident each
  step, if it keeps predicting different things, flip rate stays high

**Why zeros = first step:**
- `prev_pred` is initialized to all zeros (token 0 = PAD)
- Real outputs are almost never all-zero
- So first step: `flip_rate ≈ 1.0` — correct, maximum uncertainty at start

**Replace the 3 functions with one:**
```python
def get_error_signal(logits, prev_pred) -> Tuple[Tensor, Tensor]:
    current_pred = logits.argmax(dim=-1).to(torch.int32)     # (B, seq_len)
    flip_rate    = (current_pred != prev_pred).float().mean(dim=1)  # (B,)
    return flip_rate, current_pred
```

Returns both the error AND `current_pred` so the caller can store it in the carry.

Works for all datasets: Sudoku, Maze, ARC-AGI-1, ARC-AGI-2, any future task.

---

## Step 3 — Modify `hrm_act_v1.py` for SHREK ✅ (needs partial update — see 3f)

File: `models/OurMODEL/models/hrm/hrm_act_v1.py`

### 3a — Add new parameters to `__init__` ✅
- `self.error_encoder = nn.Linear(1, hidden_size)`
- `self.alpha = nn.Parameter(torch.tensor(0.01))`
- `self.q_head = CastedLinear(hidden_size + 1, 2)` (was `hidden_size, 2`)

### 3b — Remove random perturbation on reset ✅
- `reset_carry(use_default=False)` now does clean init instead of random noise

### 3c — Error injection and stagnation delta in forward ✅
- Error injection: `z_H = z_H + alpha * error_encoder(error)`
- Stagnation delta: `delta = norm(z_H - z_H_start) / norm(z_H_start)`
- Q-head input: `[cls_token, delta]`

### 3d — `task_type` added to forward ✅ (being removed in 3f)

### 3e — `z_H_start` initialized before loop ✅

### 3f — Add `prev_pred` to carry, remove `task_type`

**Changes needed:**

**1. `InnerCarry` dataclass — add `prev_pred`:**
```python
@dataclass
class HierarchicalReasoningModel_ACTV1InnerCarry:
    z_H: torch.Tensor
    z_L: torch.Tensor
    prev_pred: torch.Tensor   # (B, seq_len) int32 — zeros = fresh start
```

**2. `empty_carry()` — initialize `prev_pred` to zeros:**
```python
prev_pred=torch.zeros(batch_size, self.config.seq_len, dtype=torch.int32)
```

**3. `reset_carry()` — zero out `prev_pred` for halted sequences:**
```python
new_prev_pred = carry.prev_pred.clone()
new_prev_pred[reset_flag] = 0   # reset sequences start fresh
# add prev_pred=new_prev_pred to both return branches
```

**4. `_Inner.forward()` — update error signal call:**
```python
# OLD (task-specific):
error = get_error_signal(output, task_type)

# NEW (universal):
error, current_pred = get_error_signal(output, carry.prev_pred)
```

**5. `new_carry` — store current prediction:**
```python
new_carry = HierarchicalReasoningModel_ACTV1InnerCarry(
    z_H=z_H.detach(),
    z_L=z_L.detach(),
    prev_pred=current_pred.detach(),   # ← store for next step
)
```

**6. Remove `task_type` parameter from both `_Inner.forward()` and outer `forward()`**
- Remove from all 3 `self.inner(...)` call sites too

---

## Step 4 — Pass `task_type` from Training Loop ✅ (being reverted — see Step 4b)

~~`get_task_type(data_path)` → `task_type` → passed to model.forward()~~

---

## Step 4b — Remove `task_type` from `pretrain.py`

Since `task_type` is no longer needed (error signal is universal), clean up pretrain.py:

- Remove `get_task_type()` function entirely
- Remove `task_type: str = "sudoku"` param from `train_batch()` signature
- Remove `task_type=task_type` from model call inside `train_batch()`
- Remove `task_type: str = "sudoku"` param from `evaluate()` signature
- Remove `task_type=task_type` from model call inside `evaluate()`
- Remove `task_type = get_task_type(config.data_path)` from `launch()`
- Remove `task_type=task_type` from `train_batch()` and `evaluate()` calls in `launch()`

---

## Step 5 — Smoke Test

```bash
cd models/OurMODEL/
python3 pretrain.py data_path=data/sudoku-extreme-full epochs=2 eval_interval=1 global_batch_size=32
```

Check:

- [ ] Loss goes down (not NaN)
- [ ] `alpha` parameter is included in optimizer
- [ ] No shape errors in Q-head (hidden+1 dimension)
- [ ] Error signal returns a tensor, not a Python float
- [ ] `prev_pred` is on the correct device (CUDA) — no device mismatch errors

---

## Summary of SHREK Components

| Component | What it does | Where |
|---|---|---|
| `error_encoder` + `alpha` | Maps flip rate → z_H injection | `hrm_act_v1.py __init__` |
| Flip rate error signal | How much did predictions change? | `error_singals.py` |
| `prev_pred` in carry | Stores last step's predictions | `InnerCarry` dataclass |
| Stagnation delta | How much did z_H move? | `hrm_act_v1.py forward()` |
| Q-head `[cls, delta]` | Halt decision with stagnation info | `hrm_act_v1.py forward()` |
| No random perturbation | Error injection replaces noise | `reset_carry()` |

---

## File Checklist

| File | Action | Status |
| --- | --- | --- |
| `models/OurMODEL/` | Copy from hrm-mechanistic-analysis-main | ✅ Done |
| `models/OurMODEL/models/hrm/error_singals.py` | Rewrite with flip rate (Step 2b) | ⬜ Todo |
| `models/OurMODEL/models/hrm/hrm_act_v1.py` | Add `prev_pred` to carry, remove `task_type` (Step 3f) | ⬜ Todo |
| `models/OurMODEL/pretrain.py` | Remove `task_type` and `get_task_type` (Step 4b) | ⬜ Todo |
