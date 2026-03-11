# SHREK — Coding Implementation TODO

> Strategy: copy AugmentedHRM into `models/OurMODEL/` and make SHREK changes there.
> Base source: `models/hrm-mechanistic-analysis-main/`

---

## What is SHREK — simple explanation

We are building an AI that is better at solving hard puzzles (Sudoku, Mazes, ARC visual puzzles).

The existing models (HRM, AugmentedHRM) have two problems:

**Problem 1 — Getting stuck.**
The model settles on a wrong answer, becomes very confident about it, and stops thinking.
It has no way to realise it is wrong.

**Problem 2 — Going in circles.**
The model keeps changing its answer back and forth between wrong options and never converges.

AugmentedHRM tried to fix this by randomly shaking the model when it gets stuck — like bumping
someone's elbow hoping they write something different. It is blind. Sometimes it helps, often not.

**SHREK fixes both problems properly:**

1. **Flip rate** — after every thinking step, count how many answers changed compared to last step.
   Many changes = still going in circles = keep pushing.
   No changes = settled on something = check if it looks right.

2. **Learned error estimator** — a tiny network that reads the model's internal memory and asks:
   "does this look like a model that got the right answer, or one that is stuck on the wrong answer?"
   It learns this pattern from thousands of training examples.
   When it says "you look wrong" it pushes the model in a meaningful direction — not randomly.

3. **Stagnation delta** — measures how much the model's internal thinking changed this step.
   Feeds this into the halt decision: "am I still making progress, or just spinning?"

**Two sizes — same design:**
- SHREK-Large (27M parameters) — competes with HRM and AugmentedHRM
- SHREK-Tiny (7M parameters) — competes with the Tiny Recursive Model, 4x cheaper to run

**One big efficiency improvement:**
The original model ran itself TWICE every training step to compute one number.
We fixed this by storing that number from the previous step instead. Halves training cost.

---

## What we are building

Two models. Same architecture. Different sizes.

| Model | Params | hidden_size | num_heads |
|---|---|---|---|
| SHREK-Large | ~27M | 512 | 8 |
| SHREK-Tiny  | ~7M  | 256 | 4 |

One codebase. Two config YAML files. Nothing else changes.

---

## SHREK Components

| Component | What it does | Why |
|---|---|---|
| Learned error estimator | Reads z_H → predicts how wrong model is | Catches stuck-but-wrong |
| Flip rate | Fraction of tokens that changed from last step | Catches oscillating predictions |
| Combined error = 0.5 × each | One signal per puzzle per step | Covers all failure modes |
| Error injection | z_H += alpha × error_encoder(error) | Pushes model out of bad states |
| Alpha gate [0,1] | Learned scale, starts 0.01 | Safe — won't destabilize training |
| Stagnation delta | How much z_H moved this step | Tells Q-head: stuck vs converged |
| Q-head with delta | [CLS token, delta] → halt or continue | Better halt decisions |
| No random perturbation | Clean reset instead of noise | Error injection replaces this |
| Q-target caching | Store prev Q in carry, remove second inner() call | Cuts training compute ~50% |
| Auxiliary loss | Trains error estimator to predict real loss | Makes estimator accurate |

---

## Step 1 — Copy AugmentedHRM into OurMODEL ✅

```bash
cp -r models/hrm-mechanistic-analysis-main/ models/OurMODEL/
```

---

## Step 2 — Rewrite `error_singals.py`

File: `models/OurMODEL/models/hrm/error_singals.py`

Replace all three task-specific functions with one universal function.

### `get_error_signal(logits, prev_pred) -> (flip_error, current_pred)`

```python
def get_error_signal(logits, prev_pred):
    current_pred = logits.argmax(dim=-1).to(torch.int32)           # (B, seq_len)
    flip_error   = (current_pred != prev_pred).float().mean(dim=1) # (B,)
    return flip_error, current_pred
```

Works for all datasets. No task rules. No task_type needed.

---

## Step 3 — Modify `hrm_act_v1.py`

File: `models/OurMODEL/models/hrm/hrm_act_v1.py`

### 3a — New components in `__init__` ✅ (partial — add error_estimator)

```python
self.error_estimator = nn.Linear(config.hidden_size, 1)   # NEW: predicts error from z_H
self.error_encoder   = nn.Linear(1, config.hidden_size)   # already exists
self.alpha           = nn.Parameter(torch.tensor(0.01))   # already exists
self.q_head          = CastedLinear(config.hidden_size + 1, 2, bias=True)  # already exists
```

### 3b — Remove random perturbation on reset ✅

### 3c — Update `InnerCarry` dataclass

```python
@dataclass
class HierarchicalReasoningModel_ACTV1InnerCarry:
    z_H:             torch.Tensor    # (B, seq_len, hidden_size)
    z_L:             torch.Tensor    # (B, seq_len, hidden_size)
    prev_pred:       torch.Tensor    # (B, seq_len) int32 — zeros = fresh start
    prev_q_halt:     torch.Tensor    # (B,) cached Q-halt from previous ACT step
    prev_q_continue: torch.Tensor    # (B,) cached Q-continue from previous ACT step
```

### 3d — Update `empty_carry()`

```python
prev_pred       = torch.zeros(batch_size, self.config.seq_len, dtype=torch.int32)
prev_q_halt     = torch.full((batch_size,), -5.0)
prev_q_continue = torch.full((batch_size,), -5.0)
```

### 3e — Update `reset_carry()`

```python
# Zero out prev_pred and cached Q for sequences that just reset
new_prev_pred = carry.prev_pred.clone()
new_prev_pred[reset_flag] = 0

new_prev_q_halt     = carry.prev_q_halt.clone()
new_prev_q_continue = carry.prev_q_continue.clone()
new_prev_q_halt[reset_flag]     = -5.0
new_prev_q_continue[reset_flag] = -5.0
```

### 3f — Update `_Inner.forward()` — remove task_type, add combined error signal

```python
# Snapshot for stagnation delta
z_H_start = carry.z_H

# ... existing no_grad block and 1-step grad ...

output = self.lm_head(z_H)[:, self.puzzle_emb_len:]   # (B, seq_len, vocab)

# --- SHREK: Combined Error Signal ---
z_H_mean     = z_H[:, self.puzzle_emb_len:].mean(dim=1)               # (B, hidden_size)
learned_err  = torch.sigmoid(self.error_estimator(z_H_mean))          # (B,)
flip_err, current_pred = get_error_signal(output, carry.prev_pred)    # (B,), (B, seq_len)
error        = 0.5 * learned_err + 0.5 * flip_err                     # (B,)

# --- SHREK: Error Injection ---
error_emb = self.error_encoder(error.unsqueeze(-1))                   # (B, hidden_size)
with torch.no_grad():
    self.alpha.clamp_(0.0, 1.0)
z_H = z_H + (self.alpha * error_emb.unsqueeze(1)).to(z_H.dtype)

# --- SHREK: Stagnation Delta ---
delta = (z_H.float() - z_H_start.float()).norm(dim=(1,2)) / \
        (z_H_start.float().norm(dim=(1,2)) + 1e-6)                    # (B,)

# --- SHREK: Q-head ---
cls_token = z_H[:, 0].to(torch.float32)
q_input   = torch.cat([cls_token, delta.unsqueeze(-1)], dim=-1)       # (B, hidden_size+1)
q_logits  = self.q_head(q_input).to(torch.float32)                    # (B, 2)

# New carry: store current pred and Q for next step
new_carry = HierarchicalReasoningModel_ACTV1InnerCarry(
    z_H=z_H.detach(), z_L=z_L.detach(),
    prev_pred=current_pred.detach(),
    prev_q_halt=q_logits[..., 0].detach(),
    prev_q_continue=q_logits[..., 1].detach(),
)

# Also expose learned_err so pretrain.py can compute aux_loss
return new_carry, output, (q_logits[..., 0], q_logits[..., 1]), learned_err
```

### 3g — Remove second inner() call — use cached Q instead

```python
# OLD (runs full model twice — doubles training cost):
next_q_halt, next_q_continue = self.inner(new_inner_carry, new_current_data)[-1]
outputs["target_q_continue"] = torch.sigmoid(
    torch.where(is_last_step, next_q_halt, torch.maximum(next_q_halt, next_q_continue)))

# NEW (use cached Q from carry — 50% cheaper):
outputs["target_q_continue"] = torch.sigmoid(
    torch.where(is_last_step,
        new_inner_carry.prev_q_halt,
        torch.maximum(new_inner_carry.prev_q_halt, new_inner_carry.prev_q_continue))
)
```

### 3h — Remove `task_type` from outer `forward()`

Remove `task_type` param from both `_Inner.forward()` and `HierarchicalReasoningModel_ACTV1.forward()`.

---

## Step 4 — Update `pretrain.py`

- Remove `get_task_type()` function entirely
- Remove `task_type` param from `train_batch()` and `evaluate()`
- Remove all `task_type=task_type` passing
- Add auxiliary loss:

```python
# In train_batch(), after forward pass:
# learned_err comes from model output — teaches estimator to predict real loss
aux_loss   = F.mse_loss(learned_err.squeeze(-1), lm_loss.detach())
total_loss = lm_loss + 0.1 * aux_loss
((1 / global_batch_size) * total_loss).backward()
```

---

## Step 5 — Two config YAML files

### `config_large.yaml`
```yaml
hidden_size: 512
num_heads: 8
expansion: 2
halt_max_steps: 32
```

### `config_tiny.yaml`
```yaml
hidden_size: 256
num_heads: 4
expansion: 2
halt_max_steps: 32
```

Head dimension stays same: 512/8 = 256/4 = 64. Same architecture, just smaller.

---

## Step 6 — Smoke Test

```bash
cd models/OurMODEL/

# Large
python3 pretrain.py --config config_large.yaml data_path=data/sudoku-extreme-full epochs=2 eval_interval=1 global_batch_size=32

# Tiny
python3 pretrain.py --config config_tiny.yaml data_path=data/sudoku-extreme-full epochs=2 eval_interval=1 global_batch_size=32
```

Check:
- [ ] Loss goes down (not NaN)
- [ ] `alpha` stays between 0 and 1
- [ ] `aux_loss` decreases — error estimator is learning
- [ ] No device mismatch on `prev_pred`
- [ ] No shape errors in Q-head
- [ ] Training is faster — second inner() call is gone

---

## Benchmark comparison

| Model | Params | FLOPs/step | Gets stuck? | Universal? |
|---|---|---|---|---|
| HRM | 27M | large | Yes | Yes |
| AugmentedHRM | 27M | large | Sometimes | Yes |
| Tiny Recursive | ~7M | small | Yes | Yes |
| SHREK-Large | 27M | large | No | Yes |
| SHREK-Tiny | ~7M | small | No | Yes |

---

## File Checklist

| File | Action | Status |
|---|---|---|
| `models/OurMODEL/models/hrm/error_singals.py` | Rewrite — flip rate only | ⬜ Todo |
| `models/OurMODEL/models/hrm/hrm_act_v1.py` | Steps 3a–3h | ⬜ Todo |
| `models/OurMODEL/pretrain.py` | Step 4 | ⬜ Todo |
| `models/OurMODEL/config_large.yaml` | Create | ⬜ Todo |
| `models/OurMODEL/config_tiny.yaml` | Create | ⬜ Todo |
