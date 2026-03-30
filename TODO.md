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

| Model       | Params | hidden_size | num_heads |
| ----------- | ------ | ----------- | --------- |
| SHREK-Large | ~27M   | 512         | 8         |
| SHREK-Tiny  | ~7M    | 256         | 4         |

One codebase. Two config YAML files. Nothing else changes.

---

## SHREK Components

| Component                   | What it does                                      | Why                               |
| --------------------------- | ------------------------------------------------- | --------------------------------- |
| Learned error estimator     | Reads z_H → predicts how wrong model is           | Catches stuck-but-wrong           |
| Flip rate                   | Fraction of tokens that changed from last step    | Catches oscillating predictions   |
| Combined error = 0.5 × each | One signal per puzzle per step                    | Covers all failure modes          |
| Error injection             | z_H += alpha × error_encoder(error)               | Pushes model out of bad states    |
| Alpha gate [0,1]            | Learned scale, starts 0.01                        | Safe — won't destabilize training |
| Stagnation delta            | How much z_H moved this step                      | Tells Q-head: stuck vs converged  |
| Q-head with delta           | [CLS token, delta] → halt or continue             | Better halt decisions             |
| No random perturbation      | Clean reset instead of noise                      | Error injection replaces this     |
| Q-target caching            | Store prev Q in carry, remove second inner() call | Cuts training compute ~50%        |
| Auxiliary loss              | Trains error estimator to predict real loss       | Makes estimator accurate          |

---

## Step 1 — Copy AugmentedHRM into OurMODEL ✅

```bash
cp -r models/hrm-mechanistic-analysis-main/ models/OurMODEL/
```

---

## Step 2 — Rewrite `error_singals.py` ✅

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

## Step 3 — Modify `hrm_act_v1.py` ✅

File: `models/OurMODEL/models/hrm/hrm_act_v1.py`

### 3a — New components in `__init__` ✅

```python
self.error_estimator = nn.Linear(config.hidden_size, 1)   # NEW: predicts error from z_H
self.error_encoder   = nn.Linear(1, config.hidden_size)   # already exists
self.alpha           = nn.Parameter(torch.tensor(0.01))   # already exists
self.q_head          = CastedLinear(config.hidden_size + 1, 2, bias=True)  # already exists
```

### 3b — Remove random perturbation on reset ✅

### 3c — Update `InnerCarry` dataclass ✅

### 3d — Update `empty_carry()` ✅

### 3e — Update `reset_carry()` ✅

### 3f — Update `_Inner.forward()` — remove task_type, add combined error signal ✅

### 3g — Remove second inner() call — use cached Q instead ✅

### 3h — Remove `task_type` from outer `forward()` ✅

---

## Step 4 — Update `pretrain.py` ✅

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

## Step 5 — Two config YAML files ✅

Files: `config/arch/shrek_large.yaml` and `config/arch/shrek_tiny.yaml`

Head dimension stays same: 512/8 = 256/4 = 64. Same architecture, just smaller.

---

## Step 6 — Smoke Test

```bash
cd models/SHREK-HRM/

# Large (~27M)
python3 pretrain.py arch=shrek_large data_path=data/sudoku-extreme-full epochs=2 eval_interval=1 global_batch_size=32

# Tiny (~7M)
python3 pretrain.py arch=shrek_tiny data_path=data/sudoku-extreme-full epochs=2 eval_interval=1 global_batch_size=32
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

| Model          | Params | FLOPs/step | Gets stuck? | Universal? |
| -------------- | ------ | ---------- | ----------- | ---------- |
| HRM            | 27M    | large      | Yes         | Yes        |
| AugmentedHRM   | 27M    | large      | Sometimes   | Yes        |
| Tiny Recursive | ~7M    | small      | Yes         | Yes        |
| SHREK-Large    | 27M    | large      | No          | Yes        |
| SHREK-Tiny     | ~7M    | small      | No          | Yes        |

---

## File Checklist

| File                                          | Action                   | Status  |
| --------------------------------------------- | ------------------------ | ------- |
| `models/OurMODEL/models/hrm/error_singals.py` | Rewrite — flip rate only | ✅ Done |
| `models/OurMODEL/models/hrm/hrm_act_v1.py`    | Steps 3a–3h              | ✅ Done |
| `models/OurMODEL/pretrain.py`                 | Step 4                   | ✅ Done |
| `config/arch/shrek_large.yaml`                | Create                   | ✅ Done |
| `config/arch/shrek_tiny.yaml`                 | Create                   | ✅ Done |

### TODO FRA 26.mars (gjøres etter mastermøte)

- Upload TRM sudoku checkpoints to hugging face.✅
- Remove old SHREK checpoints before new getting the new checpoints.
- Upload all the checkpoints for SHREK-large and mini for the sudoku and maze dataset.
- Update dataset HuggingFace: Upload current version form local to hugging face
