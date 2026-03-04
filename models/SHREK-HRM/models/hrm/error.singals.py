"""
error_signals.py  -  SHREK Error Signal Module
================================================

WHAT IS THIS FILE?
------------------
This file answers one question for the model at every reasoning step:
    "How wrong is my current answer?"

We call this the "error signal" - a single number per puzzle that tells
the model roughly how bad its current guess is. 0 means perfect, higher
means more mistakes.

WHY DO WE NEED THIS?
--------------------
The original HRM model reasons by running the same transformer over and
over in a loop, updating its "hidden state" (internal memory) each time.
The problem: the model has NO feedback about whether it is getting closer
or further from the right answer. It is like solving a puzzle blindfolded.

SHREK fixes this by computing an error score after each loop step and
injecting that score back into the model memory. Now the model can
"feel" when it is stuck or going in the wrong direction.

HOW IT WORKS IN PRACTICE:
--------------------------
1. The model outputs "logits" - one big array of numbers per output cell
   (e.g. for Sudoku, 81 cells x 11 possible values = 891 numbers total)
2. We pick the most likely prediction for each cell (called "argmax")
3. We check how many rules that prediction breaks
4. We return a single number: the error score
5. The main model (hrm_act_v1.py) encodes this number and adds it to the
   model memory so future reasoning steps are aware of the current mistakes

ONE FILE, FOUR DATASETS:
-------------------------
We have three error functions - one per task type:
  - Sudoku     -> count rule violations (rows, columns, boxes)
  - Maze       -> count disconnected path fragments
  - ARC-AGI    -> measure output uncertainty (entropy)
                  Works for BOTH ARC-AGI-1 and ARC-AGI-2

TOKEN ENCODINGS (how datasets store values as numbers):
--------------------------------------------------------
Each dataset converts its symbols into integer "token IDs" stored in arrays.
Here is what each token ID means per dataset:

  Sudoku  (seq_len=81,  vocab_size=11):
      0 = PAD      (padding, not used in Sudoku)
      1 = blank    (an empty cell the model must fill in)
      2 = digit 1  (the digit "1")
      3 = digit 2  ...
     10 = digit 9

  Maze  (seq_len=900=30x30,  vocab_size=6):
      0 = PAD      (padding)
      1 = #        (wall - cannot walk through)
      2 = (space)  (open corridor)
      3 = S        (start position)
      4 = G        (goal position)
      5 = o        (part of the solution path)

  ARC-AGI-1 and ARC-AGI-2  (seq_len=900=30x30,  vocab_size=12):
      0 = PAD      (padding around the grid)
      1 = EOS      (end-of-sequence marker, marks grid boundary)
      2 = color 0  (the color "0" / black)
      3 = color 1  (the color "1" / blue)
     ...
     11 = color 9

These encodings come directly from the dataset builder scripts:
  dataset/build_sudoku_dataset.py  ->  arr + 1  (digits 0-9 become tokens 1-10)
  dataset/build_maze_dataset.py    ->  char2id lookup table
  dataset/build_arc_dataset.py     ->  digit + 2  (colors 0-9 become tokens 2-11)
"""

import torch
import torch.nn.functional as F


# =============================================================================
# FUNCTION 1: Sudoku Error
# =============================================================================

def compute_sudoku_error(logits: torch.Tensor) -> torch.Tensor:
    """
    Counts how many Sudoku rules the model current prediction breaks.

    SUDOKU RULES (the constraints we check):
        - Every row must contain each digit 1-9 exactly once (no repeats)
        - Every column must contain each digit 1-9 exactly once
        - Every 3x3 box must contain each digit 1-9 exactly once
        There are 9 rows + 9 columns + 9 boxes = 27 groups total.

    INPUT:
        logits  -  shape (B, 81, 11)
                   B  = number of puzzles in this batch (e.g. 32)
                   81 = number of cells in a 9x9 Sudoku grid
                   11 = for each cell, 11 possible values (token IDs 0-10)
                        The model outputs a "score" for each possible value.
                        Higher score = more confident that cell = that value.

    OUTPUT:
        error  -  shape (B,)  one number per puzzle
                  0.0  = no rule violations at all (perfect prediction)
                  higher = more rule violations
                  Divided by 243 to keep the number manageable (~0 to 8 range)

    STEP-BY-STEP:
    -------------
    Step 1: argmax over vocab dim -> predicted token ID per cell (B, 81)
    Step 2: token_id - 1 -> digit values: blank=0, digit1=1, digit9=9
    Step 3: reshape (B, 81) -> (B, 9, 9)
    Step 4: one-hot encode each cell over 9 digit slots -> (B, 9, 9, 9)
            blank cells are zeroed out so they never count as a digit
    Step 5: sum one-hots along rows, columns, boxes
            if digit appears twice in a row -> sum=2 -> excess = relu(2-1) = 1
    Step 6: sum all excesses = total violations
    Step 7: divide by 243 to normalise
    """

    B = logits.shape[0]

    # Step 1: pick the token with the highest score for each of the 81 cells
    # argmax returns index 0-10 (the token ID with highest logit value)
    pred_tokens = logits.argmax(dim=-1)           # (B, 81)  values 0 to 10

    # Step 2: convert token IDs to digit values
    # token 1 = blank -> 1-1 = 0 (will be ignored)
    # token 2 = digit 1 -> 2-1 = 1
    # token 10 = digit 9 -> 10-1 = 9
    # clamp(min=0) keeps token 0 (PAD) from becoming -1
    digit_vals = (pred_tokens - 1).clamp(min=0)   # (B, 81)  values 0=blank, 1-9=digit

    # Step 3: reshape flat 81 values into a 9x9 grid
    grid = digit_vals.view(B, 9, 9)               # (B, 9, 9)

    # Step 4: one-hot encode
    # (grid - 1).clamp(min=0): digit 1 -> index 0, digit 9 -> index 8
    # blank cells (grid==0): (0-1).clamp=0, but we zero them out with blank_mask
    one_hot = F.one_hot(
        (grid - 1).clamp(min=0),                  # (B, 9, 9) digit indices 0-8
        num_classes=9                              # 9 possible digits
    ).float()                                      # (B, 9, 9, 9)

    # zero out blank cells so they never count as any digit
    blank_mask = (grid == 0).unsqueeze(-1)         # (B, 9, 9, 1)  True = blank
    one_hot = one_hot.masked_fill(blank_mask, 0.0)

    # Step 5 + 6: count violations per row, column, and 3x3 box
    # relu(count - 1) = 0 if digit appears once (correct), 1 if twice, etc.
    violations = torch.zeros(B, device=logits.device, dtype=torch.float32)

    # rows: sum along the column dimension (dim=2), result (B, 9_rows, 9_digits)
    row_counts = one_hot.sum(dim=2)
    violations += F.relu(row_counts - 1).sum(dim=(1, 2))

    # columns: sum along the row dimension (dim=1), result (B, 9_cols, 9_digits)
    col_counts = one_hot.sum(dim=1)
    violations += F.relu(col_counts - 1).sum(dim=(1, 2))

    # 3x3 boxes: view splits the 9x9 grid into 9 boxes of 3x3 cells each
    # (B, 3 box-rows, 3 box-cols, 3 inner-rows, 3 inner-cols, 9 digits)
    # sum over inner rows (dim=3) and inner cols (dim=4)
    boxes = one_hot.view(B, 3, 3, 3, 3, 9)
    box_counts = boxes.sum(dim=(3, 4))             # (B, 3, 3, 9)
    violations += F.relu(box_counts - 1).sum(dim=(1, 2, 3))

    # Step 7: normalise - 243 = 27 groups x 9 digits
    return violations / 243.0                      # (B,)


# =============================================================================
# FUNCTION 2: Maze Error
# =============================================================================

def compute_maze_error(logits: torch.Tensor) -> torch.Tensor:
    """
    Measures how broken the model predicted maze path is.

    WHAT MAKES A VALID MAZE SOLUTION?
        The model must predict a connected chain of path cells ('o' = token 5)
        linking Start ('S' = token 3) to Goal ('G' = token 4).
        Every path cell must touch at least one other path cell (up/down/left/right).
        Isolated path cells with no neighbours = broken fragments = wrong answer.

    INPUT:
        logits  -  shape (B, 900, 6)
                   B   = number of mazes in the batch
                   900 = 30x30 = total cells in the maze grid
                   6   = 6 possible token values per cell (0-5)

    OUTPUT:
        error  -  shape (B,)  one number per maze
                  0.0 = all path cells are connected (possibly correct)
                  1.0 = every path cell is isolated (completely broken)

    STEP-BY-STEP:
    -------------
    Step 1: argmax -> predicted token per cell, reshape to (B, 30, 30)
    Step 2: path mask = True where token is 3 (S), 4 (G), or 5 (o)
    Step 3: shift path mask in all 4 directions to find cells with path neighbours
            has_up[r,c] = True means cell above (r-1, c) is also a path cell
    Step 4: isolated = path cells with no path neighbour in any direction
            error = isolated_count / total_path_count  (range 0 to 1)
    """

    B, HW, V = logits.shape
    H = W = int(HW ** 0.5)    # square grid: 30x30 -> H=W=30

    # Step 1: predicted token per cell, reshaped to 2D grid
    pred = logits.argmax(dim=-1)      # (B, 900)
    grid = pred.view(B, H, W)         # (B, 30, 30)

    # Step 2: path mask - True wherever model predicted S, G, or o
    path = (grid == 5) | (grid == 3) | (grid == 4)    # (B, 30, 30)  bool

    # Step 3: shift the mask in each direction to detect neighbours
    #
    # has_up[b, r, c] = True if cell directly ABOVE (r-1, c) is a path cell
    # Implementation: create a zero grid, then copy path[r-1] into row r
    # Row 0 stays False (no row above it = no neighbour above boundary)

    has_up = torch.zeros_like(path)
    has_up[:, 1:, :] = path[:, :-1, :]      # row 1..H-1 looks at row 0..H-2

    has_down = torch.zeros_like(path)
    has_down[:, :-1, :] = path[:, 1:, :]    # row 0..H-2 looks at row 1..H-1

    has_left = torch.zeros_like(path)
    has_left[:, :, 1:] = path[:, :, :-1]    # col 1..W-1 looks at col 0..W-2

    has_right = torch.zeros_like(path)
    has_right[:, :, :-1] = path[:, :, 1:]   # col 0..W-2 looks at col 1..W-1

    # a cell has a neighbour if ANY of the 4 directions leads to a path cell
    has_neighbour = has_up | has_down | has_left | has_right   # (B, 30, 30)

    # Step 4: isolated = path cell AND has no path neighbour
    # clamp(min=1) avoids division by zero when no path cells were predicted
    isolated   = path & ~has_neighbour                           # (B, 30, 30)
    total_path = path.float().sum(dim=(1, 2)).clamp(min=1.0)     # (B,)
    broken     = isolated.float().sum(dim=(1, 2))                # (B,)

    return broken / total_path    # (B,)  range 0 to 1


# =============================================================================
# FUNCTION 3: ARC Error (works for both ARC-AGI-1 and ARC-AGI-2)
# =============================================================================

def compute_arc_error(logits: torch.Tensor) -> torch.Tensor:
    """
    Measures how uncertain the model is about its ARC prediction.

    WHY UNCERTAINTY INSTEAD OF RULE-CHECKING?
        Sudoku and Maze have fixed rules we can verify (no repeated digits,
        connected path). ARC tasks are completely unique visual puzzles with
        no shared rules - each puzzle tests a different transformation.
        So we cannot check "rules". Instead we measure confidence:
          - A confident model assigns ~100% probability to one token per cell
          - A confused model spreads probability evenly across many tokens

    WHAT IS ENTROPY?
        Entropy measures how "spread out" a probability distribution is.
        Low entropy  = confident  = most probability on one option
        High entropy = uncertain  = probability spread across many options

        Formula: entropy = -sum( p * log(p) ) for each probability p
        The more uniform the probabilities, the higher the entropy value.

    INPUT:
        logits  -  shape (B, 900, 12)
                   B   = number of ARC puzzles in the batch
                   900 = 30x30 = max grid size (smaller grids are zero-padded)
                   12  = 12 tokens: 0=PAD, 1=EOS, 2-11=colors 0-9

    OUTPUT:
        error  -  shape (B,)  one number per puzzle
                  0.0 = completely confident (low entropy = doing well)
                  1.0 = maximally uncertain (uniform distribution = very confused)

    STEP-BY-STEP:
    -------------
    Step 1: softmax converts logits -> probabilities (all 0-1, sum to 1 per cell)
    Step 2: entropy = -sum(p * log(p)) per position -> one value per cell
    Step 3: skip trivial padding positions (PAD probability > 90%)
            padding cells are always easy and dilute the signal
    Step 4: average entropy over the remaining active grid cells
    Step 5: divide by log(vocab_size) to normalise result to 0-1 range
    """

    B, L, V = logits.shape    # B=batch, L=900 positions, V=12 token types

    # Step 1: softmax - turn raw scores into probabilities
    # .float() ensures precision - bfloat16 can lose accuracy in log operations
    probs = F.softmax(logits.float(), dim=-1)     # (B, L, V) all values in [0,1]

    # Step 2: entropy per cell position
    # +1e-8 inside the log prevents log(0) = -infinity
    # sum over vocab dim (dim=-1) collapses 12 token probs into 1 entropy value
    entropy = -(probs * (probs + 1e-8).log()).sum(dim=-1)    # (B, L)

    # Step 3: mask out trivial padding positions
    # probs[:, :, 0] = probability that each position is the PAD token
    # if PAD probability > 90%, the model is very confident it is padding
    # these positions tell us nothing about how well the model understands the task
    pad_prob   = probs[:, :, 0]                              # (B, L)
    active     = pad_prob < 0.9                              # (B, L)  True = real content
    active_cnt = active.float().sum(dim=1).clamp(min=1.0)    # (B,) avoid division by zero

    # Step 4: mean entropy over active positions only
    # multiply by active mask (zeros out padding), then divide by count of active cells
    mean_entropy = (entropy * active.float()).sum(dim=1) / active_cnt    # (B,)

    # Step 5: normalise to [0, 1]
    # log(V) = maximum possible entropy when all V tokens are equally likely
    max_entropy = torch.log(
        torch.tensor(V, dtype=torch.float32, device=logits.device)
    )
    return mean_entropy / max_entropy    # (B,)


# =============================================================================
# DISPATCHER - the only function called from hrm_act_v1.py
# =============================================================================

def get_error_signal(logits: torch.Tensor, task_type: str) -> torch.Tensor:
    """
    Routes to the correct error function based on which dataset we are using.

    INPUTS:
        logits     -  shape (B, seq_len, vocab_size)
                      Raw output from the model lm_head. Any dtype is fine.

        task_type  -  string, one of:
                        "sudoku"  -> compute_sudoku_error()
                        "maze"    -> compute_maze_error()
                        "arc"     -> compute_arc_error()
                      Use "arc" for BOTH ARC-AGI-1 and ARC-AGI-2.

    OUTPUT:
        error  -  shape (B,)  float32, same device as logits
                  0.0 = no error / confident = model doing well
                  1.0 = max error / confused = model needs to correct itself

    HOW TO CALL THIS IN hrm_act_v1.py (Step 3 of TODO.md):
        from models.hrm.error_signals import get_error_signal

        logits    = self.inner.lm_head(z_H)[:, self.puzzle_emb_len:]
        error     = get_error_signal(logits, task_type)        # (B,)
        error_emb = self.error_encoder(error.unsqueeze(-1))    # (B, hidden_size)
        z_H       = z_H + self.alpha * error_emb.unsqueeze(1)  # inject into memory
    """

    logits_f = logits.float()    # cast to float32 for consistent precision

    if task_type == "sudoku":
        return compute_sudoku_error(logits_f)
    elif task_type == "maze":
        return compute_maze_error(logits_f)
    elif task_type == "arc":
        return compute_arc_error(logits_f)
    else:
        raise ValueError(
            f"Unknown task_type '{task_type}'. "
            f"Must be one of: 'sudoku', 'maze', 'arc'. "
            f"Use 'arc' for both ARC-AGI-1 and ARC-AGI-2."
        )
