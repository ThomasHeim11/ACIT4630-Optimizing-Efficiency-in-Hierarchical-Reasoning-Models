# SHREK: This file replaces the original three task-specific error functions
# (compute_sudoku_error, compute_maze_error, compute_arc_error) with one
# universal function based on prediction flip rate.
#
# WHY: hardcoding rules per dataset is fragile. Every new dataset needs a new
# function. Flip rate needs no rules — it just compares predictions across time.
#
# HOW IT WORKS:
#   After every reasoning step the model makes a prediction.
#   We compare that prediction to what the model predicted last step.
#   If lots of tokens changed -> model is still figuring it out -> keep thinking.
#   If nothing changed -> model has settled -> check if it looks right.
#
# FIRST STEP BEHAVIOUR:
#   prev_pred is initialized to all zeros (token 0 = PAD) in empty_carry().
#   Real predictions are almost never all-zero.
#   So first step: flip_rate ≈ 1.0 — correct, start with maximum uncertainty.
#
# WORKS FOR ALL DATASETS WITHOUT CHANGES:
#   Sudoku    (seq_len=81,  vocab=11)
#   Maze      (seq_len=900, vocab=6)
#   ARC-AGI-1 (seq_len=900, vocab=12)
#   ARC-AGI-2 (seq_len=900, vocab=12)
#   Any future dataset — automatic.

from typing import Tuple
import torch


# SHREK: single universal function — replaces three task-specific functions
def get_error_signal(
    logits: torch.Tensor,
    prev_pred: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    SHREK universal error signal: prediction flip rate.

    Counts what fraction of output tokens changed compared to last step.

    INPUTS
        logits    : (B, seq_len, vocab_size)  raw scores from lm_head, any dtype
        prev_pred : (B, seq_len) int32        argmax from previous ACT step
                                              all zeros = fresh start after reset

    OUTPUTS
        flip_rate    : (B,) float32  fraction of positions that changed (0.0 to 1.0)
        current_pred : (B, seq_len) int32  this step's argmax — store in carry
    """

    # SHREK: pick most likely token at each position
    current_pred = logits.argmax(dim=-1).to(torch.int32)         # (B, seq_len)

    # SHREK: fraction of positions where prediction changed vs last step
    # True where changed, False where same -> .float() -> 1.0 or 0.0
    # .mean(dim=1) averages over seq_len -> one number per puzzle
    flip_rate = (current_pred != prev_pred).float().mean(dim=1)  # (B,)

    return flip_rate, current_pred
