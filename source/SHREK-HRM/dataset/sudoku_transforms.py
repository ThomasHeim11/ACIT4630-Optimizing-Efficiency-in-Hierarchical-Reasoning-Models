import numpy as np

def sudoku_cyclic_shift(x, perm: int):
    if perm == 0:
        return x
    mask = (x >= 2) & (x <= 10)

    idx = x[mask] - 2

    idx_shifted = (idx + perm) % 9

    val_shifted = idx_shifted + 2

    out = x.clone()
    out[mask] = val_shifted.to(out.dtype)
    return out

def add_random_hints(puzzle: np.ndarray, solution: np.ndarray) -> np.ndarray:
    easier_puzzle = puzzle.copy()
    
    # Find positions where puzzle has blanks (value 0)
    blank_positions = np.where(puzzle == 0)
    blank_indices = list(zip(blank_positions[0], blank_positions[1]))
    
    if len(blank_indices) == 0:
        return easier_puzzle  # No blanks to fill
    
    num_hints = np.random.randint(1, len(blank_indices))
    
    # Randomly select positions to fill
    selected_positions = np.random.choice(len(blank_indices), size=num_hints, replace=False)
    
    # Fill selected positions with solution values
    for idx in selected_positions:
        row, col = blank_indices[idx]
        easier_puzzle[row, col] = solution[row, col]
    
    return easier_puzzle