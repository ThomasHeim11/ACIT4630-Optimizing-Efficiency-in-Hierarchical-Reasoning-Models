import torch

def maze_swap(x, perm: int):
    if perm == 0:
        return x
    
    tmp = x.clone()
    y = torch.where(x == 3, 100, torch.where(x == 4, 101, x))
    return torch.where(y == 100, 4, torch.where(y == 101, 3, y))
