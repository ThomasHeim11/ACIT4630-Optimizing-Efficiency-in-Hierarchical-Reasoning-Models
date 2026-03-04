import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize, ListedColormap
from scipy.interpolate import griddata

def forward_z(model, z_H): # z_H: (seq_len, hidden_dim)
    logits = model.model.inner.lm_head(z_H.to("cuda"))[1:] # (81, 11) for sudoku
    return logits

def differentiable_conflict_loss(board): # Heuristic error metric in the paper
    loss = 0.0
    
    board_one_hot = torch.zeros(9, 9, 9)
    for i in range(9):
        for j in range(9):
            num = int(board[i, j].item()) - 1
            if num >= 0:
                board_one_hot[i, j, num] = 1.0
    
    for i in range(9):
        row_sum = torch.sum(board_one_hot[i], dim=0)
        row_excess = torch.relu(row_sum - 1.0)
        loss += torch.sum(row_excess)
    
    for j in range(9):
        col_sum = torch.sum(board_one_hot[:, j], dim=0)
        col_excess = torch.relu(col_sum - 1.0)
        loss += torch.sum(col_excess)
    
    for box_i in range(3):
        for box_j in range(3):
            box_sum = torch.sum(board_one_hot[3*box_i:3*box_i+3, 3*box_j:3*box_j+3], dim=(0, 1))
            box_excess = torch.relu(box_sum - 1.0)
            loss += torch.sum(box_excess)
    
    return loss

def plot_landscape_heatmap(data_dict, z_proj_list, dests=None, density=0.05):
    if not data_dict:
        raise ValueError("data_dict is empty.")

    all_i, all_j = zip(*data_dict.keys())
    min_i, max_i = min(all_i), max(all_i)
    min_j, max_j = min(all_j), max(all_j)

    ni = max_i - min_i + 1     
    nj = max_j - min_j + 1     

    grid = np.full((nj, ni), np.nan, dtype=float)
    for (i, j), v in data_dict.items():
        grid[j - min_j, i - min_i] = float(v)

    
    main_grid = np.ma.masked_where(grid == -1, grid)
    mask_grid = np.ma.masked_where(grid != -1, np.ones_like(grid))

    fig, ax = plt.subplots(figsize=(6, 6))

    extent = (min_i - 0.5, max_i + 0.5,
              min_j - 0.5, max_j + 0.5)

    ax.imshow(np.ones_like(grid), cmap=ListedColormap(['#cccccc']),
          origin='lower', extent=extent, vmin=0, vmax=1, zorder=0)

    im1 = ax.imshow(main_grid, cmap='viridis', origin='lower',
                    extent=extent)
    cb1 = plt.colorbar(im1, ax=ax, fraction=0.046, pad=0.04)
    cb1.set_label('Error Value')

    ax.imshow(mask_grid, cmap=ListedColormap(['none', "#B0B0B0"]),
              origin='lower', extent=extent, vmin=0, vmax=1)

    if z_proj_list:                   
        z_arr = np.asarray(z_proj_list) 
        ax.plot(z_arr[:, 0], z_arr[:, 1],
                color='red', lw=2, marker='o', markersize=4)

    if dests:
        valid = {(i, j): dests[(i, j)]
                 for (i, j) in dests.keys()
                 if data_dict.get((i, j), -1) != -1}
        if not valid:
            return

        rng = np.random.default_rng()
        
        keep = rng.random(len(valid)) < density

        start = np.array(list(valid.keys()))[keep]
        end   = np.array(list(valid.values()))[keep]

        dx_dy = end - start

        colors = np.where(end[:, 0] < 0, 'cyan', 'cyan')

        ax.quiver(start[:, 0], start[:, 1],
                dx_dy[:, 0], dx_dy[:, 1],
                angles='xy', scale_units='xy', scale=1,
                color=colors, width=0.005)

    ax.set(xlabel='PC1', ylabel='PC2')
    ax.set_aspect('equal')
    plt.tight_layout()

def plot_landscape_3d(data_dict, elev, azim):
    keys = np.array(list(data_dict.keys()))
    values = np.array(list(data_dict.values()))
    
    all_i, all_j = keys[:, 0], keys[:, 1]
    min_i, max_i = int(min(all_i)), int(max(all_i))
    min_j, max_j = int(min(all_j)), int(max(all_j))

    i_range = np.arange(min_i, max_i + 1)
    j_range = np.arange(min_j, max_j + 1)
    I, J = np.meshgrid(i_range, j_range)

    grid_z = griddata(keys, values, (I, J), method='linear')
    grid_z_main = np.where(grid_z == -1, np.nan, grid_z)
    
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')

    vmin = np.nanmin(grid_z_main)
    vmax = np.nanmax(grid_z_main)
    norm = Normalize(vmin=vmin, vmax=vmax)
    
    surf = ax.plot_surface(I, J, grid_z_main, cmap='plasma',
                           norm=norm, rstride=1, cstride=1, 
                           antialiased=True, shade=True, zorder=1, alpha=0.9)

    mappable = ScalarMappable(cmap='plasma', norm=norm)
    cb = fig.colorbar(mappable, ax=ax, shrink=0.6, aspect=20, pad=0.1)
    cb.set_label('Error Value')

    z_min_plane = vmin if not np.isnan(vmin) else 0
    mask_indices = np.where(grid_z == -1)
    if mask_indices[0].size > 0:
        ax.scatter(I[mask_indices], J[mask_indices], z_min_plane,
                   c='red', marker='x', s=20, label='Value = -1', zorder=2)

    if ax.get_legend_handles_labels()[0]:
        ax.legend(loc='upper left')

    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.set_zlabel('Clipped Error Value')
    ax.set_title('Error Value Landscape on Principle Plane')
    ax.view_init(elev, azim)

    z_min_overall = min(np.nanmin(grid_z_main), 0)
    z_max_overall = max(np.nanmax(grid_z_main), 50)
    
    ax.set_xlim(min_i, max_i)
    ax.set_ylim(min_j, max_j)
    ax.set_zlim(z_min_overall - 1, z_max_overall + 1)
    
    plt.tight_layout()