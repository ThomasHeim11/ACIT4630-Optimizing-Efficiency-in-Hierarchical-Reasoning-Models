import numpy as np
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA

def plot_trajectory(tensor_list, arrow_scale=100.0, linewidth=2.5,
                           colormap='plasma', figsize=(10, 8)):   
    # Validate input
    if not isinstance(tensor_list, list) or len(tensor_list) < 2:
        raise ValueError("Input must be a list with at least 2 tensors")
    
    n_tensors = len(tensor_list)
    
    # Flatten tensors
    flattened_vectors = [tensor.cpu().float().reshape(1, -1) for tensor in tensor_list]
    data_matrix = np.vstack(flattened_vectors)
    
    # Apply PCA
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(data_matrix)
    
    # Create plot
    fig, ax = plt.subplots(figsize=figsize)
    
    x_coords = pca_result[:, 0]
    y_coords = pca_result[:, 1]
    
    # Color gradient
    cmap = plt.cm.get_cmap(colormap)
    colors = cmap(np.linspace(0, 1, n_tensors - 1))
    
    # Draw arrows only
    for i in range(n_tensors - 1):
        start_x, start_y = x_coords[i], y_coords[i]
        end_x, end_y = x_coords[i + 1], y_coords[i + 1]
        
        dx = end_x - start_x
        dy = end_y - start_y
        
        ax.arrow(
            start_x, start_y, 
            dx, dy,
            head_width=0.03 * arrow_scale,
            head_length=0.05 * arrow_scale,
            fc=colors[i],
            ec=colors[i],
            linewidth=linewidth,
            length_includes_head=True,
            alpha=0.8,
            shape='full'
        )
    
    # Set equal aspect ratio
    ax.set_aspect('equal', adjustable='box')
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    mid_x = (xlim[0] + xlim[1]) / 2
    mid_y = (ylim[0] + ylim[1]) / 2
    half  = max(xlim[1] - xlim[0], ylim[1] - ylim[0]) / 2
    ax.set_xlim(mid_x - half, mid_x + half)
    ax.set_ylim(mid_y - half, mid_y + half)
    
    # Simple labels
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.set_title('Reasoning Trajectory')
    ax.grid(True, alpha=0.2)
    
    sm = plt.cm.ScalarMappable(cmap=cmap,
                               norm=plt.Normalize(vmin=1, vmax=n_tensors-1))
    sm.set_array([]) 
    
    cbar = fig.colorbar(sm, ax=ax, shrink=0.8, aspect=30)
    
    cbar.set_label('Reasoning Step', fontsize=18) 
    cbar.ax.tick_params(labelsize=16)
    cbar.set_ticks(np.linspace(1, n_tensors-1, min(5, n_tensors-1)))
    cbar.set_ticklabels([f'{int(t)}' for t in cbar.get_ticks()])
    
    plt.tight_layout()

