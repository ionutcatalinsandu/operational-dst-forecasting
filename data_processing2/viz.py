import numpy as np
import matplotlib.pyplot as plt
from math import ceil, sqrt

def plot_2d_shape(pairs, ax=None, color='#2E86AB', linewidth=1.5, alpha=0.8):
    """
    Clean minimalist plot of 2D embedding shape.
    
    Parameters:
    - pairs: Array of 2D points (n_points, 2)
    - ax: Matplotlib axis (creates new if None)
    - color: Line color
    - linewidth: Line width
    - alpha: Transparency
    
    Returns:
    - ax: The axis object
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(4, 4))
    
    if len(pairs) > 0:
        # Plot trajectory
        ax.plot(pairs[:, 0], pairs[:, 1], '-', color=color, 
                linewidth=linewidth, alpha=alpha)
        
        # Mark start and end
        ax.plot(pairs[0, 0], pairs[0, 1], 'o', color='green', 
                markersize=6, alpha=0.7, zorder=5)
        ax.plot(pairs[-1, 0], pairs[-1, 1], 'o', color='red', 
                markersize=6, alpha=0.7, zorder=5)
    
    # Minimalist styling
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.2, linewidth=0.5)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(labelsize=8)
    
    return ax


def plot_shape_sequence(pairs_list, ncols=8, figsize=None, titles=None):
    """
    Plot sequence of 2D shapes in a grid.
    
    Parameters:
    - pairs_list: List of 2D arrays, each shape (n_points, 2)
    - ncols: Number of columns in grid
    - figsize: Figure size (auto if None)
    - titles: Optional list of titles for each subplot
    
    Returns:
    - fig, axes: Figure and axes array
    """
    n_shapes = len(pairs_list)
    nrows = (n_shapes + ncols - 1) // ncols
    
    if figsize is None:
        figsize = (ncols * 2, nrows * 2)
    
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False)
    
    # Flatten axes for easier iteration
    axes_flat = axes.flatten()
    
    # Hide all axes initially
    for ax in axes_flat:
        ax.set_visible(False)
    
    # Plot each shape
    for i, pairs in enumerate(pairs_list):
        if i < len(axes_flat):
            ax = axes_flat[i]
            ax.set_visible(True)
            
            if pairs is not None and len(pairs) > 0:
                plot_2d_shape(pairs, ax=ax)
                
                if titles and i < len(titles):
                    ax.set_title(titles[i], fontsize=10, pad=5)
                else:
                    ax.set_title(f"t = {i}", fontsize=10, pad=5)
    
    plt.tight_layout()
    return fig, axes


# Helper function to extract shape snapshots
def get_shape_snapshots(pairs_df, column, time_indices, window_size=50):
    """
    Extract 2D shapes at specific time points.
    
    Parameters:
    - pairs_df: DataFrame from embeddings_to_2d_pairs
    - column: Column name to extract
    - time_indices: List of time indices
    - window_size: Points to include before each time
    
    Returns:
    - List of 2D arrays
    """
    shapes = []
    
    for t in time_indices:
        start = max(0, t - window_size)
        window_pairs = []
        
        for k in range(start, t + 1):
            if k < len(pairs_df) and pairs_df[column].iloc[k] is not None:
                window_pairs.extend(pairs_df[column].iloc[k])
        
        if window_pairs:
            shapes.append(np.array(window_pairs))
        else:
            shapes.append(None)
    
    return shapes


def plot_clean_distributions(df, columns=None, figsize=(15, 10), bins=50):
    """
    Create clean, well-organized histogram plots for dataset distributions.
    
    Parameters:
    - df: DataFrame
    - columns: list of columns to plot (if None, plots all numeric columns)
    - figsize: figure size tuple
    - bins: number of bins for histograms
    """
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()
    
    n_cols = len(columns)
    n_rows = ceil(sqrt(n_cols))
    n_plot_cols = ceil(n_cols / n_rows)
    
    fig, axes = plt.subplots(n_rows, n_plot_cols, figsize=figsize)
    axes = axes.flatten() if n_cols > 1 else [axes]
    
    for i, col in enumerate(columns):
        ax = axes[i]
        
        # Remove NaN values for plotting
        data = df[col].dropna()
        
        # Plot histogram
        ax.hist(data, bins=bins, alpha=0.7, color='steelblue', edgecolor='black', linewidth=0.5)
        
        # Clean formatting
        ax.set_title(f'{col}', fontsize=10, fontweight='bold', pad=10)
        ax.set_xlabel('Value', fontsize=9)
        ax.set_ylabel('Frequency', fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=8)
        
        # Add basic statistics
        mean_val = data.mean()
        std_val = data.std()
        ax.axvline(mean_val, color='red', linestyle='--', alpha=0.8, label=f'Mean: {mean_val:.2f}')
        ax.legend(fontsize=8)
    
    # Hide empty subplots
    for i in range(n_cols, len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    plt.show()