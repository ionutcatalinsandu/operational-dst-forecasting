import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Tuple
from scipy import stats

def compare_distributions(arr1: np.ndarray, 
                         arr2: np.ndarray, 
                         labels: Tuple[str, str] = ('Original', 'Transformed'),
                         bins: int = 50,
                         figsize: Tuple[int, int] = (12, 5),
                         color1: str = 'blue',
                         color2: str = 'green',
                         alpha: float = 0.7) -> None:
    """
    Display two histograms side by side for distribution comparison.
    
    Parameters:
    -----------
    arr1, arr2 : array-like
        Arrays to compare
    labels : tuple of str
        Labels for the two distributions
    bins : int
        Number of bins for histograms
    figsize : tuple
        Figure size (width, height)
    color1, color2 : str
        Colors for the histograms
    alpha : float
        Transparency level
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # First histogram
    ax1.hist(arr1, bins=bins, color=color1, alpha=alpha, edgecolor='black')
    ax1.set_title(f'{labels[0]} Distribution')
    ax1.set_xlabel('Value')
    ax1.set_ylabel('Frequency')
    ax1.grid(True, alpha=0.3)
    
    # Add statistics
    ax1.axvline(np.mean(arr1), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(arr1):.2f}')
    ax1.axvline(np.median(arr1), color='orange', linestyle='--', linewidth=2, label=f'Median: {np.median(arr1):.2f}')
    ax1.legend()
    
    # Second histogram
    ax2.hist(arr2, bins=bins, color=color2, alpha=alpha, edgecolor='black')
    ax2.set_title(f'{labels[1]} Distribution')
    ax2.set_xlabel('Value')
    ax2.set_ylabel('Frequency')
    ax2.grid(True, alpha=0.3)
    
    # Add statistics
    ax2.axvline(np.mean(arr2), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(arr2):.2f}')
    ax2.axvline(np.median(arr2), color='orange', linestyle='--', linewidth=2, label=f'Median: {np.median(arr2):.2f}')
    ax2.legend()
    
    # Add text with key statistics
    textstr1 = f'Skew: {stats.skew(arr1):.2f}\nKurtosis: {stats.kurtosis(arr1):.2f}'
    textstr2 = f'Skew: {stats.skew(arr2):.2f}\nKurtosis: {stats.kurtosis(arr2):.2f}'
    
    ax1.text(0.05, 0.95, textstr1, transform=ax1.transAxes, fontsize=10,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    ax2.text(0.05, 0.95, textstr2, transform=ax2.transAxes, fontsize=10,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.show()


# Quick version without statistics
def quick_compare(arr1: np.ndarray, 
                  arr2: np.ndarray,
                  labels: Tuple[str, str] = ('Before', 'After'),
                  bins: int = 50) -> None:
    """Quick histogram comparison without extra statistics"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    
    ax1.hist(arr1, bins=bins, alpha=0.7, color='steelblue')
    ax1.set_title(labels[0])
    ax1.set_ylabel('Count')
    
    ax2.hist(arr2, bins=bins, alpha=0.7, color='darkorange')
    ax2.set_title(labels[1])
    
    plt.tight_layout()
    plt.show()

