import numpy as np
import matplotlib.pyplot as plt
import xgboost as xgb
from matplotlib import rcParams

PUBLICATION_STYLE = {
    'font.size': 11,
    'font.family': 'serif',
    'font.serif': ['Times New Roman'],
    'axes.linewidth': 0.8,
    'axes.labelsize': 11,
    'axes.labelweight': 'bold',
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.facecolor': 'white',
    'savefig.dpi': 300,
    'figure.dpi': 100,
}


def _clean(name: str) -> str:
    """Display-only: fix typo and replace underscores."""
    return name.replace("elecrtric", "electric").replace("_", " ")


def create_ice_plot(
    model,
    dmatrix,
    vary_feature: str,
    dst_mask: np.ndarray | None = None,
    grid_resolution: int = 60,
    sample_size: int = 300,
    random_state: int = 123,
    is_scaled: bool = False,
    figsize: tuple = (8, 5),
    save_path: str | None = None,
):
    """
    ICE plot: predicted Dst vs a single feature.

    Individual sample curves are drawn transparently; the bold line is the
    mean ICE curve across all selected samples.

    Parameters
    ----------
    model : xgboost.Booster
    dmatrix : xgboost.DMatrix
    vary_feature : str
        Feature to sweep. Must match a name in dmatrix.feature_names.
    dst_mask : np.ndarray of bool, shape (N,), optional
        Row-aligned boolean mask to filter samples before plotting.
        E.g. dst_values <= -50 for storm periods only.
    mask_label : str, optional
        Short description of the mask shown as a corner annotation.
    grid_resolution : int
        Number of grid points for the ICE sweep (default: 60).
    sample_size : int
        Number of samples drawn uniformly from the masked subset (default: 300).
    random_state : int
    is_scaled : bool
        If True, appends "(scaled)" to the x-axis label.
    figsize : tuple
    save_path : str, optional

    Returns
    -------
    fig : matplotlib.Figure
    curves : np.ndarray, shape (sample_size, grid_resolution)
    grid : np.ndarray, shape (grid_resolution,)
    """
    rcParams.update(PUBLICATION_STYLE)

    # ------------------------------------------------------------------ #
    # 1. Feature index
    # ------------------------------------------------------------------ #
    original_names = list(dmatrix.feature_names)
    if vary_feature not in original_names:
        raise ValueError(f"'{vary_feature}' not found in feature names.")
    vary_idx = original_names.index(vary_feature)

    # ------------------------------------------------------------------ #
    # 2. Apply mask and sample uniformly
    # ------------------------------------------------------------------ #
    X_full = dmatrix.get_data().toarray()

    if dst_mask is not None:
        X_full = X_full[np.asarray(dst_mask, dtype=bool)]

    n_available = X_full.shape[0]
    if n_available == 0:
        raise ValueError("No samples remain after applying dst_mask.")

    rng = np.random.default_rng(random_state)
    n = min(sample_size, n_available)
    idx = rng.choice(n_available, size=n, replace=False)
    X = X_full[idx]

    print(f"ICE: {n} samples ({n_available} available after mask).")

    # ------------------------------------------------------------------ #
    # 3. Grid over vary_feature (1st–99th percentile to avoid outliers)
    # ------------------------------------------------------------------ #
    grid = np.linspace(
        np.percentile(X[:, vary_idx], 1),
        np.percentile(X[:, vary_idx], 99),
        grid_resolution,
    )

    # ------------------------------------------------------------------ #
    # 4. Compute ICE curves
    # ------------------------------------------------------------------ #
    curves = np.zeros((n, grid_resolution))
    for k, z in enumerate(grid):
        X_mod = X.copy()
        X_mod[:, vary_idx] = z
        dm_mod = xgb.DMatrix(X_mod, feature_names=original_names)
        curves[:, k] = model.predict(dm_mod)

    # ------------------------------------------------------------------ #
    # 5. Plot
    # ------------------------------------------------------------------ #
    fig, ax = plt.subplots(figsize=figsize)

    # Individual ICE lines
    for curve in curves:
        ax.plot(grid, curve, color='#4393c3', alpha=0.15, linewidth=0.6, zorder=2)

    # IQR band
    ax.fill_between(grid,
                    np.percentile(curves, 25, axis=0),
                    np.percentile(curves, 75, axis=0),
                    color='#4393c3', alpha=0.3, zorder=4, label='IQR')

    # Mean curve
    ax.plot(grid, curves.mean(axis=0),
            color='#b2182b', linewidth=2.2, zorder=5, label='Mean ICE')

    # Storm reference lines
    ax.axhline(y=-50,  color='orange', linestyle=':', linewidth=1.1, alpha=0.8, zorder=1)
    ax.axhline(y=-100, color='red',    linestyle=':', linewidth=1.1, alpha=0.8, zorder=1)
    ax.text(grid[-1], -50,  ' $-$50 nT',  va='bottom', ha='right', fontsize=8, color='darkorange')
    ax.text(grid[-1], -100, ' $-$100 nT', va='bottom', ha='right', fontsize=8, color='red')

    # Axes labels
    x_label = _clean(vary_feature)
    if is_scaled:
        x_label += ' (scaled)'
    ax.set_xlabel(x_label)
    ax.set_ylabel('Predicted Dst, nT')

    ax.legend(loc='best', frameon=True, framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)

    # # Corner annotation
    # annotation = mask_label or (f'$n = {n:,}$')
    # ax.text(0.02, 0.03, annotation,
    #         transform=ax.transAxes, fontsize=8, style='italic', va='bottom',
    #         bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight',
                    facecolor='white', edgecolor='none', format='png')
        print(f"Saved: {save_path}")

    return fig, curves, grid