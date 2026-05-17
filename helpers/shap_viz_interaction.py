import shap
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import rcParams

PUBLICATION_STYLE = {
    'font.size': 11,
    'font.family': 'serif',
    'font.serif': ['Times New Roman'],
    'axes.linewidth': 0.8,
    'axes.labelsize': 11,
    'axes.labelweight': 'bold',
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'figure.facecolor': 'white',
    'savefig.dpi': 300,
    'figure.dpi': 100,
}


def create_shap_interaction_plot(model, dmatrix, save_path=None,
                                  top_k=10, figsize=(10, 8),
                                  sample_size=1000, random_state=123):
    """
    Generate publication-quality SHAP interaction heatmap.

    Parameters:
    -----------
    model : trained XGBoost Booster
    dmatrix : xgboost.core.DMatrix
    save_path : str, optional path to save the figure as PNG
    top_k : int, number of top interacting features to display
    figsize : tuple, figure dimensions in inches
    sample_size : int, number of samples for SHAP computation
    random_state : int, for reproducible sampling

    Returns:
    --------
    fig : matplotlib Figure
    shap_interaction_values : np.ndarray
    top_features : list of str, feature names in display order
    """
    rcParams.update(PUBLICATION_STYLE)

    # Sample from DMatrix
    n_total = dmatrix.num_row()
    rng = np.random.default_rng(random_state)
    indices = sorted(rng.choice(n_total, size=min(sample_size, n_total), replace=False).tolist())
    sample = dmatrix.slice(indices)

    # Fix known typos in feature names
    feature_names = [
        name.replace("elecrtric", "electric")
        for name in dmatrix.feature_names
    ]
    sample.feature_names = feature_names

    # Compute SHAP interaction values
    explainer = shap.TreeExplainer(model)
    shap_interaction_values = explainer.shap_interaction_values(sample)
    print(f"Interaction values shape: {shap_interaction_values.shape}")

    # Mean absolute interaction matrix
    mean_abs = np.abs(shap_interaction_values).mean(axis=0)

    # Rank features by total off-diagonal interaction strength
    interaction_strength = mean_abs.sum(axis=0) - np.diag(mean_abs)
    top_idx = np.argsort(interaction_strength)[::-1][:top_k]
    top_features = [feature_names[i] for i in top_idx]

    # Subset and mask diagonal
    top_matrix = mean_abs[np.ix_(top_idx, top_idx)]
    masked = top_matrix.copy()
    np.fill_diagonal(masked, np.nan)

    # Clean up feature names for display (replace underscores with spaces)
    display_names = [f.replace('_', ' ') for f in top_features]

    # Plot
    fig, ax = plt.subplots(figsize=figsize)

    sns.heatmap(
        masked,
        xticklabels=display_names,
        yticklabels=display_names,
        annot=True,
        fmt='.3f',
        annot_kws={'size': 8},
        cmap='RdBu_r',
        center=0,
        square=True,
        linewidths=0.4,
        linecolor='white',
        ax=ax,
        cbar_kws={
            'label': 'Mean |SHAP Interaction Value| (nT)',
            'shrink': 0.8,
        },
    )

    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)

    # Style the colorbar label
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=9)
    cbar.set_label('Mean |SHAP Interaction Value| (nT)', size=10)

    plt.tight_layout()

    if save_path:
        plt.savefig(
            save_path,
            dpi=300,
            bbox_inches='tight',
            facecolor='white',
            edgecolor='none',
            format='png',
        )
        print(f"Saved: {save_path}")

    return fig, shap_interaction_values, top_features