import shap
import matplotlib.pyplot as plt
import numpy as np
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


def create_shap_plot(model, dmatrix, save_path=None,
                     max_features=15, figsize=(8, 6),
                     sample_size=1000, random_state=123):
    """
    Generate publication-quality SHAP summary plot.

    Parameters:
    -----------
    model : trained XGBoost Booster
    dmatrix : xgboost.core.DMatrix, feature data (typically test set)
    save_path : str, optional path to save the figure as PNG
    max_features : int, number of top features to display
    figsize : tuple, figure dimensions in inches
    sample_size : int, number of samples to use for SHAP computation
    random_state : int, for reproducible sampling

    Returns:
    --------
    fig : matplotlib Figure
    shap_values : np.ndarray, computed SHAP values
    """
    rcParams.update(PUBLICATION_STYLE)

    # Sample rows from DMatrix for computational efficiency
    n_total = dmatrix.num_row()
    rng = np.random.default_rng(random_state)
    indices = rng.choice(n_total, size=min(sample_size, n_total), replace=False)
    indices_sorted = sorted(indices.tolist())
    sample = dmatrix.slice(indices_sorted)

    # Retrieve feature names and fix known typos
    feature_names = [
        name.replace("elecrtric", "electric")
        for name in dmatrix.feature_names
    ]
    sample.feature_names = feature_names

    # Compute SHAP values
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(sample)

    # Extract feature matrix as numpy array for summary_plot
    feature_matrix = sample.get_data().toarray()

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    shap.summary_plot(
        shap_values,
        feature_matrix,
        feature_names=feature_names,
        max_display=max_features,
        show=False,
        plot_size=None,
    )

    ax = plt.gca()
    ax.set_xlabel('SHAP Value (nT)', labelpad=8)
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    ax.set_axisbelow(True)

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

    return fig, shap_values