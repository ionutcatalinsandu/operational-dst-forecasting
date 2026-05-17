# Operational Dst Forecasting

**Authors:** Ionut-Catalin Sandu, Diana Besliu-Ionescu

The main notebook is `variant_xgboost_new_features.ipynb`. It contains a number of sections for each important step: data processing, training, evaluation, etc. Run each cell individually and observe outputs. 

# Setup

## Installing UV

See [UV documentation](https://docs.astral.sh/uv/getting-started/installation/#__tabbed_1_1) for more details.

**Windows**
```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

**Linux/macOS**
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

## Cloning the Repository

```bash
git clone https://github.com/ionutcatalinsandu/operational-dst-forecasting.git
cd operational-dst-forecasting
```

## Installing Dependencies

```bash
uv sync
```

## Activating the Environment

```bash
uv activate
```

## Downloading Data

Before running the project, you need to download two data files and place them in the `solar_cycles/` folder:

1. Download [File 1](https://drive.google.com/uc?export=download&id=1Ee62fuHekp_P8eZe8ldA3BWJ1XARaepZ)
2. Download [File 2](https://drive.google.com/uc?export=download&id=196jEUKVzn5oOipS4QbJQfIcvTkyLVwrC)

After downloading, move both files to the `solar_cycles/` directory within the project:

```bash
mv <file1> solar_cycles/
mv <file2> solar_cycles/
```

## Highlights

This repository implements a machine learning pipeline for delta-hour ahead Dst-index forecasting using solar wind and magnetospheric data. Key features and design choices:

### Features

- **Lagged Features:** 4, 8, 12, and 24-hour lagged values for all inputs to capture temporal dependencies.
- **Energy Loading State:** Quantifies energy injection into the magnetosphere via solar wind-magnetosphere coupling metrics (Alfvén speed, flow pressure, plasma beta).
- **Coupling Efficiency:** Measures how effectively solar wind energy couples to the magnetosphere (derived from clock angle and solar wind parameters).
- **Magnetotail Loading State:** Captures tail lobe pressure and cross-tail current changes indicating energy storage.
- **Distribution Transformations:** Applied via signed log and sinh-arcsinh to normalize skewed features (magnetic field components, plasma temperature) for better model convergence.

### Training Strategy

- **Sample Weighting:** Storms weighted 0.7, non-storms 0.3 to balance extreme event detection with overall accuracy.
- **Solar Cycle Split:** Train/val/test separation by solar cycle (not time) to avoid leakage from repeating patterns.
- **Feature Completeness Filter:** Retains only rows with ≥50% feature availability, balancing data quality with sample size.

### Model & Evaluation

- **XGBoost Regression:** MAE loss with early stopping; 177 features total.
- **Metrics:** F1 score for storm detection (threshold: -50 nT), MAE, R^2, and confusion matrix analysis.
- **Explainability:** SHAP values, feature interactions, and ICE plots to interpret model decisions.

## Project Structure

| Path | Purpose |
|------|---------|
| `data_processing/` | Raw data reading and initial transformations. See `data.py` for I/O. |
| `data_processing2/` | Feature engineering and scaled transformations. Key file: `features_scaled.py` computes all model features with distribution transformations. |
| `helpers/` | Model utilities. Notable: `metrics.py` for forecast evaluation, `shap_viz.py` for feature explainability. |
| `solar_cycles/` | Input data folder. Download the two files here. |
| `graphs_for_paper*/` | Output visualizations from notebooks. |

**Key Feature Files:**
- `data_processing2/constants.py` — Column names and thresholds (e.g., storm threshold at -50 nT)
- `data_processing2/features_scaled.py` — Computes lagged features and applies transformations (sigmoid arcsine, signed log)
- `helpers/metrics.py` — Storm detection metrics (F1, precision, recall)

## Citation

Please cite this work if you use it. 

**Authors:** Ionut-Catalin Sandu, Diana Besliu-Ionescu