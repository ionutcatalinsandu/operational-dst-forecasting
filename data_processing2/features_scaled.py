import pandas as pd
import numpy as np

from typing import List, Dict, Tuple
from sklearn.base import BaseEstimator

from data_processing2.constants import *
from data_processing2.embeddings import add_takens_embeddings, add_2d_pairs, add_trajectory_features, add_lagged_history, add_signal_features
from data_processing2.magneto import add_energy_loading_state, add_magnetotail_loading_state, add_coupling_efficiency_state, add_recovery_state
from data_processing.scalers import apply_transformation_pipeline, apply_transformation_pipeline_raw, apply_transformation
from data_processing.features import _add_lag_based_feature, _add_rolling_features, _add_convolution_features, add_convolution_feature

def compute_features(
        dataset: pd.DataFrame, 
        column_transforms: Dict[str, List[BaseEstimator]],
        target_transforms: List[BaseEstimator],
        fit_transform: bool = True
    ) -> Tuple[pd.DataFrame, List[BaseEstimator], Dict[str, List[BaseEstimator]]]:

    # Add magnetospheric features first (they use raw data)
    fit_column_transforms = {}
    dataset, _fit_trans = add_magneto_sphere_features(dataset, column_transforms)
    fit_column_transforms.update(_fit_trans)
    # Transform target
    if target_transforms:
        dataset, fit_target_transforms = apply_transformation_pipeline(dataset, DST_COLUMN, target_transforms, fit_transform)
    else:
        fit_target_transforms = target_transforms

    # Add all other features (solar, magnetic, others) before scaling
    dataset, _fit_trans = add_all_domain_features(dataset, column_transforms, fit_transform)
    fit_column_transforms.update(_fit_trans)

    return dataset, fit_target_transforms, fit_column_transforms


def add_magneto_sphere_features(
        data: pd.DataFrame, 
        column_transforms: Dict[str, List[BaseEstimator]],
        fit_transform: bool = True,
    ) -> pd.DataFrame:

    to_drop_after_magnets = [ 
        VECTOR_B_MAG_COL,
        B_LAT_ANGLE_COL,
        B_LONG_ANGLE_COL,
        BX_COL,
        MAGNETOSONIC_MACH_COL,
        AE_COLUMN,
    ]
    
    # print(">> Adding magnetosphere features...")
    dataset = data.copy()
    fit_column_transforms = {}
    # Recovery states (multiple timescales)
    storm_threshold_short = STORM_THRESHOLD     # -50 
    storm_threshold_medium = STORM_THRESHOLD*2  # -100
    storm_threshold_long = STORM_THRESHOLD*3    # -150
    
    dataset, short_recovery_col = add_recovery_state(dataset, lookback_hours=8, storm_threshold=storm_threshold_short)
    dataset, medium_recovery_col = add_recovery_state(dataset, lookback_hours=24, storm_threshold=storm_threshold_medium)
    dataset, long_recovery_col = add_recovery_state(dataset, lookback_hours=48, storm_threshold=storm_threshold_long)

    dataset, _ = _add_lag_based_feature(dataset, short_recovery_col, [0, 4, 8]) # 4, 8 previously 
    dataset, _ = _add_lag_based_feature(dataset, medium_recovery_col, [0, 4, 8])
    dataset, _ = _add_lag_based_feature(dataset, long_recovery_col, [0, 4, 8])
    # print(" >>>> Done adding Recovery State!")
   
    # Energy loading state
    dataset = add_energy_loading_state(dataset)
    # print(" >>>> Done adding Energy Loading State!")

    # Magnetotail loading state
    dataset = add_magnetotail_loading_state(dataset)
    # print(" >>>> Done adding Magnetotail Loading State!")

    # Coupling efficiency state
    dataset = add_coupling_efficiency_state(dataset)
    # print(" >>>> Done adding Coupling Efficiency State!")

    # Apply transformations to all matching columns in the DataFrame
    dataset, fit_column_transforms = apply_transforms_to_derived_columns(dataset, column_transforms, fit_transform)

    dataset, _ = _add_lag_based_feature(dataset, ENERGY_LOADING_STATE_COL, [0, 4, 8]) # 4, 8 previously
    dataset, _ = _add_lag_based_feature(dataset, MAGNETOTAIL_LOADING_STATE_COL, [0, 4, 8])
    dataset, _ = _add_lag_based_feature(dataset, COUPLING_EFFICIENCY_STATE, [0, 4, 8])

    dataset.drop(columns=to_drop_after_magnets, inplace=True)
    # print("Done!")
    
    return dataset, fit_column_transforms


def add_all_domain_features(
        dataset: pd.DataFrame, 
        column_transforms: Dict[str, List[BaseEstimator]],
        fit_transform: bool = True,
    ) -> Tuple[pd.DataFrame, Dict[str, List[BaseEstimator]]]:
    """
    Add features from all domains (solar, magnetic, others) following engineer-then-scale pattern.
    """
    print("--- Working on domain features..")
    fit_column_transforms = {}
    
    # Solar wind features
    dataset, solar_transforms = add_standard_solar(dataset, column_transforms, fit_transform)
    fit_column_transforms.update(solar_transforms)
    
    # Magnetic field features
    dataset, magnetic_transforms = add_standard_magnetic(dataset, column_transforms, fit_transform)
    fit_column_transforms.update(magnetic_transforms)
    
    # Other features (pressure, electric field, etc.)
    dataset, other_transforms = add_standard_others(dataset, column_transforms, fit_transform)
    fit_column_transforms.update(other_transforms)
    print(" > Done!")
    
    return dataset, fit_column_transforms


def add_standard_solar(
        dataset: pd.DataFrame, 
        column_transforms: Dict[str, List[BaseEstimator]],
        fit_transform: bool = True,
    ) -> Tuple[pd.DataFrame, Dict[str, List[BaseEstimator]]]:
    """
    Solar wind features: rolling stats → lag rolling stats → Takens/convolution/history → scale originals → lag scaled originals.
    """
    
    # print(">> Adding solar features...")
    fit_col_transforms = {}

    # Step 1: Compute rolling features on unscaled data
    dataset, rolling_plasma_cols = _add_rolling_features(dataset, SW_SPEED_COL, window=24, overlap=0.0)
    dataset, rolling_temp_cols = _add_rolling_features(dataset, SW_TEMP_COL, window=24, overlap=0.0)
    dataset, rolling_density_cols = _add_rolling_features(dataset, SW_DENSITY_COL, window=24, overlap=0.0)
    dataset, rolling_beta_cols = _add_rolling_features(dataset, PLASMA_BETA_COL, window=24, overlap=0.0)
    # print(" >>>> Done adding solar rolling features!")

    # Step 2: Lag the rolling features
    for _col in rolling_plasma_cols + rolling_temp_cols + rolling_density_cols + rolling_beta_cols:
        dataset, _ = _add_lag_based_feature(dataset, _col, [0, 8, 12]) # 8, 12 previously
    # print(" >>>> Done lagging solar rolling features!")

    # Step 3: Takens/convolution/history on unscaled originals
    solar_columns = [SW_SPEED_COL, SW_TEMP_COL, SW_DENSITY_COL, PLASMA_BETA_COL]
    solar_ms = [5, 6, 5, 6]
    solar_taus = [1, 3, 2, 2]

    dataset = run_takens_pipeline(dataset, solar_columns, solar_ms, solar_taus)
    # dataset = run_history_pipeline(dataset, solar_columns, n_lags=10)
    # Convolution on speed/density/temp
    dataset = add_solar_convolutions(dataset)
    # print(" >>>> Done adding solar Takens/signal/convolution features!")
    # print(" >>>> Done scaling solar features!")
    dataset, fit_col_transforms = apply_transforms_to_derived_columns(dataset, column_transforms, fit_transform)
    
    # Step 5: Lag the scaled originals
    dataset, _ = _add_lag_based_feature(dataset, PLASMA_BETA_COL, [0, 8, 12]) # 8, 12 previously
    dataset, _ = _add_lag_based_feature(dataset, SW_DENSITY_COL, [0, 8, 12])
    dataset, _ = _add_lag_based_feature(dataset, SW_TEMP_COL, [0, 8, 12])
    dataset, _ = _add_lag_based_feature(dataset, SW_SPEED_COL, [0, 8, 12])

    # print(" >>>> Done lagging solar scaled features!")
    

    return dataset, fit_col_transforms


def add_standard_magnetic(
        dataset: pd.DataFrame,
        column_transforms: Dict[str, List[BaseEstimator]],
        fit_transform: bool = True,
    ) -> Tuple[pd.DataFrame, Dict[str, List[BaseEstimator]]]:
    """
    Magnetic field features: rolling stats → lag rolling stats → Takens/convolution/history → scale originals → lag scaled originals.
    """
    
    # print(">> Adding magnetic field features...")
    fit_col_transforms = {}

    # Step 1: Compute rolling features on unscaled data
    dataset, rolling_bz_cols = _add_rolling_features(dataset, BZ_COL, window=12, overlap=0.3)
    dataset, rolling_scalar_b_cols = _add_rolling_features(dataset, SCALAR_B_COL, window=12, overlap=0.3)
    # print(" >>>> Done adding magnetic rolling features!")

    # Step 2: Lag the rolling features
    for _col in rolling_bz_cols + rolling_scalar_b_cols:
        dataset, _ = _add_lag_based_feature(dataset, _col, [0, 4, 8]) # 4, 8 previously
    # print(" >>>> Done lagging magnetic rolling features!")
    
    # Step 3: Takens/convolution/history on unscaled originals
    magnetic_columns = [BZ_COL, SCALAR_B_COL]
    magnetic_ms = [4, 4]
    magnetic_taus = [1, 1]
    
    dataset = run_takens_pipeline(dataset, magnetic_columns, magnetic_ms, magnetic_taus)
    # dataset = run_history_pipeline(dataset, magnetic_columns, n_lags=10)
    # Convolution on BZ and Scalar B
    dataset = add_magnetic_convolutions(dataset)
    # print(" >>>> Done adding magnetic Takens/signal/convolution features!")

    dataset, fit_col_transforms = apply_transforms_to_derived_columns(dataset, column_transforms, fit_transform)

    # Step 5: Lag the scaled originals
    dataset, _ = _add_lag_based_feature(dataset, BZ_COL, [0, 4, 8]) # previously 4, 8
    dataset, _ = _add_lag_based_feature(dataset, BY_COL, [0, 4, 8]) # previously 4
    dataset, _ = _add_lag_based_feature(dataset, SCALAR_B_COL, [0, 4, 8])
    # print(" >>>> Done lagging magnetic scaled features!")

    return dataset, fit_col_transforms


def add_standard_others(
        dataset: pd.DataFrame,
        column_transforms: Dict[str, List[BaseEstimator]],
        fit_transform: bool = True,
    ) -> Tuple[pd.DataFrame, Dict[str, List[BaseEstimator]]]:
    """
    Other features (pressure, E-field, alpha/proton): rolling stats → lag rolling stats → Takens/convolution/history → scale originals → lag scaled originals.
    """
    
    # print(">> Adding other features (pressure, E-field, alpha/proton)...")
    fit_col_transforms = {}

    # Step 1: Compute rolling features on unscaled data
    dataset, rolling_pressure_cols = _add_rolling_features(dataset, FLOW_PRESSURE_COL, window=24, overlap=0.3)
    dataset, rolling_efield_cols = _add_rolling_features(dataset, E_FIELD_COL, window=24, overlap=0.3)
    dataset, rolling_alpha_cols = _add_rolling_features(dataset, ALPHA_PROTON_RATIO_COL, window=24, overlap=0.3)
    # print(" >>>> Done adding other rolling features!")

    # Step 2: Lag the rolling features
    for _col in rolling_pressure_cols + rolling_efield_cols + rolling_alpha_cols:
        dataset, _ = _add_lag_based_feature(dataset, _col, [0, 4, 8]) # previously 4, 8
    # print(" >>>> Done lagging other rolling features!")

    # Step 3: Takens/convolution/history on unscaled originals
    others_columns = [FLOW_PRESSURE_COL, E_FIELD_COL, ALPHA_PROTON_RATIO_COL]
    others_ms = [5, 4, 5]
    others_taus = [2, 1, 2]
    
    dataset = run_takens_pipeline(dataset, others_columns, others_ms, others_taus)
    # dataset = run_history_pipeline(dataset, others_columns, n_lags=10)
    # Convolution on pressure and E-field
    dataset = add_others_convolutions(dataset)
    # print(" >>>> Done adding others Takens/signal/convolution features!")


    dataset, fit_col_transforms = apply_transforms_to_derived_columns(dataset, column_transforms, fit_transform)

    # Step 5: Lag the scaled originals
    dataset, _ = _add_lag_based_feature(dataset, FLOW_PRESSURE_COL, [0, 4, 8]) # previously 4, 8
    dataset, _ = _add_lag_based_feature(dataset, E_FIELD_COL, [0, 4, 8]) # previously 4, 8
    dataset, _ = _add_lag_based_feature(dataset, ALPHA_PROTON_RATIO_COL, [0, 8, 12]) # previously 8, 12
    # print(" >>>> Done lagging other scaled features!")

    return dataset, fit_col_transforms


def run_takens_pipeline(dataset: pd.DataFrame, col_names: List[str], ms: List[int], taus: List[int]) -> pd.DataFrame:
    """
    Takens embeddings pipeline on unscaled data to preserve topological structure.
    """
    
    # print(">> Adding Takens embeddings...")
    dataset, taken_cols = add_takens_embeddings(dataset, col_names, ms, taus)
    # print(">> Adding 2D pairs...")
    dataset, pair_cols = add_2d_pairs(dataset, taken_cols)
    # print(">> Adding 2D trajectory features...")
    dataset, feature_2d_cols = add_trajectory_features(dataset, pair_cols)
    # print("Done Takens pipeline!")
    dataset.drop(columns=taken_cols + pair_cols, inplace=True)

    return dataset


def run_history_pipeline(dataset: pd.DataFrame, col_names: List[str], n_lags: int) -> pd.DataFrame:
    """
    Signal processing features (max jump, jump concentration, trend strength) on unscaled data.
    """
    
    # print(">> Adding signal processing features...")
    dataset, lag_columns = add_lagged_history(dataset, col_names, n_lags=n_lags)
    dataset, signal_features = add_signal_features(dataset, lag_columns)
    dataset.drop(columns=lag_columns, inplace=True)
    # print("Done signal processing features!")

    return dataset


def add_solar_convolutions(dataset: pd.DataFrame) -> pd.DataFrame:
    """
    Apply convolution features to solar wind parameters.
    """
    # Speed: enhancement patterns (stream acceleration)
    dataset = add_convolution_feature(dataset, SW_SPEED_COL, "enhancement_short")
    dataset = add_convolution_feature(dataset, SW_SPEED_COL, "enhancement_long")
    
    # Density: compression patterns (shocks)
    dataset = add_convolution_feature(dataset, SW_DENSITY_COL, "compression_short")
    dataset = add_convolution_feature(dataset, SW_DENSITY_COL, "compression_long")
    
    # Temperature: shock heating and thermal evolution
    dataset = add_convolution_feature(dataset, SW_TEMP_COL, "compression_short")
    dataset = add_convolution_feature(dataset, SW_TEMP_COL, "enhancement_long")
    
    return dataset


def add_magnetic_convolutions(dataset: pd.DataFrame) -> pd.DataFrame:
    """
    Apply convolution features to magnetic field parameters.
    """
    # BZ: storm onset patterns (southward transitions)
    dataset = add_convolution_feature(dataset, BZ_COL, "storm_onset_short")
    dataset = add_convolution_feature(dataset, BZ_COL, "storm_onset_long")
    
    # Scalar B: field enhancement and compression
    dataset = add_convolution_feature(dataset, SCALAR_B_COL, "compression_short")
    dataset = add_convolution_feature(dataset, SCALAR_B_COL, "enhancement_long")
    
    return dataset


def add_others_convolutions(dataset: pd.DataFrame) -> pd.DataFrame:
    """
    Apply convolution features to pressure and E-field.
    """
    # Flow pressure: compression patterns
    dataset = add_convolution_feature(dataset, FLOW_PRESSURE_COL, "compression_short")
    dataset = add_convolution_feature(dataset, FLOW_PRESSURE_COL, "compression_long")
    
    return dataset


def apply_transforms_to_derived_columns(
    dataset: pd.DataFrame,
    column_transforms: Dict[str, List[BaseEstimator]],
    fit_transform: bool = True
) -> Tuple[pd.DataFrame, Dict[str, List[BaseEstimator]]]:
    """
    Apply transformations to derived columns based on their base column names.
    
    Searches the DataFrame for columns that contain base column names as substrings
    and applies the corresponding transformations.
    
    Parameters
    ----------
    dataset : pd.DataFrame
        The dataframe to search and transform
    column_transforms : Dict[str, List[BaseEstimator]]
        Mapping from base column names to their transformation pipelines
    fit_transform : bool
        Whether to fit the transformers or just transform
        
    Returns
    -------
    dataset : pd.DataFrame
        Dataset with transformed columns
    fitted_transforms : Dict[str, List[BaseEstimator]]
        Mapping from derived column names to their fitted transformers
    """
    if not column_transforms:
        return dataset, column_transforms
    
    fitted_transforms = {}
    
    # Search DataFrame columns for matches
    for col in dataset.columns:
        matching_base = next(
            (base_col for base_col in sorted(column_transforms.keys(), key=len, reverse=True)
             if base_col in col),
            None
        )
        
        if matching_base is not None:
            dataset, fitted_transf = apply_transformation_pipeline(
                dataset, 
                col,
                column_transforms[matching_base],
                fit_transform=fit_transform
            )
            fitted_transforms[col] = fitted_transf
    
    return dataset, fitted_transforms