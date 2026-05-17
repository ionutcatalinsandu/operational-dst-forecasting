import pandas as pd
import numpy as np

from typing import List, Tuple, Optional, Dict, Union
from scipy.interpolate import interp1d
from scipy import signal
from scipy.stats import wasserstein_distance, ks_2samp
from sklearn.preprocessing import RobustScaler
from statsmodels.tsa.seasonal import seasonal_decompose
import warnings
from sklearn.base import BaseEstimator
from numba import jit

from data_processing.scalers import SignedLogTransformer
from data_processing.constants import *
from data_processing.scalers import SignedLogTransformer, apply_transformation_pipeline




def compute_features(
        dataset: pd.DataFrame, 
        log_columns=False,
        target_transform: List=[BaseEstimator],
        energy_columns=False, 
        compute_scaler=False, 
        scaler: RobustScaler = None,
        angular: bool = True
    ) -> pd.DataFrame:

    log_scaler = SignedLogTransformer()

    if log_columns:
        for column in LOG_ROBUST_COLUMNS:
            if column in dataset.columns:
                dataset[column] = log_scaler.transform(dataset[column])
        print("Done selecting and sclaing values!")
                
    if len(target_transform):
        dataset = apply_transformation_pipeline(dataset, DST_COLUMN, target_transform)
        print("Done scaling target!")

    if energy_columns:
        dataset = compute_enou24(dataset)
        dataset = compute_epsilon24_with_edge_cases(dataset)
        print("Done adding energy columns.")

    columns_to_scale = ROBUST_ONLY_COLUMNS + LOG_ROBUST_COLUMNS + ANGULAR_ROBUST_ONLY
    remaining_columns = [col for col in dataset.columns if col not in columns_to_scale]
    temp_dataset = dataset[columns_to_scale].copy()

    if compute_scaler and scaler is None:
        print("# Fitting scaler and scaling data...")
        new_scaler = RobustScaler(quantile_range=(0.10, 0.90))
        temp_dataset = new_scaler.fit_transform(temp_dataset)
        temp_dataset = pd.DataFrame(temp_dataset, columns=columns_to_scale, index=dataset.index)
        temp_dataset[remaining_columns] = dataset[remaining_columns].copy()
        scaler = new_scaler
        dataset = temp_dataset.copy()

    if not compute_scaler and scaler is not None:
        print("# Scaling data with provided scaler ...")
        temp_dataset = scaler.transform(temp_dataset)
        temp_dataset = pd.DataFrame(temp_dataset, columns=columns_to_scale, index=dataset.index)
        temp_dataset[remaining_columns] = dataset[remaining_columns].copy()
        dataset = temp_dataset.copy()

    dataset = compute_lag_based_features(dataset, angular=angular)

    return dataset, scaler


def _add_lag_based_feature(dataset: pd.DataFrame, column_name: str, lags: List[int]):

    # print(f">> Adding {column_name} lagged features...")
    lag_column_list = []
    for index, _lag in enumerate(lags):
        dataset, _lag_col = create_lag_feature(dataset, column_name, _lag, replace_column_name=index != 0)
        lag_column_list.append(_lag_col)

    # print(f"Done adding {column_name} lagged features!")

    return dataset, lag_column_list

def _add_rolling_features(dataset: pd.DataFrame, column_name: str, window: int, overlap: float) -> Tuple[pd.DataFrame, List[str]]:

    # dataset = create_moving_average_feature(dataset, column_name, "rolling", window=window)
    dataset, rolling_std_col = add_rolling_std(dataset, column_name, window=window)
    dataset, rolling_cdf_col = add_cdf_distribution_trend(dataset, column_name, window=window, overlap_percent=overlap)
    dataset, rolling_median_col = add_rolling_median_features(dataset, column_name, window=window)

    rolling_cols = [rolling_std_col, rolling_cdf_col, rolling_median_col]

    return dataset.copy(), rolling_cols

def _add_convolution_features(dataset: pd.DataFrame) -> pd.DataFrame:

    dataset = dataset.copy()
    # Short-term: Substorm triggering (3-5 hours)
    dataset = add_convolution_feature(dataset, "BZ, nT (GSM)", "storm_onset_short")

    # Long-term: Main phase development (7-12 hours)  
    dataset = add_convolution_feature(dataset, "BZ, nT (GSM)", "storm_onset_long")

    # Short-term: Stream acceleration/shocks (2-6 hours)
    dataset = add_convolution_feature(dataset, "SW Plasma Speed, km/s", "enhancement_short")

    # Long-term: Stream development (8-24 hours)
    dataset = add_convolution_feature(dataset, "SW Plasma Speed, km/s", "enhancement_long")

    # Short-term: Sharp shocks (2-4 hours)
    dataset = add_convolution_feature(dataset, "SW Proton Density, N/cm^3", "compression_short")
    dataset = add_convolution_feature(dataset, "Flow pressure", "compression_short")

    # Long-term: Extended compression regions (6-12 hours)
    dataset = add_convolution_feature(dataset, "SW Proton Density, N/cm^3", "compression_long")
    dataset = add_convolution_feature(dataset, "Flow pressure", "compression_long")

    # Short-term: Field enhancement in shocks
    dataset = add_convolution_feature(dataset, "Scalar B, nT", "compression_short")

    # Long-term: Magnetic structure evolution  
    dataset = add_convolution_feature(dataset, "Scalar B, nT", "enhancement_long")

    # Short-term: Shock heating
    dataset = add_convolution_feature(dataset, "SW Plasma Temperature, K", "compression_short")

    # Long-term: Thermal evolution
    dataset = add_convolution_feature(dataset, "SW Plasma Temperature, K", "enhancement_long")

    return dataset

def _add_cross_correlation_features(dataset, plasma_speed_lags, proton_density_lags, bz_lags, 
                                   scalar_b_lags, flow_pressure_lags, electric_field_lags, 
                                   plasma_beta_lags, plasma_temp_lags):
    """Add the top 10 physics-motivated cross-correlation features using existing lag lists."""
    
    print(">> Adding cross-correlation features...")
    
    # Tier 1: Critical storm physics (Pearson)
    # BZ lag_4h (index 0) × Electric field lag_4h (index 0)
    dataset = add_window_based_cross_correlation(dataset, bz_lags[0], electric_field_lags[0], 12, 0.1, "pearson")
    
    # Speed lag_6h (index 0) × BZ lag_4h (index 0)  
    dataset = add_window_based_cross_correlation(dataset, plasma_speed_lags[0], bz_lags[0], 18, 0.1, "pearson")
    
    # Density lag_4h (index 0) × Pressure lag_4h (index 0)
    dataset = add_window_based_cross_correlation(dataset, proton_density_lags[0], flow_pressure_lags[0], 24, 0.1, "pearson")
    
    # Tier 2: Enhanced coupling (Instantaneous)
    # Speed lag_6h (index 0) × Density lag_4h (index 0) - mixed lag dynamic pressure
    dataset = add_window_based_cross_correlation(dataset, plasma_speed_lags[0], proton_density_lags[0], 24, 0.1, "instantaneous")
    
    # BZ lag_4h (index 0) × Scalar B lag_4h (index 0)
    dataset = add_window_based_cross_correlation(dataset, bz_lags[0], scalar_b_lags[0], 18, 0.67, "instantaneous")
    
    # Speed lag_6h (index 0) × Electric field lag_4h (index 0)
    dataset = add_window_based_cross_correlation(dataset, plasma_speed_lags[0], electric_field_lags[0], 12, 0.25, "instantaneous")
    
    # Tier 3: Background coupling (Spearman)
    # Beta lag_6h (index 0) × BZ lag_4h (index 0)
    dataset = add_window_based_cross_correlation(dataset, plasma_beta_lags[0], bz_lags[0], 24, 0.1, "spearman")
    
    # Temperature lag_8h (index 0) × Density lag_4h (index 0)
    dataset = add_window_based_cross_correlation(dataset, plasma_temp_lags[0], proton_density_lags[0], 36, 0.05, "spearman")
    
    # Scalar B lag_4h (index 0) × Density lag_4h (index 0)
    dataset = add_window_based_cross_correlation(dataset, scalar_b_lags[0], proton_density_lags[0], 24, 0.05, "spearman")
    
    # Tier 4: Source identification (Pearson)
    # Speed lag_6h (index 0) × Temperature lag_8h (index 0)
    dataset = add_window_based_cross_correlation(dataset, plasma_speed_lags[0], plasma_temp_lags[0], 48, 0.5, "pearson")
    
    print(">> Done cross-correlation features!")
    return dataset


def compute_lag_based_features(dataset: pd.DataFrame, angular: bool = False) -> pd.DataFrame:

    # Solar
    dataset, plasma_speed_lags = _add_lag_based_feature(dataset, "SW Plasma Speed, km/s", [6, 24, 72])
    dataset = _add_rolling_features(dataset, "SW Plasma Speed, km/s", window=24, overlap=0.5)
    dataset, plasma_temp_lags = _add_lag_based_feature(dataset, "SW Plasma Temperature, K", [8, 24])
    dataset = _add_rolling_features(dataset, "SW Plasma Temperature, K", window=48, overlap=0.25)
    dataset, proton_density_lags = _add_lag_based_feature(dataset, "SW Proton Density, N/cm^3", [4, 12, 36])
    dataset = _add_rolling_features(dataset, "SW Proton Density, N/cm^3", window=36, overlap=0.5)
    dataset, plasma_beta_lags = _add_lag_based_feature(dataset, "Plasma Beta", [6, 18], 24, 0.10)
    dataset, plasma_flow_long_angle_lags = _add_lag_based_feature(dataset, "SW Plasma flow long. angle", [6, 24])
    dataset, plasma_flow_lat_angle_lags = _add_lag_based_feature(dataset, "SW Plasma flow lat. angle", [6, 24])

    # Magnetic field
    # dataset, b_mag_lags = _add_lag_based_feature(dataset, "Vector B Magnitude,nT", [4, 12, 24])
    # dataset = _add_rolling_features(dataset, "Vector B Magnitude,nT", window=18, overlap=0.5)
    if angular:
        dataset, b_lat_angle_sin_lags = _add_lag_based_feature(dataset, "Lat. Angle of B (GSE)_sin", [4, 12])
        dataset, b_lat_angle_cos_lags = _add_lag_based_feature(dataset, "Lat. Angle of B (GSE)_cos", [4, 12])
        dataset, b_long_angle_sin_lags = _add_lag_based_feature(dataset, "Long. Angle of B (GSE)_sin", [4, 12])
        dataset, b_long_angle_cos_lags = _add_lag_based_feature(dataset, "Long. Angle of B (GSE)_cos", [4, 12])
    else:
        dataset, b_lat_angle_lags = _add_lag_based_feature(dataset, "Lat. Angle of B (GSE)", [4, 12])
        dataset, b_long_angle_lags = _add_lag_based_feature(dataset, "Long. Angle of B (GSE)", [4, 12])
    # dataset, bx_lags = _add_lag_based_feature(dataset, "BX, nT (GSE, GSM)", [4, 12])
    # dataset, by_lags = _add_lag_based_feature(dataset, "BY, nT (GSM)", [4, 12])
    dataset, bz_lags = _add_lag_based_feature(dataset, "BZ, nT (GSM)", [4, 6, 12, 24])
    dataset = _add_rolling_features(dataset, "BZ, nT (GSM)", window=12, overlap=0.75)
    dataset, scalar_b_lags = _add_lag_based_feature(dataset, "Scalar B, nT", [4, 12])

    # Others
    dataset, flow_pressure_lags = _add_lag_based_feature(dataset, "Flow pressure", [4, 8, 18])
    dataset = _add_rolling_features(dataset, "Flow pressure", window=18, overlap=0.67)
    dataset, electric_field_lags = _add_lag_based_feature(dataset, "E elecrtric field", [4, 8, 12])
    # dataset, alfen_lags = _add_lag_based_feature(dataset, "Alfen mach number", [4, 12])
    # dataset, mag_munch_lags = _add_lag_based_feature(dataset, "Magnetosonic Much num.", [4, 12])
    # dataset, alpa_prot_lags = _add_lag_based_feature(dataset, "Alpha/Prot. ratio", [12, 48])

    print(">> Adding convolution features...")
    dataset = _add_convolution_features(dataset)
    print(">> Done!")
    dataset = _add_cross_correlation_features(dataset, plasma_speed_lags, proton_density_lags, bz_lags, 
                                            scalar_b_lags, flow_pressure_lags, electric_field_lags, 
                                            plasma_beta_lags, plasma_temp_lags)
    
    print(">> Adding time series decomposition features...")

    # # 1. BZ Component (bz_lags[0] = 4h lag) - Most critical for storms
    # dataset = add_time_series_decomposition(
    #     dataset, bz_lags[0], 
    #     components=['trend', 'resid'], 
    #     period=672  # 28-day solar rotation
    # )

    # # 2. Solar Wind Speed (plasma_speed_lags[0] = 6h lag) - Stream structure
    # dataset = add_time_series_decomposition(
    #     dataset, plasma_speed_lags[0], 
    #     components=['seasonal', 'trend', 'resid'], 
    #     period=2160  # ~3 month periods
    # )

    # # 3. Proton Density (proton_density_lags[0] = 4h lag) - Compression events
    # dataset = add_time_series_decomposition(
    #     dataset, proton_density_lags[0], 
    #     components=['trend', 'resid'], 
    #     period=672  # Solar rotation
    # )

    # # 4. Magnetic Field Magnitude (b_mag_lags[1] = 12h lag) - Magnetic structure
    # dataset = add_time_series_decomposition(
    #     dataset, b_mag_lags[1], 
    #     components=['seasonal', 'resid'], 
    #     period=4380  # Semi-annual variation
    # )

    # # 5. Plasma Temperature (plasma_temp_lags[0] = 8h lag) - Thermal state
    # dataset = add_time_series_decomposition(
    #     dataset, plasma_temp_lags[0], 
    #     components=['trend', 'resid'], 
    #     period=168  # Weekly patterns
    # )

    # print(">> Done with decomposition features!")

    dataset.dropna(inplace=True)

    return dataset


def add_rolling_std(df: pd.DataFrame, column: str, window: int, min_periods: int = 3) -> Tuple[pd.DataFrame, str]:
    """
    Add rolling std with expanding window for cold start.
    
    Strategy:
    - Rows 0 to (window-1): Use expanding std (all available history)
    - Rows window+: Use fixed rolling window std
    - Always require min_periods (default: 3) for statistical validity
    
    Physics rationale: 
    Early magnetospheric variability estimates use all available context.
    Once sufficient history exists, focus on recent dynamics.
    """
    new_col = f"{column}_std_{window}r"
    
    # Compute both expanding and rolling
    expanding_std = df[column].expanding(min_periods=min_periods).std()
    rolling_std = df[column].rolling(window=window, min_periods=min_periods).std()
    
    # Use rolling where available, expanding otherwise
    df[new_col] = rolling_std.fillna(expanding_std)
    
    return df, new_col


def add_enhanced_rolling_std(df, column, window, overlap_percent=0.25):
    """
    Compute rolling std with overlapping windows, propagate values across each window,
    and adjust by subtracting the simple rolling std to avoid redundancy.
    
    Args:
        df (pd.DataFrame): Input DataFrame.
        column (str): Column name for which to compute std.
        window (int): Window size in rows (e.g., 24 for 24 hours).
        overlap_percent (float): Overlap percentage (0 ≤ overlap < 1). Default: 0.25.
    
    Returns:
        pd.DataFrame: New columns with adjusted std values.
    """
    # --- Step 1: Compute simple rolling std (no overlap) ---
    simple_std_col = f"{column}_std_{window}r"
    df[simple_std_col] = df[column].rolling(window=window).std()
    
    # --- Step 2: Compute sliding window std with overlap ---
    step = max(1, int(window * (1 - overlap_percent)))
    arr = df[column].to_numpy()
    n = len(arr)
    
    # Initialize output array with NaNs
    sliding_std = np.full(n, np.nan)
    
    # Slide window and compute std
    for i in range(0, n - window + 1, step):
        current_window = arr[i:i + window]
        # print(current_window)
        current_window_std = np.nanstd(current_window, ddof=1)
        sliding_std[i:i + window] = current_window_std  # Propagate to all window positions
    
    # Store sliding std results
    # sliding_std_col = f"{column}_sliding_std_{window}r_overlap{int(overlap_percent*100)}"
    # df[sliding_std_col] = sliding_std
    
    # --- Step 3: Subtract simple rolling std to adjust values ---
    adjusted_std_col = f"{column}_adjusted_std_{window}r"
    df[adjusted_std_col] = df[simple_std_col] - sliding_std
    
    return df


def add_rolling_median_features(df: pd.DataFrame, column: str, window: int, 
                                min_valid_points: int = 3) -> Tuple[pd.DataFrame, str]:
    """
    Add rolling median with expanding window for cold start.
    
    Strategy:
    - Early rows: Use expanding median (all available history)
    - Later rows: Use fixed rolling window median
    - Always require min_valid_points for statistical validity
    
    Args:
        df: Input DataFrame
        column: Column to analyze
        window: Window size in samples
        min_valid_points: Minimum non-NaN points required per window (default: 3)
    
    Returns:
        Tuple[pd.DataFrame, str]: DataFrame with rolling median feature and column name
    """
    new_col = f"{column}_rolling_median_{window}r"
    
    # Compute both expanding and rolling
    expanding_median = df[column].expanding(min_periods=min_valid_points).mean()
    rolling_median = df[column].rolling(window=window, min_periods=min_valid_points).median()
    
    # Use rolling where available, expanding otherwise
    df[new_col] = rolling_median.fillna(expanding_median)
    
    return df, new_col


def add_cdf_distribution_trend(
    df: pd.DataFrame,
    column: str,
    window: int,
    overlap_percent: float = 0.25,
    distance_metric: str = "wasserstein",
    min_valid_points: int = 3
) -> Tuple[pd.DataFrame, str]:
    """
    Compute distributional trend using CDF distances.
    
    Cold start strategy: 
    - First window: NaN (no reference for comparison)
    - Subsequent windows: compared to previous window
    - Forward-only interpolation between computed values
    """
    
    step = max(1, int(window * (1 - overlap_percent)))
    arr = df[column].to_numpy()
    n = len(arr)
    
    dist_trend = np.full(n, np.nan)
    prev_window = None
    
    for i in range(0, n, step):
        window_end = min(i + window, n)
        current_window = arr[i:window_end]
        current_window = current_window[~np.isnan(current_window)]
        
        if len(current_window) < min_valid_points:
            continue
        
        # First window: NaN (no previous reference)
        if prev_window is None:
            dist = np.nan
        else:
            try:
                if distance_metric == "wasserstein":
                    dist = wasserstein_distance(prev_window, current_window)
                elif distance_metric == "ks":
                    dist = ks_2samp(prev_window, current_window).statistic
                else:
                    raise ValueError(f"Unsupported metric: {distance_metric}")
            except Exception as e:
                warnings.warn(f"Distance failed at index {i}: {e}")
                dist = np.nan
        
        assign_idx = min(window_end - 1, n - 1)
        dist_trend[assign_idx] = dist
        prev_window = current_window
    
    # Forward-only interpolation between valid points
    interpolated_dist = pd.Series(dist_trend).interpolate(
        method='linear',
        limit_direction='forward',
        limit_area='inside'
    )
    
    metric_suffix = "wass" if distance_metric == "wasserstein" else distance_metric
    trend_col_name = f"{column}_cdf_{metric_suffix}_trend_{window}r"
    df[trend_col_name] = interpolated_dist.values
    
    return df, trend_col_name


def create_lag_feature(df: pd.DataFrame, column: str, shift: int, replace_column_name: bool = True) -> tuple[pd.DataFrame, str]:

    column_name = column
    if replace_column_name:
        column_name = f"{column}_lag_{shift}"
        
    df[column_name] = df[column].shift(shift)

    return df.copy(), column_name


def create_moving_average_feature(df: pd.DataFrame, column: str, method: str = "rolling", 
                                  window: int = 5, span: int = None, std_mult: float = 2.0) -> pd.DataFrame:
    """
    Create moving average or Bollinger bands feature.
    
    Args:
        df (pd.DataFrame): Input dataframe.
        column (str): Column to process.
        method (str): Type of method ('rolling', 'ewm', 'bollinger', 'median').
        window (int): Window size for rolling.
        span (int): Span for exponential weighted.
        std_mult (float): Std multiplier for Bollinger bands.
        
    Returns:
        pd.DataFrame: DataFrame with new feature(s) added.
    """
    
    if method == "rolling":
        df[f"{column}_ma_{window}"] = df[column].rolling(window=window, min_periods=1).mean()
    
    elif method == "ewm":
        df[f"{column}_ewm_{span}"] = df[column].ewm(span=span, adjust=False).mean()
    
    elif method == "median":
        df[f"{column}_median_{window}"] = df[column].rolling(window=window, min_periods=1).median()
    
    elif method == "bollinger":
        rolling_mean = df[column].rolling(window=window, min_periods=1).mean()
        rolling_std = df[column].rolling(window=window, min_periods=1).std()
        df[f"{column}_bb_upper_{window}"] = rolling_mean + std_mult * rolling_std
        df[f"{column}_bb_lower_{window}"] = rolling_mean - std_mult * rolling_std
        # df[f"{column}_bollinger_dff"] = df[f"{column}_bb_upper_{window}"] - df[f"{column}_bb_lower_{window}"]
    
    else:
        raise ValueError(f"Unknown method '{method}'. Choose from 'rolling', 'ewm', 'median', 'bollinger'.")
    
    return df.copy()

def compute_enou24(df: pd.DataFrame) -> pd.DataFrame:

    rho = df["SW Proton Density, N/cm^3"]
    v = df["SW Plasma Speed, km/s"]
    bx = df["BX, nT (GSE, GSM)"]
    by = df["BY, nT (GSE)"]
    bz = df["BZ, nT (GSE)"]

    # Absolute values of By, Bz for angle calculation
    abs_by = np.abs(by)
    abs_bz = np.abs(bz)

    # Clock angle theta (avoid division by zero by adding tiny epsilon)
    epsilon = 1e-10
    theta = np.arctan2(abs_by, abs_bz + epsilon)  # angle in radians

    # Conditional for Bz
    bz_negative = bz < 0
    bz_positive = bz >= 0

    # Magnetic field magnitude part (Bx² + By²)^0.43
    bxy_magnitude = (bx ** 2 + by ** 2) ** 0.43

    # Sine component (adjusted for Bz condition)
    sin_term = np.zeros(len(df))

    # For Bz < 0
    sin_term[bz_negative] = (np.sin(theta[bz_negative] / 2) ** 2.7) + 0.25

    # For Bz > 0 (use pi - theta)
    sin_term[bz_positive] = (np.sin((np.pi - theta[bz_positive]) / 2) ** 2.7) + 0.25

    # Final enou24 formula
    enou24 = (
        3.78e7 *
        (rho ** 0.24) *
        (v ** 1.47) *
        bxy_magnitude *
        sin_term
    )

    # Add to dataframe
    new_df = df.copy()
    new_df[ENOU_24_COLUMN] = enou24

    return new_df

def compute_epsilon24_with_edge_cases(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute epsilon24 (W) based on given DataFrame, properly handling By and Bz zero cases.

    Parameters:
        df (pd.DataFrame): DataFrame with "SW Plasma Speed, km/s", "BX, nT (GSE)", "BY, nT (GSE)", "BZ, nT (GSE)"

    Returns:
        pd.Series: Computed epsilon24 (W)
    """

    v = df["SW Plasma Speed, km/s"]
    bx = df["BX, nT (GSE, GSM)"]
    by = df["BY, nT (GSE)"]
    bz = df["BZ, nT (GSE)"]

    # Bscalar24 = sqrt(Bx^2 + By^2 + Bz^2)
    bscalar = np.sqrt(bx ** 2 + by ** 2 + bz ** 2)

    # Initialize epsilon24 with zeros
    epsilon24 = np.zeros(len(df))

    # Masks for different cases
    mask_by_zero = (by == 0) & (bz != 0)    # By = 0, Bz ≠ 0 -> epsilon = 0
    mask_bz_zero = (bz == 0) & (by != 0)    # Bz = 0, By ≠ 0 -> special angle
    mask_both_zero = (by == 0) & (bz == 0)  # By = 0, Bz = 0 -> epsilon = 0
    mask_valid = (by != 0) & (bz != 0)      # Both non-zero, valid for computation

    # --- Normal case: By ≠ 0 and Bz ≠ 0 ---
    abs_by = np.abs(by[mask_valid])
    abs_bz = np.abs(bz[mask_valid])

    theta_valid = np.arctan2(abs_by, abs_bz)

    sin_term = np.zeros(len(df))

    # For Bz < 0
    mask_bz_neg = bz[mask_valid] < 0
    sin_term_valid = np.zeros(len(abs_by))
    sin_term_valid[mask_bz_neg] = np.sin(theta_valid[mask_bz_neg] / 2) ** 4

    # For Bz >= 0 (pi - theta)
    mask_bz_pos = ~mask_bz_neg
    sin_term_valid[mask_bz_pos] = np.sin((np.pi - theta_valid[mask_bz_pos]) / 2) ** 4

    # Apply computed values back into the main sin_term array
    sin_term[mask_valid] = sin_term_valid

    # --- Case when Bz == 0 (By ≠ 0) ---
    # theta = pi/2
    theta_bz_zero = np.pi / 2
    sin_term[mask_bz_zero] = np.sin(theta_bz_zero / 2) ** 4  # This is a constant value: (sin(pi/4))^4

    # --- Final epsilon24 formula ---
    epsilon24 = 1.997e7 * v * (bscalar ** 2) * sin_term

    # By = 0 case and By = Bz = 0 case are left as 0 (already initialized to zero)

    new_df = df.copy()
    new_df[EPSILON_COLUMN] = epsilon24

    return new_df


def add_window_based_cross_correlation(df: pd.DataFrame,
                                     col1: str,
                                     col2: str, 
                                     window_size: int = 24,
                                     overlap_percent: float = 0.5,
                                     method: str = 'pearson',
                                     interpolation_method: str = 'linear') -> pd.DataFrame:
    """
    Compute cross-correlation using sliding windows with interpolation for missing values.
    
    This approach is much faster than rolling correlation because:
    1. Computes correlation only at window centers (not every point)
    2. Uses efficient interpolation to fill intermediate values
    3. Maintains temporal patterns while reducing computation time
    
    Args:
        df: Input dataframe
        col1: First column name
        col2: Second column name
        window_size: Size of correlation window in data points (hours for hourly data)
        overlap_percent: Overlap between windows (0.0 = no overlap, 0.9 = 90% overlap)
        method: Correlation method ('pearson', 'spearman', 'instantaneous')
        interpolation_method: Interpolation method ('linear', 'cubic', 'quadratic')
    
    Returns:
        DataFrame with new cross-correlation column
        
    Example:
        window_size=24, overlap_percent=0.5 means:
        - 24-hour correlation windows
        - 12-hour step between window centers  
        - ~2x speedup vs rolling correlation
    """
    df = df.copy()
    
    # Clean column names for feature naming
    col1_clean = col1.replace(' ', '_').replace(',', '').replace('(', '').replace(')', '')
    col2_clean = col2.replace(' ', '_').replace(',', '').replace('(', '').replace(')', '')
    
    # Calculate step size based on overlap
    step_size = max(1, int(window_size * (1 - overlap_percent)))
    
    # Initialize arrays
    n_points = len(df)
    correlation_values = []
    window_centers = []
    
    # Calculate correlations at window centers
    for start_idx in range(0, n_points - window_size + 1, step_size):
        end_idx = start_idx + window_size
        center_idx = start_idx + window_size // 2
        
        # Extract window data
        window_data1 = df[col1].iloc[start_idx:end_idx]
        window_data2 = df[col2].iloc[start_idx:end_idx]
        
        # Compute correlation for this window
        corr_value = compute_window_correlation(window_data1, window_data2, method)
        
        correlation_values.append(corr_value)
        window_centers.append(center_idx)
    
    # Handle edge cases for interpolation
    if len(correlation_values) == 0:
        # No valid windows - return zeros
        new_col_name = f"window_corr_{col1_clean}_{col2_clean}_{window_size}w_{int(overlap_percent*100)}ovlp"
        df[new_col_name] = 0.0
        return df
    
    elif len(correlation_values) == 1:
        # Only one window - use constant value
        new_col_name = f"window_corr_{col1_clean}_{col2_clean}_{window_size}w_{int(overlap_percent*100)}ovlp"
        df[new_col_name] = correlation_values[0]
        return df
    
    # Interpolate correlation values for all data points
    interpolated_correlation = interpolate_correlation_values(
        window_centers, correlation_values, n_points, interpolation_method
    )
    
    # Create column name
    new_col_name = f"window_corr_{col1_clean}_{col2_clean}_{window_size}w_{int(overlap_percent*100)}ovlp"
    df[new_col_name] = interpolated_correlation
    
    return df


def compute_window_correlation(series1: pd.Series, 
                             series2: pd.Series, 
                             method: str = 'pearson') -> float:
    """
    Compute correlation between two series with robust error handling.
    
    Args:
        series1: First data series
        series2: Second data series  
        method: Correlation method
        
    Returns:
        Correlation value (0.0 if computation fails)
    """
    try:
        # Remove NaN values
        valid_mask = ~(series1.isna() | series2.isna())
        
        if valid_mask.sum() < 3:  # Need at least 3 points
            return 0.0
        
        clean_series1 = series1[valid_mask]
        clean_series2 = series2[valid_mask]
        
        if method == 'pearson':
            # Check for zero variance
            if clean_series1.std() == 0 or clean_series2.std() == 0:
                return 0.0
            
            correlation = clean_series1.corr(clean_series2)
            
        elif method == 'spearman':
            # Spearman correlation (rank-based)
            if len(clean_series1.unique()) < 2 or len(clean_series2.unique()) < 2:
                return 0.0
                
            from scipy.stats import spearmanr
            correlation, _ = spearmanr(clean_series1, clean_series2)
            
        elif method == 'instantaneous':
            # Normalized element-wise product (mean across window)
            norm_series1 = (clean_series1 - clean_series1.mean()) / (clean_series1.std() + 1e-8)
            norm_series2 = (clean_series2 - clean_series2.mean()) / (clean_series2.std() + 1e-8)
            correlation = (norm_series1 * norm_series2).mean()
            
        else:
            raise ValueError(f"Unknown correlation method: {method}")
        
        # Safety checks
        if np.isnan(correlation) or np.isinf(correlation):
            return 0.0
            
        # Clip to valid correlation range
        return np.clip(correlation, -1.0, 1.0)
        
    except Exception as e:
        warnings.warn(f"Correlation computation failed: {e}")
        return 0.0

def interpolate_correlation_values(window_centers: list,
                                 correlation_values: list, 
                                 n_points: int,
                                 method: str = 'linear') -> np.ndarray:
    """
    Interpolate correlation values to fill all time points.
    
    Args:
        window_centers: Indices where correlations were computed
        correlation_values: Computed correlation values
        n_points: Total number of points needed
        method: Interpolation method
        
    Returns:
        Array of interpolated correlation values
    """
    if len(window_centers) != len(correlation_values):
        raise ValueError("Mismatch between window centers and correlation values")
    
    # Create interpolation function
    if method == 'linear':
        # Extend boundaries to avoid extrapolation issues
        extended_centers = [0] + window_centers + [n_points - 1]
        extended_values = [correlation_values[0]] + correlation_values + [correlation_values[-1]]
        
        # Linear interpolation
        interpolated = np.interp(range(n_points), extended_centers, extended_values)
        
    else:
        try:
            # Use scipy for more advanced interpolation
            if len(window_centers) < 2:
                # Not enough points for advanced interpolation
                interpolated = np.full(n_points, correlation_values[0])
            else:
                # Extend boundaries
                extended_centers = [0] + window_centers + [n_points - 1]
                extended_values = [correlation_values[0]] + correlation_values + [correlation_values[-1]]
                
                # Create interpolator
                interpolator = interp1d(extended_centers, extended_values, 
                                      kind=method, bounds_error=False, 
                                      fill_value='extrapolate')
                
                interpolated = interpolator(range(n_points))
                
        except Exception as e:
            warnings.warn(f"Advanced interpolation failed, falling back to linear: {e}")
            # Fallback to linear interpolation
            extended_centers = [0] + window_centers + [n_points - 1]
            extended_values = [correlation_values[0]] + correlation_values + [correlation_values[-1]]
            interpolated = np.interp(range(n_points), extended_centers, extended_values)
    
    return interpolated


def add_dynamic_pressure(df: pd.DataFrame, 
                        speed_col: str = "SW Plasma Speed, km/s",
                        density_col: str = "SW Proton Density, N/cm^3") -> pd.DataFrame:
    """
    Compute solar wind dynamic pressure - key driver of magnetospheric compression.
    
    Formula: P_dyn = ρ * v² * m_p = 1.67×10⁻⁶ * n * v² [nPa]
    
    Physics: Dynamic pressure compresses the magnetosphere, affecting:
    - Magnetopause standoff distance
    - Ring current particle injection
    - Substorm triggering threshold
    
    Args:
        df: Input dataframe
        speed_col: Solar wind speed column name
        density_col: Proton density column name
    
    Returns:
        DataFrame with new "Dynamic_Pressure_nPa" column
    """
    df = df.copy()
    
    # Constants
    m_p = 1.67e-27  # Proton mass (kg)
    conversion_factor = 1.67e-6  # Conversion to nPa for typical units
    
    # Calculate dynamic pressure
    # P_dyn = n * v² * m_p (converted to nPa)
    speed_ms = df[speed_col] * 1000  # Convert km/s to m/s
    density_m3 = df[density_col] * 1e6  # Convert N/cm³ to N/m³
    
    dynamic_pressure = conversion_factor * density_m3 * (speed_ms / 1000)**2
    
    df["Dynamic_Pressure_nPa"] = dynamic_pressure
    
    return df

def add_electric_field(df: pd.DataFrame,
                      speed_col: str = "SW Plasma Speed, km/s", 
                      bz_col: str = "BZ, nT (GSM)") -> pd.DataFrame:
    """
    Compute dawn-dusk electric field - primary driver of magnetic reconnection.
    
    Formula: E = V × B_z [mV/m] (taking only the z-component)
    
    Physics: The solar wind electric field drives magnetic reconnection at
    the magnetopause, controlling energy input during storms.
    Only the southward (negative) Bz component contributes to reconnection.
    
    Args:
        df: Input dataframe
        speed_col: Solar wind speed column name
        bz_col: IMF Bz component column name
    
    Returns:
        DataFrame with new "Electric_Field_mV_m" column
    """
    df = df.copy()
    
    # Convert units: km/s * nT → mV/m
    # 1 km/s * 1 nT = 1×10⁻³ mV/m
    conversion_factor = 1e-3
    
    # Calculate electric field
    # For reconnection, we want the magnitude when Bz is southward (negative)
    # Use -Bz so that southward field gives positive electric field
    electric_field = df[speed_col] * (-df[bz_col]) * conversion_factor
    
    # Set to zero when Bz is northward (no reconnection)
    electric_field = np.maximum(electric_field, 0)
    
    df["Electric_Field_mV_m"] = electric_field
    
    return df

def add_alfven_mach_number(df: pd.DataFrame,
                          speed_col: str = "SW Plasma Speed, km/s",
                          b_col: str = "Scalar B, nT", 
                          density_col: str = "SW Proton Density, N/cm^3") -> pd.DataFrame:
    """
    Compute Alfvén Mach number - characterizes solar wind-magnetosphere coupling efficiency.
    
    Formula: M_A = V_sw / V_A, where V_A = B / sqrt(μ₀ * ρ * m_p)
    
    Physics: The Alfvén Mach number determines:
    - Shock formation and compression ratios
    - Wave propagation and turbulence levels  
    - Magnetosphere-solar wind coupling efficiency
    - Particle acceleration processes
    
    High M_A (>10): Strong shocks, efficient energy transfer
    Low M_A (<3): Weak coupling, limited storm potential
    
    Args:
        df: Input dataframe
        speed_col: Solar wind speed column name
        b_col: Magnetic field magnitude column name
        density_col: Proton density column name
    
    Returns:
        DataFrame with new "Alfven_Mach_Number" column
    """
    df = df.copy()
    
    # Physical constants
    mu_0 = 4 * np.pi * 1e-7  # Permeability of free space (H/m)
    m_p = 1.67e-27  # Proton mass (kg)
    
    # Unit conversions
    speed_ms = df[speed_col] * 1000  # km/s to m/s
    b_tesla = df[b_col] * 1e-9  # nT to Tesla
    density_m3 = df[density_col] * 1e6  # N/cm³ to N/m³
    
    # Calculate Alfvén speed: V_A = B / sqrt(μ₀ * ρ * m_p)
    alfven_speed = b_tesla / np.sqrt(mu_0 * density_m3 * m_p)
    
    # Calculate Alfvén Mach number
    alfven_mach = speed_ms / alfven_speed
    
    # Handle division by zero (very rare in real data)
    alfven_mach = np.where(alfven_speed > 0, alfven_mach, np.nan)
    
    df["Alfven_Mach_Number"] = alfven_mach
    
    return df


# def add_convolution_feature(df: pd.DataFrame,
#                            column_name: str,
#                            filter_type: str,
#                            normalize_output: bool = True) -> pd.DataFrame:
#     """
#     Apply convolution-based pattern detection to space weather time series.
    
#     This function detects specific temporal signatures in space weather data
#     that are associated with different magnetospheric processes.
    
#     Args:
#         df: Input dataframe
#         column_name: Name of column to convolve
#         filter_type: Type of convolution filter to apply
#         normalize_output: Whether to normalize the convolution output
        
#     Returns:
#         DataFrame with new convolution feature column
        
#     Physics motivation:
#     - Storm onset: Detects rapid transitions in IMF/solar wind
#     - Compression: Identifies shock and CME compression signatures  
#     - Flux rope: Recognizes magnetic cloud rotation patterns
#     - Shock: Detects discontinuities and sudden changes
#     - Decay: Captures exponential recovery processes
#     """
#     df = df.copy()
    
#     # Get the convolution kernel for the specified filter type
#     kernel = get_convolution_kernel(filter_type)
    
#     # Extract the data series, handling NaN values
#     data_series = df[column_name].ffill().bfill().fillna(0)
    
#     # Apply convolution
#     try:
#         # Use 'same' mode to keep the same length as input
#         convolved = signal.convolve(data_series.values, kernel, mode='same')
        
#         # Skip normalization for scaled/logged data (as requested)
#         # if normalize_output:
#         #     convolved = normalize_convolution_output(convolved, filter_type)
        
#         # Handle any remaining inf/nan values
#         convolved = np.nan_to_num(convolved, nan=0.0, posinf=0.0, neginf=0.0)
        
#     except Exception as e:
#         warnings.warn(f"Convolution failed for {column_name} with {filter_type}: {e}")
#         convolved = np.zeros(len(data_series))
    
#     # Create descriptive column name
#     column_clean = column_name.replace(' ', '_').replace(',', '').replace('(', '').replace(')', '')
#     new_column_name = f"conv_{filter_type}_{column_clean}"
    
#     df[new_column_name] = convolved
    
#     return df


def add_convolution_feature(df: pd.DataFrame,
                           column_name: str,
                           filter_type: str) -> pd.DataFrame:
    """
    Apply causal convolution-based pattern detection to space weather time series.
    
    Physics-motivated kernels detect specific solar wind signatures:
    - storm_onset: Rapid BZ transitions
    - enhancement: Speed/field increases
    - compression: Pressure pulses
    
    Args:
        df: Input dataframe
        column_name: Name of column to convolve
        filter_type: Type of convolution filter to apply
        
    Returns:
        DataFrame with new convolution feature column
        
    NaN Handling:
    - NaN values in source data propagate to convolution output
    - No artificial filling (preserves data gaps)
    """
    df = df.copy()
    
    # Get the convolution kernel for the specified filter type
    kernel = get_convolution_kernel(filter_type)
    
    # Extract data WITHOUT filling NaN (preserve gaps)
    data_series = df[column_name].values
    
    # Apply causal convolution
    try:
        convolved = causal_convolution(data_series, kernel)
        
        # Only handle computational inf (not NaN - those are meaningful)
        convolved = np.where(np.isinf(convolved), np.nan, convolved)
        
    except Exception as e:
        warnings.warn(f"Causal convolution failed for {column_name} with {filter_type}: {e}")
        convolved = np.full(len(data_series), np.nan)  # NaN on failure, not 0
    
    # Create descriptive column name
    column_clean = column_name.replace(' ', '_').replace(',', '').replace('(', '').replace(')', '')
    new_column_name = f"conv_{filter_type}_{column_clean}"
    
    df[new_column_name] = convolved
    
    return df


def get_convolution_kernel(filter_type: str) -> np.ndarray:
    """
    Define physics-motivated convolution kernels for space weather pattern detection.
    
    Args:
        filter_type: Type of filter/pattern to detect
        
    Returns:
        Numpy array representing the convolution kernel
        
    Physics-based kernel definitions with short/long timescale variants:
    """
    kernels = {
        # Storm onset detection - SHORT (3-5 hours): Rapid substorm triggering
        'storm_onset_short': np.array([-1, -0.5, 0, 0.5, 1]) / 2.0,
        
        # Storm onset detection - LONG (7-12 hours): Main phase development  
        'storm_onset_long': np.array([-1, -1, -0.5, -0.2, 0, 0.2, 0.5, 1, 1, 1, 1]) / 6.0,
        
        # Compression detection - SHORT (2-4 hours): Sharp shocks
        'compression_short': np.array([0.2, 1.0, 0.2]) / 1.4,
        
        # Compression detection - LONG (6-12 hours): Extended compression (CME sheath)
        'compression_long': np.array([0.1, 0.2, 0.4, 0.8, 1.0, 1.0, 0.8, 0.4, 0.2, 0.1]) / 5.0,
        
        # Enhancement detection - SHORT (2-6 hours): Rapid increases  
        'enhancement_short': np.array([0.1, 0.3, 0.6, 1.0]) / 2.0,
        
        # Enhancement detection - LONG (8-24 hours): Gradual stream development
        'enhancement_long': np.array([0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]) / 6.0,
        
        # Legacy kernels (kept for backward compatibility)
        'storm_onset': np.array([-1, -1, -0.5, 0, 0.5, 1, 1]) / 3.5,
        'compression': np.array([0.1, 0.2, 0.4, 1.0, 0.4, 0.2, 0.1]) / 2.4,
        'shock': np.array([0, 0, 0, 1, -1, 0, 0]),
        'flux_rope': np.array([-1, -0.5, 0, 0.5, 1, 0.5, 0, -0.5, -1]) / 3.0,
        'decay': np.array([1, 0.8, 0.6, 0.4, 0.2, 0.1, 0.05]) / 3.15,
        'enhancement': np.array([0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 1]) / 3.15,
        'oscillation': np.array([-1, 0, 1, 0, -1, 0, 1]) / 2.0,
        'pulse': np.array([0, 0.2, 0.6, 1.0, 0.6, 0.2, 0]),
        'gradient': np.array([-1, -0.5, 0, 0.5, 1]) / 2.0,
        'smoothing': np.array([0.1, 0.2, 0.4, 0.6, 0.4, 0.2, 0.1]) / 2.0,
    }
    
    if filter_type not in kernels:
        available_filters = list(kernels.keys())
        raise ValueError(f"Unknown filter type '{filter_type}'. Available filters: {available_filters}")
    
    return kernels[filter_type]

def get_causal_kernels():
    """
    Updated kernel definitions optimized for causal convolution.
    
    These kernels are designed to detect patterns using only past information.
    The interpretation changes slightly - we're looking for patterns that 
    indicate a process has just completed or is in progress.
    """
    kernels = {
        # Storm onset detection - looks for recent rapid changes
        'storm_onset_short': np.array([1, 0.5, 0, -0.5, -1]) / 2.0,  # Reversed pattern
        
        # Main phase development - looks for sustained changes
        'storm_onset_long': np.array([1, 1, 1, 0.5, 0.2, 0, -0.2, -0.5, -1, -1, -1]) / 6.0,
        
        # Sharp compression - recent spike
        'compression_short': np.array([0.2, 1.0, 0.2]) / 1.4,
        
        # Extended compression - sustained high values
        'compression_long': np.array([0.1, 0.2, 0.4, 0.8, 1.0, 1.0, 0.8, 0.4, 0.2, 0.1]) / 5.0,
        
        # Enhancement detection - gradual increase pattern
        'enhancement_short': np.array([1.0, 0.6, 0.3, 0.1]) / 2.0,
        
        # Stream development - long-term enhancement
        'enhancement_long': np.array([1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.15, 0.1, 0.05]) / 6.0,
    }
    
    return kernels

@jit(nopython=True)
def causal_convolution(signal, kernel):
    """
    Perform causal convolution with NaN propagation.
    
    Kernel design convention: kernel[0]=oldest, kernel[-1]=newest (standard conv)
    Implementation: Reverses kernel so kernel[0] applies to signal[t] (causal)
    
    At time t: result[t] = sum(reversed_kernel[j] * signal[t-j])
    
    Example with kernel=[0.1, 0.5, 1.0] at time t=10:
    Reversed: [1.0, 0.5, 0.1]
    result[10] = 1.0*signal[10] + 0.5*signal[9] + 0.1*signal[8]
                 ↑ newest (kernel[-1])  ↑ middle   ↑ oldest (kernel[0])
    
    NaN behavior:
    - If ANY value in window is NaN → result is NaN
    - Cold start: First len(kernel)-1 points use partial kernel
    """
    n_signal = len(signal)
    n_kernel = len(kernel)
    result = np.full(n_signal, np.nan)
    
    # Reverse kernel: kernel[0] now applies to current time
    causal_kernel = kernel[::-1]
    
    for i in range(n_signal):
        conv_sum = 0.0
        has_nan = False
        
        # Apply reversed kernel
        for j in range(n_kernel):
            past_index = i - j  # j=0 is current, j=1 is one step back, etc.
            
            if past_index >= 0:
                value = signal[past_index]
                
                if np.isnan(value):
                    has_nan = True
                    break
                
                conv_sum += value * causal_kernel[j]
        
        if not has_nan:
            result[i] = conv_sum
    
    return result

def normalize_convolution_output(convolved: np.ndarray, filter_type: str) -> np.ndarray:
    """
    Normalize convolution output based on the filter type and expected range.
    
    Args:
        convolved: Raw convolution output
        filter_type: Type of filter used
        
    Returns:
        Normalized convolution output
    """
    # Different normalization strategies based on filter physics
    if filter_type in ['storm_onset', 'flux_rope', 'oscillation', 'gradient']:
        # These can be positive or negative - normalize to [-1, 1]
        max_abs = np.max(np.abs(convolved))
        if max_abs > 0:
            normalized = convolved / max_abs
        else:
            normalized = convolved
            
    elif filter_type in ['compression', 'pulse', 'enhancement', 'decay', 'smoothing']:
        # These are mostly positive - normalize to [0, 1] 
        min_val = np.min(convolved)
        max_val = np.max(convolved)
        
        if max_val > min_val:
            normalized = (convolved - min_val) / (max_val - min_val)
        else:
            normalized = np.zeros_like(convolved)
            
    elif filter_type == 'shock':
        # Shock detection can be very sharp - clip extreme values
        normalized = np.clip(convolved, -2, 2)
        
    else:
        # Default: standardize (zero mean, unit variance)
        std_val = np.std(convolved)
        if std_val > 0:
            normalized = (convolved - np.mean(convolved)) / std_val
        else:
            normalized = convolved - np.mean(convolved)
    
    return normalized


def add_time_series_decomposition(df: pd.DataFrame,
                                 column_name: str,
                                 components: List[str] = ['seasonal', 'trend', 'resid'],
                                 model: str = 'additive',
                                 period: Optional[int] = None,
                                 extrapolate_trend: Union[str, int] = 'freq') -> pd.DataFrame:
    """
    Add time series decomposition components (seasonal, trend, residual) to dataframe.
    
    For space weather data, this captures:
    - Seasonal: Solar cycle effects, equinox patterns, annual variations
    - Trend: Long-term changes in solar activity, instrumental drift
    - Residual: Anomalous events, storms, short-term variability
    
    Args:
        df: Input dataframe
        column_name: Name of column to decompose
        components: List of components to add ['seasonal', 'trend', 'resid']
        model: 'additive' or 'multiplicative' decomposition
        period: Periodicity for seasonal decomposition (if None, auto-detect)
        extrapolate_trend: How to handle trend extrapolation at boundaries
        
    Returns:
        DataFrame with new decomposition columns
        
    Example:
        # Add all components
        df = add_time_series_decomposition(df, "BZ, nT (GSM)_lag_4h", 
                                         components=['seasonal', 'trend', 'resid'])
        
        # Add only residuals (anomaly detection)
        df = add_time_series_decomposition(df, "SW Plasma Speed, km/s_lag_6h",
                                         components=['resid'])
    """
    df_result = df.copy()
    
    # Validate inputs
    if column_name not in df.columns:
        warnings.warn(f"Column '{column_name}' not found in dataframe. Skipping decomposition.")
        return df_result
    
    valid_components = ['seasonal', 'trend', 'resid']
    components = [c for c in components if c in valid_components]
    
    if not components:
        warnings.warn("No valid components specified. Choose from: seasonal, trend, resid")
        return df_result
    
    try:
        # Extract data series
        data_series = df_result[column_name].copy()
        
        # Handle missing values by interpolation (decomposition requires complete series)
        if data_series.isna().any():
            # Forward fill, backward fill, then linear interpolation for any remaining gaps
            data_series = data_series.ffill().bfill()
            if data_series.isna().any():
                data_series = data_series.interpolate(method='linear')
            
            # If still missing values, fill with mean
            if data_series.isna().any():
                data_series = data_series.fillna(data_series.mean())
        
        # Auto-detect period if not specified
        if period is None:
            period = estimate_periodicity(data_series)
        
        # Ensure minimum data length for decomposition
        min_length = 2 * period if period else 20
        if len(data_series) < min_length:
            warnings.warn(f"Insufficient data for decomposition. Need at least {min_length} points, got {len(data_series)}")
            return df_result
        
        # Perform decomposition
        decomposition = seasonal_decompose(
            data_series, 
            model=model, 
            period=period,
            extrapolate_trend=extrapolate_trend
        )
        
        # Clean column name for feature naming
        # col_clean = clean_column_name(column_name)
        
        # Add requested components
        if 'seasonal' in components:
            seasonal_col = f"{column_name}_seasonal"
            df_result[seasonal_col] = decomposition.seasonal.values
            
        if 'trend' in components:
            trend_col = f"{column_name}_trend"
            df_result[trend_col] = decomposition.trend.values
            
        if 'resid' in components:
            resid_col = f"{column_name}_resid"
            df_result[resid_col] = decomposition.resid.values
            
        # Handle any remaining NaN values in decomposition results
        new_cols = [col for col in df_result.columns if col not in df.columns]
        for col in new_cols:
            if df_result[col].isna().any():
                df_result[col] = df_result[col].fillna(0)
        
        print(f"Successfully decomposed '{column_name}' into: {', '.join([f'{column_name}_{c}' for c in components])}")
        
    except Exception as e:
        warnings.warn(f"Decomposition failed for '{column_name}': {e}")
        return df_result
    
    return df_result


def estimate_periodicity(series: pd.Series, max_period: int = None) -> int:
    """
    Estimate the dominant periodicity in a time series for space weather data.
    
    Args:
        series: Time series data
        max_period: Maximum period to consider (default: len(series)//4)
        
    Returns:
        Estimated period in hours
    """
    if max_period is None:
        max_period = min(len(series) // 4, 8760)  # Max 1 year for hourly data
    
    # Space weather typical periods (in hours for hourly data)
    typical_periods = [
        24,     # Daily (diurnal variations)
        168,    # Weekly (7 days)
        672,    # ~Monthly (28 days, solar rotation)
        2160,   # ~3 months (seasonal)
        4380,   # ~6 months (semi-annual)
        8760    # Annual
    ]
    
    # Filter periods that make sense for data length
    valid_periods = [p for p in typical_periods if p <= max_period and p <= len(series) // 3]
    
    if not valid_periods:
        # Fallback: use a reasonable default based on data length
        return max(24, min(len(series) // 10, 168))
    
    # For space weather, solar rotation period (~27 days = 672 hours) is often dominant
    # Choose the largest reasonable period that fits the data
    return max(valid_periods)