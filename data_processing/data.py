
import pandas as pd
import numpy as np
from data_processing.constants import *

from typing import Tuple, List, Union
from datetime import datetime

import warnings

def read_data(format_path: str, lst_path: str):
    column_names = pd.read_fwf(format_path)
    column_names = column_names["FORMAT OF THE SUBSETTED FILE"].values[1:]
    
    # Extract column name (between first and last space)
    parsed_names = []
    for col in column_names:
        col = col.strip()
        # Remove number prefix (before first space)
        col = col[col.find(" ")+1:]
        # Remove format suffix (after last space)
        col = col[:col.rfind(" ")].strip()
        parsed_names.append(col)
    
    column_names = parsed_names

    df = pd.read_csv(lst_path, header=None, sep=r'\s+', engine='python')
    column_mapper = {i: name for i, name in enumerate(column_names)}
    df.rename(columns=column_mapper, inplace=True)

    good_columns = [name for name in column_names if name not in BAD_COLUMNS]
    x = df[good_columns].copy()
    
    return {
        "x": x,
        "column_names": column_names,
        "good_columns": good_columns,
    }

def process_dataset(dataset: pd.DataFrame, replace_outliers=True, print_max_values=False):

    if replace_outliers:
        print("Replacing outliers...")
        for column, max_value in COLUMNS_WITH_THRESHOLD:
            dataset = replace_outliers_with_mean(dataset, column, max_value)
        print("Done replacing outliers with mean.")
    else:
        print("Replacing outliers...")
        for column, max_value in COLUMNS_WITH_THRESHOLD:
            dataset = replace_outliers_with_nan(dataset, column, max_value)
        print("Done replacing outliers with nan.")

    # check if there are any strange outliers
    if print_max_values:
        for column in dataset.columns:
            print(column, " - Max value: ", dataset[column].max())
        # done checking

    dataset = create_time_column(dataset, FULL_DATE_COLUMN)

    # return correct columns
    dataset = dataset[CANDIDATE_COLUMNS].copy()

    # dataset = convert_angular_features(dataset, ANGULAR_TRIGONOMETRIC, "degrees", True)
    dataset.dropna(inplace=True)
    
    return dataset.copy()

def replace_outliers_with_mean(df: pd.DataFrame, column_name: str, max_value: float = 9999, window_hours: int = 5, 
                               min_valid_points: int = 3, print_count: bool=False, storm_threshold: int = -50):
    """
    Replaces outliers with local window mean only if sufficient nearby valid data exists.
    Otherwise, outliers become NaN and are removed later.
    
    Parameters:
    - df: pandas DataFrame
    - column_name: str, the name of the column to check
    - max_value: numeric, the threshold value for detecting outliers (default 9999)
    - window_hours: int, size of window for local mean (±window_hours)
    - min_valid_points: int, minimum valid points in window required for imputation (default 3)
    - print_count: bool, whether to print statistics
    - storm_threshold: int, DST threshold for additional analysis (default -50)
    
    Returns:
    - df: pandas DataFrame with outliers replaced by local mean or NaN
    """
    column_data = df[column_name].copy()
    
    if print_count:
        replacement_counter = (column_data >= max_value)
        replacement_percent = sum(replacement_counter) / len(replacement_counter)
        
        dst_rows_affected = (df[column_name] >= max_value) & (df[DST_COLUMN] <= storm_threshold)
        dst_rows_affected_percent = sum(dst_rows_affected) / len(dst_rows_affected)
        
        print(f"Column: {column_name}")
        print(f"  Percent needing replacement: {replacement_percent:.2%}")
        print(f"  DST rows affected: {dst_rows_affected_percent:.2%}")
    
    # Find outlier positions
    outlier_mask = column_data >= max_value
    
    if not outlier_mask.any():
        return df  # No outliers found
    
    # Set outliers to NaN temporarily for rolling calculation
    clean_data = column_data.copy()
    clean_data[outlier_mask] = float('nan')
    
    # Compute rolling mean with minimum valid points requirement
    window_size = window_hours * 2 + 1
    rolling_mean = clean_data.rolling(
        window=window_size, 
        center=True, 
        min_periods=min_valid_points  # KEY: require minimum valid points
    ).mean()
    
    # Replace outliers with rolling mean (which will be NaN if insufficient data)
    df[column_name] = column_data.where(~outlier_mask, rolling_mean)
    
    return df


def replace_outliers_with_nan(df, column_name, max_value=9999):
    """
    Replaces values greater than max_value in the specified column with NaN.
    This helps in counting NaN values per row later.
    
    Parameters:
    - df: pandas DataFrame
    - column_name: str, the name of the column to check
    - max_value: numeric, the threshold value for detecting outliers (default 9999)
    - print_count: bool, whether to print the percentage of replaced values
    - storm_threshold: int, DST threshold for additional analysis (default -50)
    
    Returns:
    - df: pandas DataFrame with outliers replaced by NaN
    """
    column_data = df[column_name]
    # Replace values greater than max_value with NaN
    df[column_name] = column_data.apply(lambda x: float('nan') if x >= max_value else x)
    
    return df

def create_time_column(df: pd.DataFrame, column_name: str) -> pd.DataFrame:

    df["DOY"] = df["DOY"].astype(str).str.zfill(3)  # Pad DOY to 3 digits if needed
    df["Hour"] = df["Hour"].astype(str).str.zfill(2)  # Pad Hour to 2 digits if needed

    # Combine YEAR, DOY, and Hour into datetime
    df[column_name] = pd.to_datetime(df["YEAR"].astype(str) + df["DOY"] + df["Hour"], format="%Y%j%H", errors='coerce')

    return df


def temporal_split(df: pd.DataFrame, 
                  time_col: str, 
                  split_date: Union[str, datetime, pd.Timestamp]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split a dataframe into before and after a specified date.
    
    This is a basic forward-time split that preserves temporal order,
    ensuring no data leakage for time series validation.
    
    Args:
        df (pd.DataFrame): Input dataframe with time series data
        time_col (str): Name of the datetime column
        split_date (Union[str, datetime, pd.Timestamp]): Date to split on
            Can be string (e.g., '2020-01-01'), datetime object, or pandas Timestamp
    
    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: (before_df, after_df)
            before_df: All data strictly before split_date
            after_df: All data from split_date onwards
    
    Example:
        >>> df = pd.DataFrame({
        ...     'datetime': pd.date_range('2020-01-01', periods=100, freq='H'),
        ...     'value': np.random.randn(100)
        ... })
        >>> train_df, test_df = temporal_split(df, 'datetime', '2020-01-03')
        >>> print(f"Train: {len(train_df)}, Test: {len(test_df)}")
    """
    # Input validation
    if time_col not in df.columns:
        raise ValueError(f"Column '{time_col}' not found in dataframe. Available columns: {list(df.columns)}")
    
    if df.empty:
        raise ValueError("Input dataframe is empty")
    
    # Convert split_date to pandas Timestamp for consistent comparison
    if isinstance(split_date, str):
        split_date = pd.to_datetime(split_date)
    elif isinstance(split_date, datetime):
        split_date = pd.Timestamp(split_date)
    
    # Ensure the time column is datetime type
    if not pd.api.types.is_datetime64_any_dtype(df[time_col]):
        try:
            df = df.copy()
            df[time_col] = pd.to_datetime(df[time_col])
        except Exception as e:
            raise ValueError(f"Could not convert '{time_col}' to datetime: {e}")
    
    # Validate split_date is within data range
    min_date = df[time_col].min()
    max_date = df[time_col].max()
    
    if split_date <= min_date:
        warnings.warn(f"Split date {split_date} is before or at minimum date {min_date}. "
                     f"'Before' dataframe will be empty.")
    
    if split_date >= max_date:
        warnings.warn(f"Split date {split_date} is after or at maximum date {max_date}. "
                     f"'After' dataframe will be empty.")
    
    # Create masks for splitting
    before_mask = df[time_col] < split_date
    after_mask = df[time_col] >= split_date
    
    # Split the dataframe
    before_df = df[before_mask].copy()
    after_df = df[after_mask].copy()
    
    # Reset indices to avoid confusion
    before_df.reset_index(drop=True, inplace=True)
    after_df.reset_index(drop=True, inplace=True)
    
    return before_df, after_df

def get_recommended_cross_correlations() -> List[Tuple[str, str, int, str]]:
    """
    Get physics-motivated cross-correlation pairs for space weather forecasting.
    
    Returns:
        List of tuples: (col1, col2, window_hours, method)
    
    Physics justification for each pair:
    """
    recommendations = [
        # Storm main phase coupling
        ("SW Plasma Speed, km/s", "BZ, nT (GSM)", 12, "pearson"),
        # Reasoning: Sustained fast southward IMF drives intense storms
        
        # Compression and magnetic structure  
        ("SW Proton Density, N/cm^3", "Scalar B, nT", 6, "pearson"),
        # Reasoning: Compressed plasma and enhanced magnetic field indicate shocks/CMEs
        
        # Dynamic pressure and magnetic coupling
        ("SW Plasma Speed, km/s", "SW Proton Density, N/cm^3", 8, "pearson"), 
        # Reasoning: Combined for dynamic pressure, identifies stream interactions
        
        # Fast coupling for substorm triggering
        ("SW Plasma Speed, km/s", "BZ, nT (GSM)", 3, "instantaneous"),
        # Reasoning: Rapid IMF changes in fast streams trigger substorms
        
        # Magnetic field structure coherence
        ("Scalar B, nT", "BZ, nT (GSM)", 18, "pearson"),
        # Reasoning: Coherent magnetic structures (flux ropes) have sustained Bz
        
        # Density-magnetic field anti-correlation (rarefaction detection)
        ("SW Proton Density, N/cm^3", "BZ, nT (GSM)", 12, "spearman"),
        # Reasoning: Magnetic flux ropes often show density depletion with strong fields
    ]
    
    return recommendations


def convert_angular_features(df: pd.DataFrame, 
                           angular_columns: List[str],
                           angle_unit: str = 'degrees',
                           drop_original: bool = True) -> pd.DataFrame:
    """
    Convert angular features to trigonometric sin/cos components for space weather data.
    
    This function handles the circular nature of angular parameters by converting them
    to sin/cos components, which preserves the circular boundary conditions
    (e.g., 359° ≈ 1°) that are important for magnetic field directions.
    
    Args:
        df: Input dataframe
        angular_columns: List of column names containing angular data
        angle_unit: Unit of input angles ('degrees' or 'radians')
        drop_original: Whether to remove original angular columns after conversion
    
    Returns:
        DataFrame with original angular columns replaced by sin/cos components
        
    Example:
        Original column: "Long. Angle of B (GSE)"
        New columns: "Long_Angle_of_B_GSE_sin", "Long_Angle_of_B_GSE_cos"
        
    Physics motivation:
        - Preserves circular nature: sin(359°) ≈ sin(1°), cos(359°) ≈ cos(1°)
        - Eliminates discontinuities at angle boundaries
        - Maintains full directional information in 2D representation
        - Works well with machine learning algorithms that assume linear features
    """
    df_result = df.copy()
    
    # Track which columns were processed
    processed_columns = []
    skipped_columns = []
    
    for col in angular_columns:
        if col not in df.columns:
            skipped_columns.append(col)
            continue
            
        try:
            # Get the angular data
            angular_data = df_result[col].copy()
            
            # Handle missing values - forward fill, then backward fill, then use 0
            angular_data = angular_data.ffill().bfill().fillna(0.0)
            
            # Convert to radians if necessary
            if angle_unit.lower() == 'degrees':
                angular_rad = np.deg2rad(angular_data)
            elif angle_unit.lower() == 'radians':
                angular_rad = angular_data
            else:
                raise ValueError(f"Unknown angle unit: {angle_unit}. Use 'degrees' or 'radians'.")
            
            # Compute sin and cos components
            sin_component = np.sin(angular_rad)
            cos_component = np.cos(angular_rad)
            
            # Handle any numerical issues (inf, nan)
            sin_component = np.nan_to_num(sin_component, nan=0.0, posinf=0.0, neginf=0.0)
            cos_component = np.nan_to_num(cos_component, nan=0.0, posinf=0.0, neginf=0.0)
            
            # Create new column names
            # Clean the original column name for safe feature naming
            # col_clean = clean_column_name(col)
            sin_col_name = f"{col}_sin"
            cos_col_name = f"{col}_cos"
            
            # Add new columns
            df_result[sin_col_name] = sin_component
            df_result[cos_col_name] = cos_component
            
            # Remove original column if requested
            if drop_original:
                df_result.drop(columns=[col], inplace=True)
            
            processed_columns.append(col)
            
        except Exception as e:
            warnings.warn(f"Failed to convert angular column '{col}': {e}")
            skipped_columns.append(col)
    
    # Print summary
    if processed_columns:
        print(f"Successfully converted {len(processed_columns)} angular columns:")
        for col in processed_columns:
            # col_clean = clean_column_name(col)
            print(f"  {col} → {col}_sin, {col}_cos")
    
    if skipped_columns:
        print(f"Skipped {len(skipped_columns)} columns (not found or conversion failed):")
        for col in skipped_columns:
            print(f"  {col}")
    
    return df_result


def clean_column_name(column_name: str) -> str:
    """
    Clean column names for safe feature naming.
    
    Args:
        column_name: Original column name
        
    Returns:
        Cleaned column name suitable for use in feature names
    """
    # Replace common problematic characters
    cleaned = column_name.replace(' ', '_')
    cleaned = cleaned.replace(',', '')
    cleaned = cleaned.replace('(', '')
    cleaned = cleaned.replace(')', '')
    cleaned = cleaned.replace('.', '')
    cleaned = cleaned.replace('-', '_')
    cleaned = cleaned.replace('/', '_')
    
    # Remove multiple consecutive underscores
    while '__' in cleaned:
        cleaned = cleaned.replace('__', '_')
    
    # Remove leading/trailing underscores
    cleaned = cleaned.strip('_')
    
    return cleaned