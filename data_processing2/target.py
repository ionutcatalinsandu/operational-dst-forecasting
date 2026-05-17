import pandas as pd
import numpy as np
from scipy import stats
from tqdm import tqdm

def _compute_adaptive_severity(dst_values, window_hours, causal=True, min_periods=10):
    """
    Compute adaptive severity scores efficiently using simple Python loops.
    
    Parameters:
    -----------
    dst_values : np.array
        Array of DST values
    window_hours : int
        Size of rolling window in hours
    causal : bool
        If True, use only past values; if False, use centered window
    min_periods : int
        Minimum number of valid observations in window
        
    Returns:
    --------
    np.array
        Array of severity scores (Z-scores)
    """
    n = len(dst_values)
    severity_scores = np.full(n, np.nan)
    
    for i in tqdm(range(n)):
        # Skip if current value is NaN
        if np.isnan(dst_values[i]):
            continue
            
        # Define window boundaries
        if causal:
            # Causal: only past values [i-window+1, i]
            start_idx = max(0, i - window_hours + 1)
            end_idx = i + 1
        else:
            # Centered: past and future [i-window/2, i+window/2]
            half_window = window_hours // 2
            start_idx = max(0, i - half_window)
            end_idx = min(n, i + half_window + 1)
        
        # Extract valid (non-NaN) values from window
        window_values = dst_values[start_idx:end_idx]
        window_values = window_values[~np.isnan(window_values)]
        
        # Need minimum number of valid observations
        if len(window_values) >= min_periods:
            current_value = dst_values[i]
            
            # Calculate empirical CDF: P(X <= current_value)
            percentile = np.sum(window_values <= current_value) / len(window_values)
            
            # Clip to avoid extreme values and convert to Z-score
            percentile = np.clip(percentile, 0.001, 0.999)
            severity_scores[i] = stats.norm.ppf(percentile)
    
    return severity_scores


def transform_dst_adaptive_severity(df, dst_col, window_days=90, target_col=None, causal=True):
    """
    Transform DST to adaptive severity percentiles (Z-scores of local distribution).
    
    Physics: Solar cycle changes baseline magnetospheric state, but relative 
    disturbance patterns remain consistent. This transformation is immune to 
    distribution shift by normalizing against local conditions.
    
    Formula: Severity(t) = Φ⁻¹(F_local(DST(t)))
    where F_local is empirical CDF over rolling window
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe with datetime index
    dst_col : str
        Column name containing DST values
    window_days : int, default=90
        Rolling window size in days for local distribution
    target_col : str, optional
        Output column name. If None, replaces original column
    causal : bool, default=True
        If True, use only past values (causal window: [t-window, t])
        If False, use centered window (non-causal: [t-window/2, t+window/2])
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with transformed DST column
    """
    df_copy = df.copy()
    
    if target_col is None:
        target_col = dst_col
    
    window_hours = window_days * 24
    dst_values = df_copy[dst_col].values.astype(np.float64)
    
    # Call optimized function for fast computation
    severity_scores = _compute_adaptive_severity(
        dst_values, 
        window_hours, 
        causal=causal, 
        min_periods=10
    )
    
    df_copy[target_col] = severity_scores
    
    return df_copy


def convert_dst_to_categories(df, dst_col, category_thresholds, target_col=None):
    """
    Convert DST values to categorical bins based on storm classification thresholds.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe containing DST values
    dst_col : str
        Column name containing DST values in nT
    category_thresholds : dict
        Dictionary mapping category names to DST threshold values (in nT).
        Should be ordered from most negative (severe) to most positive (quiet).
        Example: {
            'severe_storm': -200,
            'intense_storm': -100, 
            'moderate_storm': -50,
            'quiet': 0,
            'very_quiet': 50
        }
    target_col : str, optional
        Output column name. If None, replaces the original column.
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with DST values converted to categorical labels
        
    Examples:
    ---------
    # Standard geomagnetic storm classification
    storm_categories = {
        'severe_storm': -200,
        'intense_storm': -100,
        'moderate_storm': -50,
        'quiet': 0,
        'very_quiet': 50
    }
    
    df_categorical = convert_dst_to_categories(df, 'DST', storm_categories)
    
    # Custom classification
    custom_categories = {
        'extreme': -150,
        'high': -75,
        'normal': -25,
        'calm': 25
    }
    
    df_custom = convert_dst_to_categories(df, 'DST', custom_categories, 'DST_category')
    """
    
    df_copy = df.copy()
    
    if target_col is None:
        target_col = dst_col
    
    # Extract category names and thresholds
    categories = list(category_thresholds.keys())
    thresholds = list(category_thresholds.values())
    
    # Validate that thresholds are in ascending order
    if thresholds != sorted(thresholds):
        raise ValueError("Category thresholds must be in ascending order (most negative to most positive)")
    
    # Get DST values
    dst_values = df_copy[dst_col].values
    
    # Create categorical labels
    categorical_labels = np.full(len(dst_values), None, dtype=object)
    
    for i, dst_value in enumerate(dst_values):
        if pd.isna(dst_value):
            categorical_labels[i] = np.nan
            continue
            
        # Find appropriate category
        # Start from the most severe (most negative) threshold
        assigned = False
        for j, threshold in enumerate(thresholds):
            if dst_value <= threshold:
                categorical_labels[i] = categories[j]
                assigned = True
                break
        
        # If value is above all thresholds, assign to the last (most positive) category
        if not assigned:
            categorical_labels[i] = categories[-1]
    
    # Convert to pandas categorical for efficiency and ordering
    df_copy[target_col] = pd.Categorical(
        categorical_labels, 
        categories=categories, 
        ordered=True
    )
    
    return df_copy


def get_standard_storm_categories():
    """
    Return standard geomagnetic storm classification categories.
    
    Based on NOAA Space Weather Scales and common research classifications.
    
    Returns:
    --------
    dict
        Standard storm category thresholds
    """
    return {
        'severe_storm': -200,      # G4-G5 level storms
        'intense_storm': -100,     # G3 level storms  
        'moderate_storm': -50,     # G1-G2 level storms
        'quiet': 0,                # Normal conditions
        'very_quiet': 50           # Exceptionally calm conditions
    }

def analyze_category_distribution(df, category_col):
    """
    Analyze the distribution of categorical storm classifications.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with categorical storm classifications
    category_col : str
        Column name containing categorical labels
        
    Returns:
    --------
    pd.DataFrame
        Summary statistics of category distribution
    """
    
    # Count and percentage distribution
    counts = df[category_col].value_counts(dropna=False)
    percentages = df[category_col].value_counts(normalize=True, dropna=False) * 100
    
    # Create summary dataframe
    summary = pd.DataFrame({
        'count': counts,
        'percentage': percentages.round(2)
    })
    
    # Add cumulative statistics
    summary['cumulative_count'] = counts.cumsum()
    summary['cumulative_percentage'] = percentages.cumsum().round(2)
    
    print("Category Distribution Summary:")
    print("=" * 40)
    print(summary)
    print(f"\nTotal observations: {len(df)}")
    print(f"Missing values: {df[category_col].isna().sum()}")
    
    return summary