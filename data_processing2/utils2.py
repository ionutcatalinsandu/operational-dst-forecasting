import pandas as pd
from .constants import DST_COLUMN
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Union, Dict, List, Tuple
import warnings

def display_basic_stats(data: pd.DataFrame, threshold: float):

    storms = len(data[data[DST_COLUMN] <= threshold])
    rows_with_nan = data.isna().any(axis=1).sum()
    print(f"Full data: > Len: {len(data)}, > Storms: {storms}; {storms/len(data):.2%}, > Rows with NaN: {rows_with_nan} ({rows_with_nan/len(data):.2%})")

def filter_rows_by_completeness(
    df: pd.DataFrame,
    min_data_percent: float = 70.0,
    exclude_cols: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Filter dataframe rows based on data completeness threshold.
    
    Keeps rows where at least min_data_percent% of values are non-NaN.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    min_data_percent : float
        Minimum percentage of non-NaN values required per row (0-100)
        Example: 70.0 means "keep rows with at least 70% data present"
    exclude_cols : list, optional
        Columns to exclude from completeness calculation
        (e.g., timestamps, target variables, identifiers)
        
    Returns:
    --------
    pd.DataFrame
        Filtered dataframe
        
    Example:
    --------
    df_clean = filter_rows_by_completeness(
        df, 
        min_data_percent=80.0,
        exclude_cols=['Epoch', 'DST']
    )
    """
    
    # Select columns to check
    if exclude_cols is None:
        cols_to_check = df.columns.tolist()
    else:
        cols_to_check = [col for col in df.columns if col not in exclude_cols]
    
    if len(cols_to_check) == 0:
        warnings.warn("No columns to check after exclusions. Returning original dataframe.")
        return df.copy()
    
    # Compute completeness per row
    n_cols = len(cols_to_check)
    n_valid_per_row = df[cols_to_check].notna().sum(axis=1)
    completeness_percent = (n_valid_per_row / n_cols) * 100
    
    # Filter: keep rows meeting threshold
    mask = completeness_percent >= min_data_percent
    df_filtered = df[mask].copy()
    
    # Print summary
    n_removed = (~mask).sum()
    print(f"Completeness filter (≥{min_data_percent}%): "
          f"kept {len(df_filtered):,}/{len(df):,} rows "
          f"({len(df_filtered)/len(df)*100:.1f}%), "
          f"removed {n_removed:,}")
    
    return df_filtered


def analyze_row_completeness(
    df: pd.DataFrame,
    exclude_cols: Optional[List[str]] = None,
    bins: int = 20,
    plot_hist: bool = False,
) -> pd.Series:
    """
    Analyze data completeness distribution across rows.
    
    Useful for deciding an appropriate min_data_percent threshold.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    exclude_cols : list, optional
        Columns to exclude from analysis
    bins : int
        Number of bins for histogram display
        
    Returns:
    --------
    pd.Series
        Completeness percentage per row
        
    Example:
    --------
    completeness = analyze_row_completeness(df, exclude_cols=['Epoch', 'DST'])
    print(f"Median completeness: {completeness.median():.1f}%")
    print(f"25th percentile: {completeness.quantile(0.25):.1f}%")
    """
    
    # Select columns to check
    if exclude_cols is None:
        cols_to_check = df.columns
    else:
        cols_to_check = [col for col in df.columns if col not in exclude_cols]
    
    # Compute completeness per row
    n_cols = len(cols_to_check)
    n_valid_per_row = df[cols_to_check].isna().sum(axis=1)
    completeness_percent = (n_valid_per_row / n_cols) * 100
    
    # Print summary statistics
    print("Row Completeness Analysis: ", f"  Mean:   {completeness_percent.mean():.1f}%", f"  Median: {completeness_percent.median():.1f}%", f"  Std:    {completeness_percent.std():.1f}%")
    if plot_hist:
        # Plot histogram
        import matplotlib.pyplot as plt
        
        plt.figure(figsize=(10, 5))
        plt.hist(completeness_percent, bins=bins, edgecolor='black', alpha=0.7)
        plt.xlabel('Data Completeness (%)')
        plt.ylabel('Number of Rows')
        plt.title(f'Distribution of Row Completeness (n={len(df):,} rows, {n_cols} columns)')
        plt.grid(True, alpha=0.3)
        plt.axvline(completeness_percent.median(), color='red', linestyle='--', 
                    label=f'Median: {completeness_percent.median():.1f}%')
        plt.legend()
        plt.tight_layout()
        plt.show()
    
    return completeness_percent

def analyze_nan_distribution(
    df: pd.DataFrame,
    n_batches: int = 10,
    figsize: tuple = (15, 10),
    title: Optional[str] = None
) -> pd.DataFrame:
    """
    Analyze NaN distribution across rows with temporal batching.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe to analyze
    n_batches : int
        Number of temporal batches to split dataset into
    figsize : tuple
        Figure size for subplots
    title : str, optional
        Overall title for the figure
        
    Returns:
    --------
    pd.DataFrame
        Summary statistics per batch (batch_id, start_idx, end_idx, 
        mean_nan_pct, median_nan_pct, max_nan_pct, n_rows)
    """
    
    # Compute NaN percentage per row
    nan_pct_per_row = df.isna().sum(axis=1) / len(df.columns) * 100
    
    # Split into batches
    n_rows = len(df)
    batch_size = n_rows // n_batches
    
    # Create figure with subplots
    n_cols = min(3, n_batches)  # Max 3 columns
    n_rows_plot = (n_batches + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows_plot, n_cols, figsize=figsize)
    if n_batches == 1:
        axes = np.array([axes])
    axes = axes.flatten()
    
    # Summary statistics storage
    batch_stats = []
    
    for i in range(n_batches):
        start_idx = i * batch_size
        end_idx = (i + 1) * batch_size if i < n_batches - 1 else n_rows
        
        batch_data = nan_pct_per_row.iloc[start_idx:end_idx]
        
        # Plot histogram
        ax = axes[i]
        ax.hist(batch_data, bins=50, edgecolor='black', alpha=0.7)
        ax.set_xlabel('NaN Percentage per Row (%)')
        ax.set_ylabel('Frequency')
        ax.set_title(f'Batch {i+1}: Rows {start_idx}-{end_idx}')
        ax.grid(True, alpha=0.3)
        
        # Add statistics text
        stats_text = (
            f'Mean: {batch_data.mean():.1f}%\n'
            f'Median: {batch_data.median():.1f}%\n'
            f'Max: {batch_data.max():.1f}%'
        )
        ax.text(0.98, 0.97, stats_text,
                transform=ax.transAxes,
                verticalalignment='top',
                horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
                fontsize=9)
        
        # Store batch statistics
        batch_stats.append({
            'batch_id': i + 1,
            'start_idx': start_idx,
            'end_idx': end_idx,
            'n_rows': len(batch_data),
            'mean_nan_pct': batch_data.mean(),
            'median_nan_pct': batch_data.median(),
            'max_nan_pct': batch_data.max(),
            'min_nan_pct': batch_data.min(),
            'std_nan_pct': batch_data.std()
        })
    
    # Hide unused subplots
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')
    
    # Overall title
    if title:
        fig.suptitle(title, fontsize=14, y=0.995)
    else:
        fig.suptitle(
            f'NaN Distribution Analysis ({n_batches} batches, {n_rows} total rows)',
            fontsize=14, y=0.995
        )
    
    plt.tight_layout()
    plt.show()
    
    # Return summary as DataFrame
    return pd.DataFrame(batch_stats)

def read_solar_phases(filepath='solar_phase.csv'):
    """
    Read solar cycle phase data from CSV.
    Adjusts end dates to include the full day (23:59:59).
    """
    df = pd.read_csv(filepath)
    df['start_date'] = pd.to_datetime(df['start_date'])
    # Add 23:59:59 to end dates to include the full day
    df['end_date'] = pd.to_datetime(df['end_date']) + pd.Timedelta(hours=23, minutes=59, seconds=59)
    return df


def add_solar_cycle_features(df, solar_phases_df, date_column='datetime', 
                             cap_date='2023-01-01', missing_rate=0.15):
    """
    Add solar cycle phase and cycle number to dataframe.
    Now properly handles full days without gaps.
    """
    df = df.copy()
    
    if not pd.api.types.is_datetime64_any_dtype(df[date_column]):
        df[date_column] = pd.to_datetime(df[date_column])
    
    df['solar_phase'] = None
    df['solar_cycle'] = np.nan
    
    cap_datetime = pd.to_datetime(cap_date)
    
    # Map each row to its solar cycle phase and number
    for idx, row in solar_phases_df.iterrows():
        # For the last row, extend to cover all future dates
        if idx == len(solar_phases_df) - 1:
            mask = (df[date_column] >= row['start_date'])
        else:
            mask = (df[date_column] >= row['start_date']) & (df[date_column] <= row['end_date'])
        
        # Always set solar_cycle (no cap)
        df.loc[mask, 'solar_cycle'] = row['cycle_count']
        
        # Set solar_phase with cap
        if row['end_date'] > cap_datetime:
            phase_mask = mask & (df[date_column] < cap_datetime)
        else:
            phase_mask = mask
            
        df.loc[phase_mask, 'solar_phase'] = row['phase']
    
    # Convert solar_cycle to integer
    df['solar_cycle'] = df['solar_cycle'].astype('Int64')
    
    # Randomly mask phase values
    if missing_rate > 0:
        valid_phase_indices = df[df['solar_phase'].notna()].index
        
        if len(valid_phase_indices) > 0:
            n_missing = int(len(valid_phase_indices) * missing_rate)
            missing_indices = np.random.choice(
                valid_phase_indices, 
                size=n_missing, 
                replace=False
            )
            df.loc[missing_indices, 'solar_phase'] = np.nan
    
    return df