import numpy as np
import pandas as pd
from numba import jit
from typing import List

def add_lagged_history(df, columns, n_lags=10):
    """
    Add lagged history columns to DataFrame.
    
    Parameters:
    - df: DataFrame
    - columns: list of column names to compute lags for
    - n_lags: number of lags to compute
    
    Returns:
    - tuple: (modified_df, new_column_names)
    """
    result_df = df.copy()
    new_columns = []
    
    for col in columns:
        series = df[col].values
        n = len(series)
        
        # Pre-allocate with NaN
        lags = np.full((n, n_lags), np.nan)
        
        # Fill valid portions
        for lag in range(min(n_lags, n)):
            lags[lag:, lag] = series[:n-lag]
        
        # Store each row as array
        new_col_name = f"{col}_lags"
        result_df[new_col_name] = [lags[i] for i in range(n)]
        new_columns.append(new_col_name)
    
    return result_df, new_columns


@jit(nopython=True)
def _compute_signal_features_batch(lag_histories):
    """
    Compute signal processing features for a batch of lag histories efficiently.
    
    Parameters:
    - lag_histories: 2D array (n_samples, n_lags)
    
    Returns:
    - 2D array (n_samples, 3) with [max_jump, jump_concentration, trend_strength]
    """
    n_samples = lag_histories.shape[0]
    features = np.empty((n_samples, 3))
    
    for i in range(n_samples):
        lag_history = lag_histories[i]
        
        # If lag history contains NaN, return NaN features
        if np.any(np.isnan(lag_history)):
            features[i] = np.array([np.nan, np.nan, np.nan])
            continue
        
        # 1. Maximum Jump: Largest single change (shock detector)
        diffs = np.abs(lag_history[1:] - lag_history[:-1])
        max_jump = np.max(diffs)
        
        # 2. Jump Concentration: Is change concentrated in few points? (discontinuity detector)
        total_variation = np.sum(diffs)
        jump_concentration = max_jump / total_variation if total_variation > 1e-10 else 0.0
        
        # 3. Trend Strength: Linear correlation with time (systematic evolution detector)
        n_points = len(lag_history)
        time_indices = np.arange(n_points, dtype=np.float64)
        
        # Manual correlation calculation (numba-compatible)
        mean_time = np.mean(time_indices)
        mean_values = np.mean(lag_history)
        
        numerator = 0.0
        time_variance = 0.0
        value_variance = 0.0
        
        for j in range(n_points):
            time_dev = time_indices[j] - mean_time
            value_dev = lag_history[j] - mean_values
            numerator += time_dev * value_dev
            time_variance += time_dev * time_dev
            value_variance += value_dev * value_dev
        
        denominator = np.sqrt(time_variance * value_variance)
        trend_strength = abs(numerator / denominator) if denominator > 1e-10 else 0.0
        
        features[i] = np.array([max_jump, jump_concentration, trend_strength])
    
    return features


def add_signal_features(df, lag_columns):
    """
    Add signal processing feature columns to DataFrame.
    
    Parameters:
    - df: DataFrame containing lag history columns
    - lag_columns: list of lag column names (from add_lagged_history)
    
    Returns:
    - tuple: (modified_df, new_column_names)
    """
    result_df = df.copy()
    new_columns = []
    
    for col in lag_columns:
        # Convert to 2D array - trust pipeline for correct dimensions
        lag_histories = np.stack(df[col].values)
        
        # Compute features in batch
        features = _compute_signal_features_batch(lag_histories)
        
        # Add feature columns
        feature_names = [f'{col}_max_jump', f'{col}_jump_conc', f'{col}_trend_str']
        for j, feat_name in enumerate(feature_names):
            result_df[feat_name] = features[:, j]
            new_columns.append(feat_name)
    
    return result_df, new_columns


def add_lagged_history(df, columns, n_lags=10):
    """
    Add lagged history columns to DataFrame.
    
    Parameters:
    - df: DataFrame
    - columns: list of column names to compute lags for
    - n_lags: number of lags to compute
    
    Returns:
    - tuple: (modified_df, new_column_names)
    """
    result_df = df.copy()
    new_columns = []
    
    for col in columns:
        series = df[col].values
        n = len(series)
        
        # Pre-allocate with NaN
        lags = np.full((n, n_lags), np.nan)
        
        # Fill valid portions
        for lag in range(min(n_lags, n)):
            lags[lag:, lag] = series[:n-lag]
        
        # Store each row as array
        new_col_name = f"{col}_lags"
        result_df[new_col_name] = [lags[i] for i in range(n)]
        new_columns.append(new_col_name)
    
    return result_df, new_columns


def add_takens_embeddings(df: pd.DataFrame, columns: List[str], ms: List[int], taus: List[int]):
    """
    Add Takens embedding columns to DataFrame.
    
    Parameters:
    - df: DataFrame
    - columns: list of column names to compute embeddings for
    - ms: embedding dimensions
    - taus: time delays
    
    Returns:
    - tuple: (modified_df, new_column_names)
    """
    result_df = df.copy()
    new_columns = []
    
    for col, m, tau in zip(columns, ms, taus):
        # Extract values
        values = df[col].values
        
        # Compute embedding
        embedding = _fast_takens_single(values, m, tau)
        
        # Store each row's embedding as array (creates object dtype column)
        new_col_name = f"{col}_embedding"
        result_df[new_col_name] = [embedding[i] for i in range(len(embedding))]
        new_columns.append(new_col_name)
    
    return result_df, new_columns

@jit(nopython=True)
def _fast_takens_single(series, m, tau):
    """
    Fast Takens embedding for a single time series with NaN propagation.
    
    Parameters:
    - series: 1D numpy array (may contain NaN)
    - m: embedding dimension
    - tau: time delay
    
    Returns:
    - 2D array (n_samples × m) where each row is an m-dimensional embedding vector
    
    Cold start behavior:
    - First (m-1)*tau rows: NaN (insufficient history)
    - Remaining rows: embedding coordinates [x(t), x(t-tau), ..., x(t-(m-1)*tau)]
    - NaN in source series propagates to embedding coordinates
    """
    n = len(series)
    max_lag = (m - 1) * tau
    n_valid = n - max_lag
    
    # Pre-allocate with NaN
    embedding = np.full((n, m), np.nan)
    
    if n_valid <= 0:
        # Entire series too short for embedding - return all NaN
        return embedding
    
    # Fill valid portion (rows >= max_lag)
    for i in range(n_valid):
        current_idx = i + max_lag
        
        # Build embedding vector: [x(t), x(t-tau), x(t-2*tau), ..., x(t-(m-1)*tau)]
        for j in range(m):
            time_offset = j * tau
            source_idx = current_idx - time_offset
            embedding[current_idx, j] = series[source_idx]
            # NaN in series[source_idx] propagates naturally
    
    return embedding


def add_2d_pairs(df, embedding_columns):
    """
    Add 2D coordinate pairs columns to DataFrame.
    
    Parameters:
    - df: DataFrame containing embedding columns (object dtype with arrays)
    - embedding_columns: list of embedding column names
    
    Returns:
    - tuple: (modified_df, new_column_names)
    """
    result_df = df.copy()
    new_columns = []
    
    for col in embedding_columns:
        # Convert list of arrays to 2D numpy array
        # df[col] contains list of 1D arrays (from add_takens_embeddings)
        embeddings = np.stack(df[col].values)
        
        # Convert to 2D pairs
        pairs = _fast_embeddings_to_2d(embeddings)
        
        # Store as list of arrays (object dtype column)
        new_col_name = f"{col}_2d"
        result_df[new_col_name] = [pairs[i] for i in range(len(pairs))]
        new_columns.append(new_col_name)
    
    return result_df, new_columns


@jit(nopython=True)
def _fast_embeddings_to_2d(embeddings):
    """
    Convert embeddings to consecutive 2D coordinate pairs.
    
    Parameters:
    - embeddings: 2D array (n_points, m) where m >= 2
    
    Returns:
    - 3D array (n_points, m-1, 2) of consecutive pairs
      pairs[i, j] = [embedding[i, j], embedding[i, j+1]]
    
    Example: embedding = [x0, x1, x2, x3, x4] (m=5)
    → pairs = [[x0,x1], [x1,x2], [x2,x3], [x3,x4]] (m-1=4 pairs)
    
    NaN handling:
    - If any coordinate in embedding[i] is NaN, entire pairs[i] becomes NaN
    """
    n_points, m = embeddings.shape
    
    if m < 2:
        # Cannot create pairs from 1D embedding
        return np.empty((n_points, 0, 2))
    
    pairs = np.empty((n_points, m-1, 2))
    
    for i in range(n_points):
        # If any NaN in embedding, propagate to all pairs
        if np.any(np.isnan(embeddings[i])):
            pairs[i, :, :] = np.nan  # Explicit 2D slice assignment
        else:
            # Create consecutive pairs
            for j in range(m-1):
                pairs[i, j, 0] = embeddings[i, j]
                pairs[i, j, 1] = embeddings[i, j+1]
    
    return pairs


def add_trajectory_features(df, trajectory_columns):
    """
    Add trajectory feature columns to DataFrame (optimized version).
    
    Parameters:
    - df: DataFrame containing trajectory columns
    - trajectory_columns: list of trajectory column names
    
    Returns:
    - tuple: (modified_df, new_column_names)
    """
    result_df = df.copy()
    new_columns = []
    
    for col in trajectory_columns:
        # Convert to 3D array - trust pipeline for correct dimensions
        trajectories = np.stack(df[col].values)
        
        # Compute features in batch
        features = _compute_trajectory_features_batch(trajectories)
        
        # Add feature columns
        feature_names = [f'{col}_jump_mag', f'{col}_path_eff', f'{col}_turn_var']
                        # f'{col}_aspect_ratio', f'{col}_compactness', f'{col}_rel_variance']
        for j, feat_name in enumerate(feature_names):
            result_df[feat_name] = features[:, j]
            new_columns.append(feat_name)
    
    return result_df, new_columns

@jit(nopython=True)
def _compute_trajectory_features_batch(trajectories):
    """
    Compute trajectory features for a batch of trajectories efficiently.
    
    Parameters:
    - trajectories: 3D array (n_trajectories, n_points, 2)
    
    Returns:
    - 2D array (n_trajectories, 3) with [jump_mag, path_eff, turn_var]
    """
    n_traj = trajectories.shape[0]
    features = np.empty((n_traj, 3))
    
    for i in range(n_traj):
        traj = trajectories[i]
        
        # If trajectory contains NaN, return NaN features
        if np.any(np.isnan(traj)):
            features[i] = np.array([np.nan]*3)
            continue
        
        # 1. Jump Magnitude: Max distance between consecutive points
        # Manual diff instead of np.diff(traj, axis=0)
        diffs = traj[1:] - traj[:-1]
        distances = np.sqrt(np.sum(diffs**2, axis=1))
        jump_magnitude = np.max(distances)
        
        # 2. Path Efficiency: Direct distance / Total path length
        direct_distance = np.sqrt(np.sum((traj[-1] - traj[0])**2))
        total_path_length = np.sum(distances)
        path_efficiency = direct_distance / total_path_length if total_path_length > 0 else 0.0
        
        # 3. Turning Angle Variance
        if traj.shape[0] < 3:
            turning_angle_variance = 0.0
        else:
            v1 = diffs[:-1]  # vectors from point i to i+1
            v2 = diffs[1:]   # vectors from point i+1 to i+2
            
            # Compute turning angles using dot product
            dot_products = np.sum(v1 * v2, axis=1)
            norms1 = np.sqrt(np.sum(v1**2, axis=1))
            norms2 = np.sqrt(np.sum(v2**2, axis=1))
            
            # Avoid division by zero
            valid_mask = (norms1 > 1e-10) & (norms2 > 1e-10)
            cos_angles = np.zeros(len(dot_products))
            
            for j in range(len(dot_products)):
                if valid_mask[j]:
                    cos_angles[j] = dot_products[j] / (norms1[j] * norms2[j])
            
            # Clamp to [-1, 1] and compute angles
            cos_angles = np.clip(cos_angles, -1.0, 1.0)
            angles = np.arccos(cos_angles)
            
            turning_angle_variance = np.var(angles)
        
        # 4. Aspect Ratio: Shape elongation (scale-invariant)
        # Manual min/max instead of np.min/max with axis
        min_x, max_x = traj[0, 0], traj[0, 0]
        min_y, max_y = traj[0, 1], traj[0, 1]
        
        for j in range(1, traj.shape[0]):
            if traj[j, 0] < min_x:
                min_x = traj[j, 0]
            if traj[j, 0] > max_x:
                max_x = traj[j, 0]
            if traj[j, 1] < min_y:
                min_y = traj[j, 1]
            if traj[j, 1] > max_y:
                max_y = traj[j, 1]
        
        span_x = max_x - min_x
        span_y = max_y - min_y
        max_span = max(span_x, span_y)
        min_span = min(span_x, span_y)
        aspect_ratio = max_span / min_span if min_span > 1e-10 else 0.0
        
        # 5. Compactness: 4π*Area/Perimeter² (scale-invariant, measures regularity)
        if total_path_length > 1e-10:
            # Compute signed area using shoelace formula
            area = 0.0
            for j in range(traj.shape[0]):
                x_curr, y_curr = traj[j, 0], traj[j, 1]
                x_next, y_next = traj[(j + 1) % traj.shape[0], 0], traj[(j + 1) % traj.shape[0], 1]
                area += x_curr * y_next - x_next * y_curr
            area = abs(area) / 2.0
            compactness = 4.0 * np.pi * area / (total_path_length * total_path_length)
        else:
            compactness = 0.0
        
        # 6. Relative Variance: Normalized spread from centroid (scale-invariant)
        # Manual mean calculation
        centroid_x = 0.0
        centroid_y = 0.0
        for j in range(traj.shape[0]):
            centroid_x += traj[j, 0]
            centroid_y += traj[j, 1]
        centroid_x /= traj.shape[0]
        centroid_y /= traj.shape[0]
        
        # Calculate distances from centroid
        centroid_distances = np.empty(traj.shape[0])
        for j in range(traj.shape[0]):
            dx = traj[j, 0] - centroid_x
            dy = traj[j, 1] - centroid_y
            centroid_distances[j] = np.sqrt(dx * dx + dy * dy)
        
        trajectory_span = np.sqrt(span_x * span_x + span_y * span_y)
        if trajectory_span > 1e-10:
            relative_variance = np.var(centroid_distances) / (trajectory_span * trajectory_span)
        else:
            relative_variance = 0.0
        
        features[i] = np.array([jump_magnitude, path_efficiency, turning_angle_variance])
    
    return features