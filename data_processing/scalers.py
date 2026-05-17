
import numpy as np
import pandas as pd

from scipy import stats
from typing import Tuple, Optional, List, Union

from sklearn.base import BaseEstimator, TransformerMixin

class ClippingTransformer(BaseEstimator, TransformerMixin):
    """Clips values to specified range with NaN handling"""
    
    def __init__(self, clip_min: Optional[float] = None, clip_max: Optional[float] = None):
        self.clip_min = clip_min
        self.clip_max = clip_max
    
    def fit(self, X, y=None):
        # Validate that clip_min < clip_max if both provided
        if self.clip_min is not None and self.clip_max is not None:
            if self.clip_min >= self.clip_max:
                raise ValueError(f"clip_min ({self.clip_min}) must be less than clip_max ({self.clip_max})")
        return self
    
    def transform(self, X):
        X = np.asarray(X).ravel()
        
        # Create output array
        X_clipped = np.full_like(X, np.nan, dtype=float)
        
        # Only clip non-NaN values
        valid_mask = ~np.isnan(X)
        if np.any(valid_mask):
            X_clipped[valid_mask] = np.clip(X[valid_mask], self.clip_min, self.clip_max)
        
        return X_clipped
    
    def inverse_transform(self, X):
        # Clipping is not reversible - just preserve NaN and return as is
        return np.asarray(X).ravel()


class SignedLogTransformer(BaseEstimator, TransformerMixin):
    """Signed log transformation for values that can be negative with NaN handling"""
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X = np.asarray(X).ravel()
        
        # Create output array
        X_transformed = np.full_like(X, np.nan, dtype=float)
        
        # Only transform non-NaN values
        valid_mask = ~np.isnan(X)
        if np.any(valid_mask):
            X_valid = X[valid_mask]
            # sign(X) * log1p(|X|)
            X_transformed[valid_mask] = np.sign(X_valid) * np.log1p(np.abs(X_valid))
        
        return X_transformed
    
    def inverse_transform(self, X):
        X = np.asarray(X).ravel()
        
        # Create output array
        X_inv = np.full_like(X, np.nan, dtype=float)
        
        # Only inverse transform non-NaN values
        valid_mask = ~np.isnan(X)
        if np.any(valid_mask):
            X_valid = X[valid_mask]
            # sign(X) * expm1(|X|)
            X_inv[valid_mask] = np.sign(X_valid) * np.expm1(np.abs(X_valid))
        
        return X_inv
    

class YeoJohnsonTransformer(BaseEstimator, TransformerMixin):
    """Yeo-Johnson transformation - handles negative values and NaN values"""
    
    def __init__(self, standardize: bool = True):
        self.standardize = standardize
        self.lambda_ = None
        self._scaler = None
    
    def fit(self, X, y=None):
        X = np.asarray(X).ravel()
        
        # Mask out NaN values for fitting
        valid_mask = ~np.isnan(X)
        X_valid = X[valid_mask]
        
        # Check if we have enough valid data
        if len(X_valid) < 2:
            raise ValueError(f"Need at least 2 non-NaN values to fit, got {len(X_valid)}")
        
        # Fit on valid data only
        _, self.lambda_ = stats.yeojohnson(X_valid)
        
        if self.standardize:
            X_transformed = stats.yeojohnson(X_valid, lmbda=self.lambda_)
            mean_val = np.mean(X_transformed)
            std_val = np.std(X_transformed)
            
            # Ensure std is not zero for numerical stability
            if std_val < 1e-10:
                std_val = 1.0
            
            self._scaler = (mean_val, std_val)
        
        return self
    
    def transform(self, X):
        X = np.asarray(X).ravel()
        
        # Create output array
        X_transformed = np.full_like(X, np.nan, dtype=float)
        
        # Only transform non-NaN values
        valid_mask = ~np.isnan(X)
        if np.any(valid_mask):
            X_transformed[valid_mask] = stats.yeojohnson(X[valid_mask], lmbda=self.lambda_)
            
            if self.standardize and self._scaler:
                X_transformed[valid_mask] = (X_transformed[valid_mask] - self._scaler[0]) / self._scaler[1]
        
        return X_transformed
    
    def inverse_transform(self, X):
        X = np.asarray(X).ravel()
        
        # Create output array
        X_inv = np.full_like(X, np.nan, dtype=float)
        
        # Only inverse transform non-NaN values
        valid_mask = ~np.isnan(X)
        if np.any(valid_mask):
            X_valid = X[valid_mask].copy()
            
            if self.standardize and self._scaler:
                X_valid = X_valid * self._scaler[1] + self._scaler[0]
            
            # Inverse Yeo-Johnson
            X_inv[valid_mask] = self._inverse_yeojohnson(X_valid, self.lambda_)
        
        return X_inv
    
    def _inverse_yeojohnson(self, y, lmbda):
        """Inverse of Yeo-Johnson transformation"""
        # Handle special cases for numerical stability
        if np.abs(lmbda) < 1e-10:  # lmbda ≈ 0
            return np.where(y >= 0, np.expm1(y), -np.expm1(-y))
        elif np.abs(lmbda - 2) < 1e-10:  # lmbda ≈ 2
            return np.where(y >= 0, 
                          np.sqrt(np.maximum(y + 1, 0)) - 1,  # Protect against numerical errors
                          1 - np.sqrt(np.maximum(1 - y, 0)))
        else:
            pos_idx = y >= 0
            x = np.empty_like(y)
            
            # Positive values: x = (λy + 1)^(1/λ) - 1
            if np.any(pos_idx):
                # Protect against negative values under root/power
                base = np.maximum(y[pos_idx] * lmbda + 1, 1e-10)
                x[pos_idx] = np.power(base, 1/lmbda) - 1
            
            # Negative values: x = 1 - (1 - (2-λ)y)^(1/(2-λ))
            if np.any(~pos_idx):
                base = np.maximum(1 - (2-lmbda) * y[~pos_idx], 1e-10)
                x[~pos_idx] = 1 - np.power(base, 1/(2-lmbda))
            
            return x


class SinhArcsinhTransformer(BaseEstimator, TransformerMixin):
    """Sinh-arcsinh transformation for heavy-tailed distributions with NaN handling"""
    
    def __init__(self, epsilon: float = 0.0, delta: float = 1.0):
        self.epsilon = epsilon  # skewness parameter
        self.delta = delta      # tail weight parameter
    
    def fit(self, X, y=None):
        X = np.asarray(X).ravel()
        
        # Validate parameters for numerical stability
        if np.abs(self.delta) < 1e-10:
            raise ValueError(f"delta must be non-zero, got {self.delta}")
        
        # Check if we have any valid data (optional validation)
        valid_mask = ~np.isnan(X)
        if not np.any(valid_mask):
            raise ValueError("All values are NaN, cannot fit transformer")
        
        return self
    
    def transform(self, X):
        X = np.asarray(X).ravel()
        
        # Create output array
        X_transformed = np.full_like(X, np.nan, dtype=float)
        
        # Only transform non-NaN values
        valid_mask = ~np.isnan(X)
        if np.any(valid_mask):
            # Clip extreme values for numerical stability in arcsinh
            X_valid = np.clip(X[valid_mask], -1e100, 1e100)
            
            # sinh(delta * arcsinh(X) - epsilon)
            X_transformed[valid_mask] = np.sinh(
                self.delta * np.arcsinh(X_valid) - self.epsilon
            )
        
        return X_transformed
    
    def inverse_transform(self, X):
        X = np.asarray(X).ravel()
        
        # Create output array
        X_inv = np.full_like(X, np.nan, dtype=float)
        
        # Only inverse transform non-NaN values
        valid_mask = ~np.isnan(X)
        if np.any(valid_mask):
            # Clip extreme values for numerical stability in arcsinh
            X_valid = np.clip(X[valid_mask], -1e100, 1e100)
            
            # sinh((arcsinh(X) + epsilon) / delta)
            X_inv[valid_mask] = np.sinh(
                (np.arcsinh(X_valid) + self.epsilon) / self.delta
            )
        
        return X_inv


def apply_transformation(df: pd.DataFrame, 
                        column_name: str,
                        transformer: BaseEstimator,
                        fit_transform: bool = False) -> Tuple[pd.DataFrame, BaseEstimator]:
    """Apply a single transformation to a column"""
    df = df.copy()
    X = df[column_name].values
    
    # Fit and transform
    if fit_transform:
        X_transformed = transformer.fit_transform(X)
    else:
        X_transformed = transformer.transform(X)

    df[column_name] = X_transformed
    
    return df, transformer


def apply_transformation_pipeline(df: pd.DataFrame,
                                 column_name: str, 
                                 transformers: List[BaseEstimator],
                                 fit_transform: bool = False) -> Tuple[pd.DataFrame, List[BaseEstimator]]:
    """Apply a sequence of transformations to a column"""
    df = df.copy()
    fit_transformers = []
    
    for transformer in transformers:
        df, transf = apply_transformation(df, column_name, transformer, fit_transform=fit_transform)
        fit_transformers.append(transf)
    
    return df, fit_transformers

def apply_transformation_pipeline_raw(data: np.ndarray,
                                     transformers: List[BaseEstimator],
                                     fit_transform: bool = False) -> Tuple[np.ndarray, List[BaseEstimator]]:
    """
    Apply a sequence of transformations directly to a 1D numpy array.
    
    Parameters:
    - data: 1D numpy array to transform
    - transformers: List of sklearn transformers to apply sequentially
    - fit_transform: If True, fit and transform. If False, only transform (assumes pre-fitted)
    
    Returns:
    - tuple: (transformed_data, list_of_fitted_transformers)
    """
    # Reshape to 2D for sklearn, then back to 1D
    transformed_data = data
    fit_transformers = []
    
    for transformer in transformers:
        if fit_transform:
            transformed_data = transformer.fit_transform(transformed_data)
            fit_transformers.append(transformer)
        else:
            transformed_data = transformer.transform(transformed_data)
            fit_transformers.append(transformer)
    
    return transformed_data, fit_transformers


def apply_inverse_transformation_pipeline_raw(data: np.ndarray,
                                             fitted_transformers: List[BaseEstimator]) -> np.ndarray:
    """
    Apply inverse transformations to a 1D numpy array in reverse order.
    
    Parameters:
    - data: 1D numpy array to inverse transform
    - fitted_transformers: List of fitted sklearn transformers (from forward transform)
    
    Returns:
    - numpy array: inverse transformed data
    """
    # Reshape to 2D for sklearn
    inverse_data = data
    
    # Apply inverse transforms in reverse order
    for transformer in reversed(fitted_transformers):
        inverse_data = transformer.inverse_transform(inverse_data)
    
    return inverse_data


def inverse_transform_pipeline(df: pd.DataFrame,
                              column_name: str,
                              transformers: List[BaseEstimator]) -> pd.DataFrame:
    """Apply inverse transformations in reverse order"""
    df = df.copy()
    
    # Apply in reverse order
    for transformer in reversed(transformers):
        X = df[column_name].values
        X_inv = transformer.inverse_transform(X)
        df[column_name] = X_inv
    
    return df


def create_storm_classes(df: pd.DataFrame, 
                        column_name: str,
                        thresholds: List[float] = [-200, -100, -50, -30, 0]) -> pd.DataFrame:
    """
    Convert continuous DST values to storm intensity classes.
    
    Default thresholds create classes:
    - 0: Extreme storm (DST < -200)
    - 1: Severe storm (-200 ≤ DST < -100)
    - 2: Strong storm (-100 ≤ DST < -50)
    - 3: Moderate storm (-50 ≤ DST < -30)
    - 4: Minor/No storm (DST ≥ -30)
    """
    df = df.copy()
    X = df[column_name].values
    
    # Sort thresholds in ascending order
    thresholds = sorted(thresholds)
    
    # Digitize creates bins: (-inf, t0], (t0, t1], ..., (tn, inf)
    # We subtract 1 to get 0-indexed classes
    classes = np.digitize(X, thresholds) - 1
    
    # Ensure classes are non-negative
    classes = np.maximum(classes, 0)
    
    # Store as new column
    df[f'{column_name}_class'] = classes
    
    return df