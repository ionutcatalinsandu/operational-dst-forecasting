import numpy as np

from typing import Dict

from sklearn.metrics import (
    mean_absolute_error, 
    mean_squared_error, 
    f1_score, 
    classification_report,
    precision_score,
    recall_score,
    r2_score,
)

import warnings


def compute_forecast_metrics(y_true: np.ndarray, 
                           y_pred: np.ndarray,
                           storm_threshold: float = -50.0,
                           print_classification_report: bool = True) -> Dict[str, float]:
    """
    Compute comprehensive forecast metrics for space weather prediction.
    
    Calculates regression metrics (MAE, MAPE, MSE) and classification metrics
    (F1-score, precision, recall) based on storm threshold.
    
    Args:
        y_true (np.ndarray): True DST values
        y_pred (np.ndarray): Predicted DST values  
        storm_threshold (float): DST threshold for storm classification (default: -50.0 nT)
        print_classification_report (bool): Whether to print detailed classification report
    
    Returns:
        Dict[str, float]: Dictionary containing all computed metrics
            - 'mae': Mean Absolute Error
            - 'mape': Mean Absolute Percentage Error
            - 'mse': Mean Squared Error
            - 'rmse': Root Mean Squared Error
            - 'r2': R-squared (coefficient of determination)
            - 'f1_score': F1-score for storm detection
            - 'precision': Precision for storm detection
            - 'recall': Recall for storm detection
            - 'storm_rate_true': Percentage of true storm hours
            - 'storm_rate_pred': Percentage of predicted storm hours
    
    Example:
        >>> y_true = np.array([-10, -60, -80, -20, -100])
        >>> y_pred = np.array([-15, -55, -75, -25, -90])
        >>> metrics = compute_forecast_metrics(y_true, y_pred, storm_threshold=-50)
        >>> print(f"MAE: {metrics['mae']:.2f} nT")
    """
    # Input validation
    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()
    
    if len(y_true) != len(y_pred):
        raise ValueError(f"Length mismatch: y_true has {len(y_true)} elements, "
                        f"y_pred has {len(y_pred)} elements")
    
    if len(y_true) == 0:
        raise ValueError("Input arrays are empty")
    
    # Check for NaN values
    valid_mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    if not valid_mask.all():
        n_invalid = (~valid_mask).sum()
        warnings.warn(f"Found {n_invalid} NaN values. These will be excluded from calculations.")
        y_true = y_true[valid_mask]
        y_pred = y_pred[valid_mask]
    
    if len(y_true) == 0:
        raise ValueError("No valid (non-NaN) data points found")
    
    # Initialize results dictionary
    metrics = {}
    
    # Regression metrics
    metrics['mae'] = mean_absolute_error(y_true, y_pred)
    metrics['mse'] = mean_squared_error(y_true, y_pred)
    metrics['rmse'] = np.sqrt(metrics['mse'])
    metrics['r2'] = r2_score(y_true, y_pred)
    
    # MAPE calculation (handle division by zero)
    # For DST, we use absolute values since DST can be positive or negative
    # Alternative: use symmetric MAPE if preferred
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        # Standard MAPE
        mape_values = np.abs((y_true - y_pred) / y_true) * 100
        # Handle zero/near-zero values by using a small epsilon
        epsilon = 1e-8
        mape_safe = np.abs((y_true - y_pred) / (y_true + epsilon * np.sign(y_true))) * 100
        metrics['mape'] = np.mean(np.where(np.abs(y_true) > epsilon, mape_values, mape_safe))
    
    # Convert to binary classification for storm detection
    y_true_binary = (y_true <= storm_threshold).astype(int)
    y_pred_binary = (y_pred <= storm_threshold).astype(int)
    
    # Classification metrics
    if len(np.unique(y_true_binary)) > 1:  # Check if we have both classes
        metrics['f1_score'] = f1_score(y_true_binary, y_pred_binary, zero_division=0)
        metrics['precision'] = precision_score(y_true_binary, y_pred_binary, zero_division=0)
        metrics['recall'] = recall_score(y_true_binary, y_pred_binary, zero_division=0)
    else:
        # If only one class present, set metrics appropriately
        if y_true_binary[0] == 1:  # All storms
            metrics['precision'] = (y_pred_binary == 1).mean()
            metrics['recall'] = (y_pred_binary == 1).mean()
        else:  # No storms
            metrics['precision'] = 1.0 if (y_pred_binary == 0).all() else 0.0
            metrics['recall'] = 1.0  # Undefined, but set to 1 for no storms correctly identified
        
        # F1 score
        if metrics['precision'] + metrics['recall'] > 0:
            metrics['f1_score'] = 2 * (metrics['precision'] * metrics['recall']) / (metrics['precision'] + metrics['recall'])
        else:
            metrics['f1_score'] = 0.0
        
        warnings.warn(f"Only one class present in y_true (all {'storms' if y_true_binary[0] == 1 else 'non-storms'}). "
                     f"Classification metrics may not be meaningful.")
    
    # Storm occurrence rates
    metrics['storm_rate_true'] = y_true_binary.mean() * 100  # Percentage
    metrics['storm_rate_pred'] = y_pred_binary.mean() * 100  # Percentage
    
    # Additional useful metrics
    metrics['correlation'] = np.corrcoef(y_true, y_pred)[0, 1] if len(y_true) > 1 else np.nan
    
    # Print detailed classification report if requested
    if print_classification_report and len(np.unique(y_true_binary)) > 1:
        print("\n" + "="*50)
        print("STORM DETECTION CLASSIFICATION REPORT")
        print("="*50)
        print(f"Storm threshold: {storm_threshold} nT")
        print(f"Total samples: {len(y_true)}")
        print(f"True storm hours: {y_true_binary.sum()} ({metrics['storm_rate_true']:.1f}%)")
        print(f"Predicted storm hours: {y_pred_binary.sum()} ({metrics['storm_rate_pred']:.1f}%)")
        print("\nDetailed Classification Report:")
        
        # Custom target names for better readability
        target_names = ['Non-storm', 'Storm']
        print(classification_report(y_true_binary, y_pred_binary, 
                                  target_names=target_names, 
                                  digits=3))
        
        # Additional storm-specific insights
        true_positives = ((y_true_binary == 1) & (y_pred_binary == 1)).sum()
        false_positives = ((y_true_binary == 0) & (y_pred_binary == 1)).sum()
        false_negatives = ((y_true_binary == 1) & (y_pred_binary == 0)).sum()
        true_negatives = ((y_true_binary == 0) & (y_pred_binary == 0)).sum()
        
        print(f"\nConfusion Matrix Summary:")
        print(f"True Positives (storms correctly detected): {true_positives}")
        print(f"False Positives (false alarms): {false_positives}")
        print(f"False Negatives (missed storms): {false_negatives}")
        print(f"True Negatives (quiet times correctly identified): {true_negatives}")
        
        if true_positives + false_negatives > 0:
            critical_events_caught = true_positives / (true_positives + false_negatives)
            print(f"\nCritical Events Detection Rate: {critical_events_caught:.1%}")
        
        print("="*50)
    
    return metrics