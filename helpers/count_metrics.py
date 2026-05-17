import numpy as np
from typing import Dict
from sklearn.metrics import mean_absolute_error, mean_squared_error
import warnings


def compute_duration_metrics(y_true: np.ndarray, 
                            y_pred: np.ndarray,
                            print_report: bool = True) -> Dict[str, float]:
    """
    Compute metrics for storm duration prediction (count regression).
    
    Evaluates how well the model predicts "hours until recovery" for both
    storm periods (duration > 0) and quiet periods (duration = 0).
    
    Args:
        y_true: True duration counts (0, 1, 2, 3, ...)
        y_pred: Predicted duration counts
        print_report: Whether to print detailed report
    
    Returns:
        Dictionary containing:
            - 'mae_all': MAE across all samples
            - 'mae_storm': MAE for storm periods only (y_true > 0)
            - 'mae_quiet': MAE for quiet periods only (y_true = 0)
            - 'exact_match_rate': % of predictions within 0 hours of truth
            - 'within_1h_rate': % of predictions within 1 hour of truth
            - 'within_2h_rate': % of predictions within 2 hours of truth
            - 'storm_detection_rate': % of storms correctly identified (pred > 0 when true > 0)
            - 'false_alarm_rate': % of quiet times incorrectly flagged (pred > 0 when true = 0)
            - 'mean_duration_true': Average storm duration in data
            - 'mean_duration_pred': Average predicted storm duration
    """
    # Input validation
    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()
    
    if len(y_true) != len(y_pred):
        raise ValueError(f"Length mismatch: {len(y_true)} vs {len(y_pred)}")
    
    # Handle NaNs
    valid_mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    if not valid_mask.all():
        n_invalid = (~valid_mask).sum()
        warnings.warn(f"Excluding {n_invalid} NaN values")
        y_true = y_true[valid_mask]
        y_pred = y_pred[valid_mask]
    
    if len(y_true) == 0:
        raise ValueError("No valid data points")
    
    # Round predictions to integers (counts must be whole numbers)
    y_pred_rounded = np.round(y_pred).astype(int)
    y_pred_rounded = np.maximum(y_pred_rounded, 0)  # Ensure non-negative
    
    metrics = {}
    
    # Overall metrics
    metrics['mae_all'] = mean_absolute_error(y_true, y_pred_rounded)
    metrics['rmse_all'] = np.sqrt(mean_squared_error(y_true, y_pred_rounded))
    
    # Storm periods (duration > 0)
    storm_mask = y_true > 0
    if storm_mask.any():
        metrics['mae_storm'] = mean_absolute_error(
            y_true[storm_mask], 
            y_pred_rounded[storm_mask]
        )
        metrics['rmse_storm'] = np.sqrt(mean_squared_error(
            y_true[storm_mask],
            y_pred_rounded[storm_mask]
        ))
    else:
        metrics['mae_storm'] = np.nan
        metrics['rmse_storm'] = np.nan
    
    # Quiet periods (duration = 0)
    quiet_mask = y_true == 0
    if quiet_mask.any():
        metrics['mae_quiet'] = mean_absolute_error(
            y_true[quiet_mask],
            y_pred_rounded[quiet_mask]
        )
    else:
        metrics['mae_quiet'] = np.nan
    
    # Tolerance-based accuracy (how often are we close?)
    abs_error = np.abs(y_true - y_pred_rounded)
    metrics['exact_match_rate'] = (abs_error == 0).mean() * 100
    metrics['within_1h_rate'] = (abs_error <= 1).mean() * 100
    metrics['within_2h_rate'] = (abs_error <= 2).mean() * 100
    
    # Storm detection (binary: did we identify storm periods?)
    storm_detected = (y_pred_rounded > 0) & (y_true > 0)
    storm_missed = (y_pred_rounded == 0) & (y_true > 0)
    false_alarm = (y_pred_rounded > 0) & (y_true == 0)
    true_negative = (y_pred_rounded == 0) & (y_true == 0)
    
    if storm_mask.any():
        metrics['storm_detection_rate'] = storm_detected.sum() / storm_mask.sum() * 100
    else:
        metrics['storm_detection_rate'] = np.nan
    
    if quiet_mask.any():
        metrics['false_alarm_rate'] = false_alarm.sum() / quiet_mask.sum() * 100
    else:
        metrics['false_alarm_rate'] = np.nan
    
    # Duration statistics
    if storm_mask.any():
        metrics['mean_duration_true'] = y_true[storm_mask].mean()
        metrics['median_duration_true'] = np.median(y_true[storm_mask])
        metrics['max_duration_true'] = y_true[storm_mask].max()
    else:
        metrics['mean_duration_true'] = 0
        metrics['median_duration_true'] = 0
        metrics['max_duration_true'] = 0
    
    storm_pred_mask = y_pred_rounded > 0
    if storm_pred_mask.any():
        metrics['mean_duration_pred'] = y_pred_rounded[storm_pred_mask].mean()
        metrics['median_duration_pred'] = np.median(y_pred_rounded[storm_pred_mask])
    else:
        metrics['mean_duration_pred'] = 0
        metrics['median_duration_pred'] = 0
    
    # Sample counts
    metrics['n_total'] = len(y_true)
    metrics['n_storm_samples'] = storm_mask.sum()
    metrics['n_quiet_samples'] = quiet_mask.sum()
    metrics['storm_sample_rate'] = storm_mask.mean() * 100
    
    # Print report
    if print_report:
        print("\n" + "="*60)
        print("STORM DURATION PREDICTION METRICS")
        print("="*60)
        
        print(f"\nDataset Overview:")
        print(f"  Total samples: {metrics['n_total']:,}")
        print(f"  Storm samples (duration > 0): {metrics['n_storm_samples']:,} ({metrics['storm_sample_rate']:.1f}%)")
        print(f"  Quiet samples (duration = 0): {metrics['n_quiet_samples']:,}")
        
        print(f"\nAccuracy by Tolerance:")
        print(f"  Exact match (0h error): {metrics['exact_match_rate']:.1f}%")
        print(f"  Within 1 hour: {metrics['within_1h_rate']:.1f}%")
        print(f"  Within 2 hours: {metrics['within_2h_rate']:.1f}%")
        
        print(f"\nMean Absolute Error:")
        print(f"  Overall: {metrics['mae_all']:.2f} hours")
        if not np.isnan(metrics['mae_storm']):
            print(f"  Storm periods: {metrics['mae_storm']:.2f} hours")
        if not np.isnan(metrics['mae_quiet']):
            print(f"  Quiet periods: {metrics['mae_quiet']:.2f} hours")
        
        print(f"\nStorm Detection Performance:")
        if not np.isnan(metrics['storm_detection_rate']):
            print(f"  Detection rate (sensitivity): {metrics['storm_detection_rate']:.1f}%")
            print(f"  Missed storms: {storm_missed.sum():,} ({100 - metrics['storm_detection_rate']:.1f}%)")
        if not np.isnan(metrics['false_alarm_rate']):
            print(f"  False alarm rate: {metrics['false_alarm_rate']:.1f}%")
            print(f"  False alarms: {false_alarm.sum():,}")
        
        print(f"\nDuration Statistics:")
        if metrics['mean_duration_true'] > 0:
            print(f"  True storms - Mean: {metrics['mean_duration_true']:.1f}h, "
                  f"Median: {metrics['median_duration_true']:.0f}h, "
                  f"Max: {metrics['max_duration_true']:.0f}h")
        if metrics['mean_duration_pred'] > 0:
            print(f"  Predicted storms - Mean: {metrics['mean_duration_pred']:.1f}h, "
                  f"Median: {metrics['median_duration_pred']:.0f}h")
        
        print("="*60)
    
    return metrics