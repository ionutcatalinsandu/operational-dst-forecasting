import pandas as pd
import numpy as np
from typing import List
from sklearn.metrics import confusion_matrix, classification_report


def evaluate_quantile_risk_assessment(y_true: np.ndarray, 
                                     quantile_predictions: np.ndarray,
                                     quantile_levels: List[float],
                                     storm_threshold: float = -50.0,
                                     trust_threshold: float = 0.7,
                                     risk_method: str = 'interpolation') -> None:
    """
    Simplified quantile risk assessment with clean output.
    
    Args:
        y_true: True DST values
        quantile_predictions: XGBoost quantile predictions (N_samples, N_quantiles)
        quantile_levels: List of quantile levels
        storm_threshold: DST threshold for storm definition
        trust_threshold: Risk threshold for taking action
        risk_method: Method for calculating risk ('interpolation', 'conservative', 'proportion')
    """
    
    # Calculate storm risks
    storm_risks = calculate_storm_risk(
        quantile_predictions, quantile_levels, storm_threshold, risk_method
    )
    
    # Create binary labels
    actual_storms = (y_true <= storm_threshold).astype(int)
    predicted_actions = (storm_risks >= trust_threshold).astype(int)
    
    # Basic statistics
    print("QUANTILE RISK ASSESSMENT RESULTS")
    print("=" * 40)
    print(f"Storm threshold: {storm_threshold} nT")
    print(f"Trust threshold: {trust_threshold:.1%}")
    print(f"Risk method: {risk_method}")
    print(f"Total samples: {len(y_true)}")
    
    # Classification report
    print(f"\nCLASSIFICATION REPORT:")
    print("-" * 25)
    target_names = ['No Storm', 'Storm']
    print(classification_report(actual_storms, predicted_actions, 
                              target_names=target_names, digits=3, zero_division=0))
    
    # Confusion matrix and operational metrics
    cm = confusion_matrix(actual_storms, predicted_actions)
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
    
    print(f"OPERATIONAL SUMMARY:")
    print("-" * 20)
    print(f"True Alarms:     {tp:4d} (storms correctly predicted)")
    print(f"False Alarms:    {fp:4d} (unnecessary actions)")
    print(f"Missed Storms:   {fn:4d} (storms not predicted)")
    print(f"Correct Quiet:   {tn:4d} (quiet periods correctly identified)")
    
    if tp + fn > 0:
        critical_detection_rate = tp / (tp + fn)
        print(f"Critical Event Detection: {critical_detection_rate:.1%}")
    
    # Risk distribution diagnostics
    print(f"\nRISK DIAGNOSTICS:")
    print("-" * 17)
    print(f"Mean storm risk: {storm_risks[actual_storms == 1].mean():.3f}")
    print(f"Mean quiet risk: {storm_risks[actual_storms == 0].mean():.3f}")
    print(f"Risk std (storms): {storm_risks[actual_storms == 1].std():.3f}")
    print(f"Risk std (quiet): {storm_risks[actual_storms == 0].std():.3f}")
    
    # Check for quantile collapse (your issue)
    print(f"\nQUANTILE DIAGNOSTICS:")
    print("-" * 19)
    
    # Calculate quantile spread for storm and non-storm cases
    storm_cases = quantile_predictions[actual_storms == 1]
    quiet_cases = quantile_predictions[actual_storms == 0]
    
    if len(storm_cases) > 0:
        storm_spread = np.mean(storm_cases[:, -1] - storm_cases[:, 0])  # 95th - 5th percentile
        print(f"Avg quantile spread (storms): {storm_spread:.1f} nT")
        
        # Check for quantile collapse
        storm_ranges = storm_cases[:, -1] - storm_cases[:, 0]
        collapsed_storms = (storm_ranges < 10).sum()  # Less than 10 nT spread
        print(f"Collapsed quantiles (storms): {collapsed_storms}/{len(storm_cases)} ({collapsed_storms/len(storm_cases):.1%})")
    
    if len(quiet_cases) > 0:
        quiet_spread = np.mean(quiet_cases[:, -1] - quiet_cases[:, 0])
        print(f"Avg quantile spread (quiet): {quiet_spread:.1f} nT")
    
    # High-confidence false alarms (your precision issue)
    high_confidence_mask = storm_risks >= 0.9
    if high_confidence_mask.sum() > 0:
        high_conf_accuracy = actual_storms[high_confidence_mask].mean()
        print(f"High confidence accuracy (>90%): {high_conf_accuracy:.3f}")
        print(f"High confidence predictions: {high_confidence_mask.sum()}")


def calculate_storm_risk(quantile_predictions: np.ndarray,
                        quantile_levels: List[float],
                        storm_threshold: float,
                        method: str = 'interpolation') -> np.ndarray:
    """
    Calculate storm risk probabilities from quantile predictions.
    
    Args:
        quantile_predictions: 2D array (N_samples, N_quantiles)
        quantile_levels: List of quantile levels (e.g., [0.05, 0.25, 0.50, 0.75, 0.95])
        storm_threshold: DST threshold for storm definition (e.g., -50)
        method: Risk calculation method ('interpolation', 'conservative', 'proportion')
        
    Returns:
        Array of storm risk probabilities (0 to 1)
    """
    # Sort quantile levels and predictions
    sorted_indices = np.argsort(quantile_levels)
    quantile_levels = np.array(quantile_levels)[sorted_indices]
    quantile_predictions = quantile_predictions[:, sorted_indices]
    
    n_samples = quantile_predictions.shape[0]
    storm_risks = np.zeros(n_samples)
    
    for i in range(n_samples):
        sample_quantiles = quantile_predictions[i, :]
        
        if method == 'interpolation':
            storm_risks[i] = _interpolation_method(sample_quantiles, quantile_levels, storm_threshold)
        elif method == 'conservative':
            storm_risks[i] = _conservative_method(sample_quantiles, quantile_levels, storm_threshold)
        elif method == 'proportion':
            storm_risks[i] = _proportion_method(sample_quantiles, quantile_levels, storm_threshold)
        else:
            raise ValueError(f"Unknown method: {method}")
    
    return np.clip(storm_risks, 0.0, 1.0)


def _interpolation_method(sample_quantiles: np.ndarray, 
                         quantile_levels: np.ndarray, 
                         storm_threshold: float) -> float:
    """Original interpolation method - finds exact crossing point."""
    storm_mask = sample_quantiles <= storm_threshold
    
    if not storm_mask.any():
        return 0.0
    elif storm_mask.all():
        return 1.0 - quantile_levels[0]
    else:
        # Find crossing point
        storm_indices = np.where(storm_mask)[0]
        non_storm_indices = np.where(~storm_mask)[0]
        
        if len(non_storm_indices) > 0 and len(storm_indices) > 0:
            last_non_storm_idx = non_storm_indices[-1]
            first_storm_idx = storm_indices[0]
            
            if first_storm_idx == last_non_storm_idx + 1:
                # Adjacent quantiles - interpolate
                q1_level = quantile_levels[last_non_storm_idx]
                q1_value = sample_quantiles[last_non_storm_idx]
                q2_level = quantile_levels[first_storm_idx]
                q2_value = sample_quantiles[first_storm_idx]
                
                if q2_value != q1_value:
                    interp_fraction = (storm_threshold - q1_value) / (q2_value - q1_value)
                    crossing_quantile = q1_level + interp_fraction * (q2_level - q1_level)
                    return 1.0 - crossing_quantile
                else:
                    return 1.0 - q1_level
            else:
                return 1.0 - quantile_levels[first_storm_idx]
        else:
            return storm_mask.mean()

def _conservative_method(sample_quantiles: np.ndarray, 
                        quantile_levels: np.ndarray, 
                        storm_threshold: float) -> float:
    """Conservative method - uses highest quantile that predicts storm."""
    storm_mask = sample_quantiles <= storm_threshold
    
    if not storm_mask.any():
        return 0.0
    else:
        # Find the highest quantile (closest to median) that predicts a storm
        storm_quantile_levels = quantile_levels[storm_mask]
        highest_storm_quantile = storm_quantile_levels.max()
        return 1.0 - highest_storm_quantile

def _proportion_method(sample_quantiles: np.ndarray, 
                      quantile_levels: np.ndarray, 
                      storm_threshold: float) -> float:
    """Proportion method - fraction of quantiles that predict storms."""
    storm_mask = sample_quantiles <= storm_threshold
    return storm_mask.mean()


def compare_risk_methods(y_true: np.ndarray,
                        quantile_predictions: np.ndarray, 
                        quantile_levels: List[float],
                        storm_threshold: float = -50.0,
                        trust_threshold: float = 0.7) -> None:
    """
    Compare different risk calculation methods.
    """
    methods = ['interpolation', 'conservative', 'proportion']
    
    print("RISK METHOD COMPARISON")
    print("=" * 30)
    
    for method in methods:
        print(f"\n{method.upper()} METHOD:")
        print("-" * (len(method) + 8))
        
        storm_risks = calculate_storm_risk(
            quantile_predictions, quantile_levels, storm_threshold, method
        )
        
        actual_storms = (y_true <= storm_threshold).astype(int)
        predicted_actions = (storm_risks >= trust_threshold).astype(int)
        
        # Quick metrics
        from sklearn.metrics import precision_score, recall_score, f1_score
        
        precision = precision_score(actual_storms, predicted_actions, zero_division=0)
        recall = recall_score(actual_storms, predicted_actions, zero_division=0)
        f1 = f1_score(actual_storms, predicted_actions, zero_division=0)
        
        print(f"Precision: {precision:.3f}")
        print(f"Recall:    {recall:.3f}")
        print(f"F1-Score:  {f1:.3f}")
        print(f"Action Rate: {predicted_actions.mean():.3f}")