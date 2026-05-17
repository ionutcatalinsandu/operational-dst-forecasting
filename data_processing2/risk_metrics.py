import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_curve
import matplotlib.pyplot as plt

def create_storm_mask(dst_values, storm_threshold=-50, min_duration_hours=1):
    """
    Create binary storm mask from DST values.
    
    Parameters:
    -----------
    dst_values : array-like
        Original DST values in nT
    storm_threshold : float, default=-50
        DST threshold for storm definition (nT)
    min_duration_hours : int, default=1
        Minimum consecutive hours below threshold to count as storm
        
    Returns:
    --------
    np.array
        Binary storm mask (1=storm, 0=quiet)
    """
    dst_array = np.array(dst_values)
    below_threshold = dst_array <= storm_threshold
    
    if min_duration_hours == 1:
        return below_threshold.astype(int)
    
    # Apply minimum duration filter for multi-hour storms
    storm_mask = np.zeros_like(below_threshold, dtype=int)
    i = 0
    while i < len(below_threshold):
        if below_threshold[i]:
            start = i
            while i < len(below_threshold) and below_threshold[i]:
                i += 1
            if (i - start) >= min_duration_hours:
                storm_mask[start:i] = 1
        else:
            i += 1
    
    return storm_mask


def severity_to_risk(severity_scores):
    """
    Convert severity Z-scores to risk probabilities using normal CDF.
    
    Parameters:
    -----------
    severity_scores : array-like
        Z-scores from adaptive severity transformation
        
    Returns:
    --------
    np.array
        Risk probabilities [0,1] where higher = more severe
    """
    return stats.norm.cdf(np.array(severity_scores))


def evaluate_storm_predictions(true_scores, predicted_scores, storm_mask, default_threshold=0.85):
    """
    Evaluate storm risk predictions with classification report, confusion matrix, 
    and precision-recall curve analysis.
    
    Parameters:
    -----------
    true_scores : array-like
        True severity Z-scores from adaptive severity transformation
    predicted_scores : array-like  
        Predicted severity Z-scores from model
    storm_mask : array-like
        Binary storm mask (1=storm, 0=quiet)
    default_threshold : float, default=0.85
        Default risk threshold for binary classification reports
    """
    
    # Clean data
    true_scores = np.array(true_scores)
    pred_scores = np.array(predicted_scores)
    storm_mask = np.array(storm_mask).astype(int)
    
    # Remove NaN values
    valid = ~(np.isnan(true_scores) | np.isnan(pred_scores) | np.isnan(storm_mask))
    true_scores = true_scores[valid]
    pred_scores = pred_scores[valid]
    storm_mask = storm_mask[valid]
    
    # Convert scores to risk probabilities
    pred_risk = severity_to_risk(pred_scores)
    
    # 1. Classification Report
    pred_binary = (pred_risk >= default_threshold).astype(int)
    print("=" * 60)
    print(f"CLASSIFICATION REPORT (Risk Threshold: {default_threshold:.2f})")
    print("=" * 60)
    print(classification_report(storm_mask, pred_binary, 
                              target_names=['Quiet', 'Storm'], 
                              digits=3))
    
    # 2. Confusion Matrix
    cm = confusion_matrix(storm_mask, pred_binary)
    tn, fp, fn, tp = cm.ravel()
    
    print("\n" + "=" * 40)
    print("CONFUSION MATRIX")
    print("=" * 40)
    print(f"                Predicted")
    print(f"Actual    Quiet  Storm")
    print(f"Quiet     {tn:5d}  {fp:5d}")
    print(f"Storm     {fn:5d}  {tp:5d}")
    print()
    print(f"True Positives (Correct Storm Alerts):  {tp:5d}")
    print(f"True Negatives (Correct Quiet Periods): {tn:5d}")
    print(f"False Positives (False Alarms):         {fp:5d}")
    print(f"False Negatives (Missed Storms):        {fn:5d}")
    print()
    print(f"Total Storms in Data: {np.sum(storm_mask):5d}")
    print(f"Total Alerts Issued:  {np.sum(pred_binary):5d}")
    
    # 3. Precision-Recall Curve
    precision, recall, thresholds = precision_recall_curve(storm_mask, pred_risk)
    
    # Find optimal threshold (maximum F1-score)
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
    optimal_idx = np.argmax(f1_scores)
    optimal_threshold = thresholds[optimal_idx] if optimal_idx < len(thresholds) else 1.0
    optimal_f1 = f1_scores[optimal_idx]
    
    # Plot Precision-Recall Curve
    plt.figure(figsize=(10, 6))
    
    plt.subplot(1, 2, 1)
    plt.plot(recall, precision, 'b-', linewidth=2, label='PR Curve')
    plt.plot(recall[optimal_idx], precision[optimal_idx], 'ro', markersize=10, 
             label=f'Optimal (F1={optimal_f1:.3f})')
    plt.xlabel('Recall (Storm Detection Rate)')
    plt.ylabel('Precision (Alert Accuracy)')
    plt.title('Precision-Recall Curve')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    
    # Plot F1-Score vs Threshold
    plt.subplot(1, 2, 2)
    valid_f1 = f1_scores[:-1]  # Remove last point (threshold edge case)
    valid_thresholds = thresholds[:len(valid_f1)]
    
    plt.plot(valid_thresholds, valid_f1, 'g-', linewidth=2)
    plt.axvline(optimal_threshold, color='red', linestyle='--', 
                label=f'Optimal Risk = {optimal_threshold:.3f}')
    plt.xlabel('Risk Threshold')
    plt.ylabel('F1-Score')
    plt.title('F1-Score vs Risk Threshold')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.xlim([0, 1])
    
    plt.tight_layout()
    plt.show()
    
    print(f"\n" + "=" * 50)
    print("OPTIMAL RISK THRESHOLD ANALYSIS")
    print("=" * 50)
    print(f"Optimal Risk Threshold: {optimal_threshold:.3f} ({optimal_threshold*100:.1f}th percentile)")
    print(f"Maximum F1-Score:       {optimal_f1:.3f}")
    print(f"Precision at Optimal:   {precision[optimal_idx]:.3f}")
    print(f"Recall at Optimal:      {recall[optimal_idx]:.3f}")


# Example usage
def test_evaluation():
    """Test the evaluation framework with synthetic data."""
    np.random.seed(42)
    n_points = 1000
    
    # Synthetic severity scores
    true_scores = np.random.normal(0, 1, n_points)
    pred_scores = true_scores + np.random.normal(0, 0.4, n_points)  # Add prediction error
    
    # Create storm mask from synthetic DST
    synthetic_dst = -30 + true_scores * 25 + np.random.normal(0, 15, n_points)
    storm_mask = create_storm_mask(synthetic_dst, storm_threshold=-60, min_duration_hours=2)
    
    print(f"Generated {np.sum(storm_mask)} storm hours out of {len(storm_mask)} total hours")
    print(f"Storm rate: {np.mean(storm_mask):.3f}")
    
    # Run evaluation
    evaluate_storm_predictions(true_scores, pred_scores, storm_mask)

if __name__ == "__main__":
    test_evaluation()