"""
NER-style metrics for storm boundary detection.

Evaluates storm events as entities with start/end boundaries,
similar to Named Entity Recognition evaluation in NLP.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from sklearn.metrics import classification_report, accuracy_score
import warnings


def extract_storm_entities(labels: np.ndarray) -> List[Tuple[int, int]]:
    """
    Extract storm entities as (start_idx, end_idx) tuples from binary labels.
    
    Parameters
    ----------
    labels : np.ndarray
        Binary labels (0=O, 1=I-STORM)
    
    Returns
    -------
    List[Tuple[int, int]]
        List of storm events as (start, end) index pairs (inclusive)
        
    Examples
    --------
    >>> labels = np.array([0, 0, 1, 1, 1, 0, 1, 1, 0])
    >>> extract_storm_entities(labels)
    [(2, 4), (6, 7)]  # Two storm events
    
    Notes
    -----
    - Consecutive 1s form a single storm entity
    - Storm starts at first 1 after 0
    - Storm ends at last 1 before 0
    """
    entities = []
    start = None
    
    for i, label in enumerate(labels):
        if label == 1:  # I-STORM
            if start is None:
                start = i  # Start new storm
        else:  # label == 0 (O)
            if start is not None:
                entities.append((start, i - 1))
                start = None
    
    # Handle storm extending to sequence end
    if start is not None:
        entities.append((start, len(labels) - 1))
    
    return entities


def _compute_entity_matches(
    true_entities: List[Tuple[int, int]],
    pred_entities: List[Tuple[int, int]],
    tolerance: int
) -> Dict[str, Tuple[int, int]]:
    """
    Compute match counts at different strictness levels.
    
    Returns dict mapping match_type -> (n_pred_matches, n_true_matched)
    """
    results = {
        'exact': [0, set()],
        'strict_boundary': [0, set()],
        'partial_boundary': [0, set()],
        'overlap': [0, set()]
    }
    
    for pred_start, pred_end in pred_entities:
        for true_idx, (true_start, true_end) in enumerate(true_entities):
            
            start_diff = abs(pred_start - true_start)
            end_diff = abs(pred_end - true_end)
            
            # Exact match
            if start_diff == 0 and end_diff == 0:
                results['exact'][0] += 1
                results['exact'][1].add(true_idx)
            
            # Strict boundary: both within tolerance
            if start_diff <= tolerance and end_diff <= tolerance:
                results['strict_boundary'][0] += 1
                results['strict_boundary'][1].add(true_idx)
            
            # Partial boundary: either within tolerance
            if start_diff <= tolerance or end_diff <= tolerance:
                results['partial_boundary'][0] += 1
                results['partial_boundary'][1].add(true_idx)
            
            # Any overlap
            if pred_end >= true_start and pred_start <= true_end:
                results['overlap'][0] += 1
                results['overlap'][1].add(true_idx)
    
    # Convert sets to counts
    return {k: (v[0], len(v[1])) for k, v in results.items()}


def compute_boundary_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    label_map: Optional[Dict[int, str]] = None,
    boundary_tolerance: int = 2,
    print_report: bool = True
) -> Dict[str, float]:
    """
    Compute NER-style metrics for storm boundary detection.
    
    Evaluates at multiple matching levels:
    - exact: Boundaries must match perfectly
    - strict_boundary: Start AND end within tolerance
    - partial_boundary: Start OR end within tolerance  
    - overlap: Any temporal overlap exists
    
    Parameters
    ----------
    y_true : np.ndarray
        True binary labels (0=O, 1=I-STORM)
    y_pred : np.ndarray
        Predicted binary labels
    label_map : Dict[int, str], optional
        Label names for classification report
    boundary_tolerance : int
        Hours tolerance for boundary matching (default: 2)
    print_report : bool
        Whether to print detailed report
        
    Returns
    -------
    Dict[str, float]
        Metrics dictionary with precision, recall, F1 for each match type
    """
    # Input validation
    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()
    
    if len(y_true) != len(y_pred):
        raise ValueError(f"Length mismatch: {len(y_true)} vs {len(y_pred)}")
    
    # Handle NaNs
    valid_mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    if not valid_mask.all():
        warnings.warn(f"Excluding {(~valid_mask).sum()} NaN values")
        y_true = y_true[valid_mask]
        y_pred = y_pred[valid_mask]
    
    if len(y_true) == 0:
        raise ValueError("No valid data points")
    
    # Extract entities
    true_entities = extract_storm_entities(y_true)
    pred_entities = extract_storm_entities(y_pred)
    
    n_true = len(true_entities)
    n_pred = len(pred_entities)
    
    # Compute matches
    matches = _compute_entity_matches(true_entities, pred_entities, boundary_tolerance)
    
    # Build metrics dict
    metrics = {
        'n_true_storms': n_true,
        'n_pred_storms': n_pred,
        'boundary_tolerance': boundary_tolerance
    }
    
    for match_type, (n_pred_matches, n_true_matched) in matches.items():
        precision = n_pred_matches / n_pred if n_pred > 0 else 0.0
        recall = n_true_matched / n_true if n_true > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        metrics[f'precision_{match_type}'] = precision
        metrics[f'recall_{match_type}'] = recall
        metrics[f'f1_{match_type}'] = f1
        metrics[f'n_matches_{match_type}'] = n_pred_matches
    
    # Token-level metrics
    metrics['token_accuracy'] = accuracy_score(y_true, y_pred)
    
    storm_mask = y_true == 1
    metrics['token_accuracy_storm'] = (
        (y_true[storm_mask] == y_pred[storm_mask]).mean() 
        if storm_mask.any() else np.nan
    )
    
    quiet_mask = y_true == 0
    metrics['token_accuracy_quiet'] = (
        (y_true[quiet_mask] == y_pred[quiet_mask]).mean()
        if quiet_mask.any() else np.nan
    )
    
    if print_report:
        _print_report(metrics, n_true, n_pred, matches, boundary_tolerance, 
                      y_true, y_pred, label_map)
    
    return metrics


def _print_report(
    metrics: Dict,
    n_true: int,
    n_pred: int,
    matches: Dict,
    tolerance: int,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    label_map: Optional[Dict[int, str]]
):
    """Print formatted metrics report."""
    print("\n" + "=" * 70)
    print("STORM BOUNDARY DETECTION METRICS (Binary Classification)")
    print("=" * 70)
    
    print(f"\nStorm Events: {n_true} true, {n_pred} predicted")
    print(f"Boundary tolerance: ±{tolerance} hours")
    
    for match_type, label in [
        ('exact', 'Exact Match'),
        ('strict_boundary', f'Strict Boundary (both within ±{tolerance}h)'),
        ('partial_boundary', f'Partial Boundary (either within ±{tolerance}h)'),
        ('overlap', 'Any Overlap')
    ]:
        print(f"\n{label}:")
        print(f"  P: {metrics[f'precision_{match_type}']:.1%}  "
              f"R: {metrics[f'recall_{match_type}']:.1%}  "
              f"F1: {metrics[f'f1_{match_type}']:.1%}")
    
    print(f"\nToken-Level:")
    print(f"  Overall: {metrics['token_accuracy']:.1%}")
    if not np.isnan(metrics['token_accuracy_storm']):
        print(f"  Storm (I): {metrics['token_accuracy_storm']:.1%}")
    if not np.isnan(metrics['token_accuracy_quiet']):
        print(f"  Quiet (O): {metrics['token_accuracy_quiet']:.1%}")
    
    if label_map is not None:
        print(f"\nClassification Report:")
        target_names = [label_map.get(i, f"Class_{i}") for i in range(2)]
        print(classification_report(y_true, y_pred, target_names=target_names, 
                                   digits=3, zero_division=0))
    
    print("=" * 70)