"""
Focal Loss objective for XGBoost multiclass classification.

Focal loss down-weights easy examples, focusing training on hard cases.
Particularly useful for imbalanced datasets like storm detection where
quiet periods (O) dominate over storm events (B-STORM, I-STORM).

Reference: Lin et al., "Focal Loss for Dense Object Detection", ICCV 2017
"""

import numpy as np
from typing import Tuple, Optional


def softmax(logits: np.ndarray) -> np.ndarray:
    """Numerically stable softmax over last axis."""
    shifted = logits - logits.max(axis=-1, keepdims=True)
    exp_logits = np.exp(shifted)
    return exp_logits / exp_logits.sum(axis=-1, keepdims=True)


def xgb_focal_loss_objective(
    predt: np.ndarray,
    dtrain,
    gamma: float = 2.0,
    class_weights: Optional[np.ndarray] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Focal loss objective for XGBoost multiclass classification.
    
    Parameters
    ----------
    predt : np.ndarray
        Raw predictions (logits) from XGBoost, shape (n_samples * n_classes,)
        XGBoost flattens multiclass predictions row-major.
    dtrain : xgb.DMatrix
        Training data containing labels.
    gamma : float
        Focusing parameter. Higher values increase focus on hard examples.
        gamma=0 recovers standard cross-entropy. Typical values: 1-5.
    class_weights : np.ndarray, optional
        Per-class weights, shape (n_classes,). Useful for additional
        imbalance correction beyond focal loss.
    
    Returns
    -------
    grad : np.ndarray
        Gradient, shape (n_samples * n_classes,)
    hess : np.ndarray
        Hessian diagonal approximation, shape (n_samples * n_classes,)
    
    Notes
    -----
    For multiclass, XGBoost expects gradients/hessians flattened as:
    [sample0_class0, sample0_class1, ..., sample0_classK, sample1_class0, ...]
    """
    labels = dtrain.get_label().astype(np.int32)
    n_samples = len(labels)
    n_classes = len(np.unique(labels))
    
    # Reshape logits: (n_samples * n_classes,) -> (n_samples, n_classes)
    logits = predt.reshape(n_samples, n_classes)
    
    # Compute probabilities
    probs = softmax(logits)
    
    # One-hot encode labels
    y_onehot = np.zeros((n_samples, n_classes), dtype=np.float64)
    y_onehot[np.arange(n_samples), labels] = 1.0
    
    # Get probability of true class: p_t
    p_t = probs[np.arange(n_samples), labels]  # (n_samples,)
    
    # Focal weight: (1 - p_t)^gamma
    focal_weight = (1.0 - p_t) ** gamma  # (n_samples,)
    
    # Gradient of focal loss w.r.t. logit z_j:
    # For true class (y_k = 1):
    #   dL/dz_k = focal_weight * (p_k - 1) + gamma * (1-p_k)^(gamma-1) * p_k * (1-p_k) * log(p_k)
    # For other classes (y_k = 0):
    #   dL/dz_j = focal_weight * p_j - gamma * (1-p_t)^(gamma-1) * p_j * p_t * log(p_t)
    #
    # Simplified form (combining terms):
    log_p_t = np.log(np.clip(p_t, 1e-15, 1.0))  # (n_samples,)
    
    # Base gradient (cross-entropy part): p - y
    grad_ce = probs - y_onehot  # (n_samples, n_classes)
    
    # Focal modulation
    # Term 1: (1-p_t)^gamma * (p - y)
    term1 = focal_weight[:, None] * grad_ce
    
    # Term 2: gamma * (1-p_t)^(gamma-1) * log(p_t) * p_t * (p - y)
    # This simplifies because (p - y) for true class is (p_t - 1)
    focal_derivative = gamma * ((1.0 - p_t) ** (gamma - 1)) * log_p_t * p_t
    term2 = focal_derivative[:, None] * grad_ce
    
    grad = term1 + term2  # (n_samples, n_classes)
    
    # Hessian diagonal approximation
    # Using the standard softmax hessian scaled by focal weight
    # H_jj ≈ focal_weight * p_j * (1 - p_j)
    # This is an approximation; exact focal hessian is more complex
    hess = focal_weight[:, None] * probs * (1.0 - probs)
    
    # Ensure positive hessian for numerical stability
    hess = np.maximum(hess, 1e-6)
    
    # Apply class weights if provided
    if class_weights is not None:
        weight_per_sample = class_weights[labels]  # (n_samples,)
        grad *= weight_per_sample[:, None]
        hess *= weight_per_sample[:, None]
    
    # Flatten back to XGBoost format
    return grad.flatten(), hess.flatten()


def create_focal_objective(gamma: float = 2.0, class_weights: Optional[np.ndarray] = None):
    """
    Factory function to create focal loss objective with fixed parameters.
    
    Parameters
    ----------
    gamma : float
        Focusing parameter. Default 2.0 works well for moderate imbalance.
    class_weights : np.ndarray, optional
        Per-class weights for additional imbalance handling.
    
    Returns
    -------
    callable
        Objective function compatible with xgb.train(obj=...)
    
    Example
    -------
    >>> focal_obj = create_focal_objective(gamma=2.0, class_weights=np.array([1.0, 5.0, 3.0]))
    >>> model = xgb.train(params, dtrain, obj=focal_obj, ...)
    """
    def objective(predt: np.ndarray, dtrain) -> Tuple[np.ndarray, np.ndarray]:
        return xgb_focal_loss_objective(predt, dtrain, gamma=gamma, class_weights=class_weights)
    return objective


def focal_loss_eval(
    predt: np.ndarray,
    dtrain,
    gamma: float = 2.0
) -> Tuple[str, float]:
    """
    Evaluation metric: focal loss value (for monitoring).
    
    Returns
    -------
    Tuple[str, float]
        ("focal_loss", value) for XGBoost eval display
    """
    labels = dtrain.get_label().astype(np.int32)
    n_samples = len(labels)
    n_classes = len(np.unique(labels))
    
    logits = predt.reshape(n_samples, n_classes)
    probs = softmax(logits)
    
    p_t = probs[np.arange(n_samples), labels]
    p_t = np.clip(p_t, 1e-15, 1.0)
    
    focal_weight = (1.0 - p_t) ** gamma
    loss = -focal_weight * np.log(p_t)
    
    return "focal_loss", float(loss.mean())


def create_focal_eval(gamma: float = 2.0):
    """Factory for focal loss evaluation metric."""
    def metric(predt: np.ndarray, dtrain) -> Tuple[str, float]:
        return focal_loss_eval(predt, dtrain, gamma=gamma)
    return metric