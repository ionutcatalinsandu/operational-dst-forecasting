import numpy as np


def asymmetric_loss(y_true, y_pred, alpha=2.0, beta=1.0):
    """
    Asymmetric loss function that penalizes undershooting more than overshooting.
    
    Args:
        y_true: True DST values
        y_pred: Predicted DST values  
        alpha: Penalty multiplier for undershooting (alpha > beta encourages higher predictions)
        beta: Penalty multiplier for overshooting
        
    Returns:
        Loss value
    """
    residual = y_true - y_pred
    
    # Undershooting (residual > 0): model predicted higher DST than reality (less negative)
    # This is bad for storm prediction, so penalize with alpha
    undershooting_mask = residual > 0
    
    # Overshooting (residual < 0): model predicted lower DST than reality (more negative)  
    # This is safer for storm prediction, so penalize less with beta
    overshooting_mask = residual <= 0
    
    loss = (alpha * (residual ** 2) * undershooting_mask + 
            beta * (residual ** 2) * overshooting_mask)
    
    return np.mean(loss)


def asymmetric_loss_gradient_hessian(y_true, y_pred, alpha=2.0, beta=1.0):
    """
    Gradient and Hessian for asymmetric loss (required for XGBoost custom objective).
    """
    residual = y_true - y_pred
    
    # Gradient (first derivative)
    grad = np.where(residual > 0, 
                   -2 * alpha * residual,   # Undershooting
                   -2 * beta * residual)    # Overshooting
    
    # Hessian (second derivative) 
    hess = np.where(residual > 0,
                   2 * alpha,               # Undershooting
                   2 * beta)                # Overshooting
    
    return grad, hess


def xgb_asymmetric_objective(y_pred, dtrain, alpha=2.0, beta=1.0):
    """
    XGBoost custom objective for asymmetric loss.
    """
    y_true = dtrain.get_label()
    grad, hess = asymmetric_loss_gradient_hessian(y_true, y_pred, alpha, beta)
    return grad, hess


# ============================================================================
# Focal Loss for Binary Classification
# ============================================================================

def sigmoid(x):
    """Numerically stable sigmoid function."""
    return np.where(x >= 0, 
                    1 / (1 + np.exp(-x)),
                    np.exp(x) / (1 + np.exp(x)))


def focal_loss_binary(y_true, p_pred, gamma=2.0, alpha=0.25):
    """
    Focal loss for binary classification.
    
    L(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    
    where p_t = p if y=1, else (1-p)
    
    Args:
        y_true: True binary labels (0 or 1)
        p_pred: Predicted probabilities
        gamma: Focusing parameter (higher = more focus on hard examples)
        alpha: Weighting factor for class imbalance (weight for positive class)
        
    Returns:
        Loss value
    """
    eps = 1e-7
    p_pred = np.clip(p_pred, eps, 1 - eps)
    
    # Compute p_t (probability of correct class)
    p_t = y_true * p_pred + (1 - y_true) * (1 - p_pred)
    
    # Compute alpha_t (class weight)
    alpha_t = y_true * alpha + (1 - y_true) * (1 - alpha)
    
    # Focal loss
    loss = -alpha_t * (1 - p_t) ** gamma * np.log(p_t)
    
    return np.mean(loss)


def focal_loss_gradient_hessian(y_true, raw_pred, gamma=2.0, alpha=0.25):
    """
    Gradient and Hessian for focal loss with respect to raw logits.
    """
    eps = 1e-7
    
    p = sigmoid(raw_pred)
    p = np.clip(p, eps, 1 - eps)
    
    # dFL/dp for y=1: α(1-p)^(γ-1)[γ log(p) - (1-p)/p]
    grad_p_pos = alpha * (1 - p) ** (gamma - 1) * (
        gamma * np.log(p) - (1 - p) / p  # MINUS sign
    )
    
    # dFL/dp for y=0: (1-α)p^(γ-1)[-γ log(1-p) + p/(1-p)]
    grad_p_neg = (1 - alpha) * p ** (gamma - 1) * (
        -gamma * np.log(1 - p) + p / (1 - p)
    )
    
    # Chain rule: dFL/d(logit) = dFL/dp * dp/d(logit) = dFL/dp * p(1-p)
    grad_pos = grad_p_pos * p * (1 - p)
    grad_neg = grad_p_neg * p * (1 - p)
    
    grad = y_true * grad_pos + (1 - y_true) * grad_neg
    
    # Hessian approximation (using absolute value for stability)
    hess_pos = alpha * p * (1 - p) * (1 - p) ** (gamma - 1) * (gamma + 1)
    hess_neg = (1 - alpha) * p * (1 - p) * p ** (gamma - 1) * (gamma + 1)
    hess = y_true * hess_pos + (1 - y_true) * hess_neg
    hess = np.maximum(hess, eps)
    
    return grad, hess


def xgb_focal_objective(y_pred, dtrain, gamma=2.0, alpha=0.25):
    """
    XGBoost custom objective for focal loss (binary classification).
    
    Args:
        y_pred: Raw predictions (logits) from XGBoost
        dtrain: Training DMatrix
        gamma: Focusing parameter (default=2.0)
        alpha: Class weight for positive class (default=0.25)
        
    Returns:
        grad, hess: Gradient and Hessian arrays
    """
    y_true = dtrain.get_label()
    grad, hess = focal_loss_gradient_hessian(y_true, y_pred, gamma, alpha)
    return grad, hess


def xgb_focal_eval(y_pred, dtrain, gamma=2.0, alpha=0.25):
    """
    XGBoost custom evaluation metric for focal loss.
    
    Returns:
        name, value: Metric name and value
    """
    y_true = dtrain.get_label()
    p_pred = sigmoid(y_pred)
    loss = focal_loss_binary(y_true, p_pred, gamma, alpha)
    return 'focal_loss', float(loss)


def create_focal_objective(gamma=2.0, alpha=0.25):
    """
    Factory function to create focal objective with fixed hyperparameters.
    
    Usage:
        focal_obj = create_focal_objective(gamma=2.0, alpha=0.25)
        xgb.train(..., obj=focal_obj)
    """
    def objective(y_pred, dtrain):
        return xgb_focal_objective(y_pred, dtrain, gamma, alpha)
    return objective


def create_focal_eval(gamma=2.0, alpha=0.25):
    """
    Factory function to create focal evaluation metric with fixed hyperparameters.
    
    Usage:
        focal_eval = create_focal_eval(gamma=2.0, alpha=0.25)
        xgb.train(..., custom_metric=focal_eval)
    """
    def eval_metric(y_pred, dtrain):
        return xgb_focal_eval(y_pred, dtrain, gamma, alpha)
    return eval_metric