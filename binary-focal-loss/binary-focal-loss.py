import numpy as np

def binary_focal_loss(predictions, targets, alpha=1.0, gamma=2.0):
    """
    Compute mean Binary Focal Loss.
    
    predictions : array-like (0 < p < 1)
    targets     : array-like (0 or 1)
    alpha       : balancing factor (> 0)
    gamma       : focusing parameter (>= 0)
    
    Returns:
        float (mean focal loss)
    """
    
    p = np.asarray(predictions, dtype=float)
    y = np.asarray(targets, dtype=float)
    
    if p.shape != y.shape:
        raise ValueError("predictions and targets must have the same shape")
    
    # Compute p_t
    p_t = np.where(y == 1, p, 1 - p)
    
    # Compute focal loss
    loss = -alpha * ((1 - p_t) ** gamma) * np.log(p_t)
    
    return float(np.mean(loss))