import numpy as np

def focal_loss(p, y, gamma=2.0):
    """
    Binary Focal Loss (mean reduction).
    
    p : np.ndarray of shape (N,) -> predicted probabilities (0 < p < 1)
    y : np.ndarray of shape (N,) -> binary labels {0,1}
    gamma : focusing parameter (>= 0)
    
    Returns:
        scalar mean loss
    """
    
    p = np.asarray(p, dtype=float)
    y = np.asarray(y, dtype=float)
    
    # Compute loss components
    loss_pos = - (1 - p)**gamma * y * np.log(p)
    loss_neg = - (p**gamma) * (1 - y) * np.log(1 - p)
    
    loss = loss_pos + loss_neg
    
    return np.mean(loss)
    pass