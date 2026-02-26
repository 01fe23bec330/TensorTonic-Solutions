import numpy as np

def dice_loss(p, y, eps=1e-8):
    """
    Compute Dice Loss.
    
    p : array-like (predicted probabilities)
    y : array-like (binary mask {0,1})
    eps : small constant for numerical stability
    
    Returns:
        scalar Dice loss
    """
    
    p = np.asarray(p, dtype=float)
    y = np.asarray(y, dtype=float)
    
    # Ensure shapes match
    if p.shape != y.shape:
        raise ValueError("p and y must have the same shape")
    
    # Flatten (works for 1D or 2D)
    p = p.reshape(-1)
    y = y.reshape(-1)
    
    intersection = np.sum(p * y)
    sum_p = np.sum(p)
    sum_y = np.sum(y)
    
    dice = (2.0 * intersection + eps) / (sum_p + sum_y + eps)
    
    return 1.0 - dice
    pass