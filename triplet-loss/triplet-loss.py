import numpy as np

def triplet_loss(anchor, positive, negative, margin=1.0):
    """
    Compute Triplet Loss using squared Euclidean distance.
    
    anchor   : (N, D) or (D,)
    positive : same shape as anchor
    negative : same shape as anchor
    margin   : float >= 0
    
    Returns:
        float (mean triplet loss)
    """
    
    if margin < 0:
        raise ValueError("margin must be >= 0")
    
    a = np.asarray(anchor, dtype=float)
    p = np.asarray(positive, dtype=float)
    n = np.asarray(negative, dtype=float)
    
    # Handle single vector case
    if a.ndim == 1:
        a = a[np.newaxis, :]
        p = p[np.newaxis, :]
        n = n[np.newaxis, :]
    
    if a.shape != p.shape or a.shape != n.shape:
        raise ValueError("All inputs must have the same shape")
    
    # Squared Euclidean distances
    d_ap = np.sum((a - p) ** 2, axis=1)
    d_an = np.sum((a - n) ** 2, axis=1)
    
    # Triplet loss
    losses = np.maximum(0.0, d_ap - d_an + margin)
    
    return float(np.mean(losses))
    pass