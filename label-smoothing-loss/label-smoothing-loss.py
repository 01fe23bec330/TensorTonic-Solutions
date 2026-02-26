import numpy as np

def label_smoothing_loss(predictions, target, epsilon=0.1):
    """
    Compute cross-entropy loss with label smoothing.
    
    predictions : array-like of shape (K,) - probabilities
    target      : int - correct class index
    epsilon     : smoothing factor (0 <= epsilon <= 1)
    
    Returns:
        float loss
    """
    
    p = np.asarray(predictions, dtype=float)
    K = p.shape[0]
    
    if not (0 <= target < K):
        raise ValueError("Invalid target index")
    
    # Build smoothed target distribution
    q = np.full(K, epsilon / K, dtype=float)
    q[target] = (1 - epsilon) + (epsilon / K)
    
    # Cross-entropy
    loss = -np.sum(q * np.log(p))
    
    return float(loss)