import numpy as np

def kl_divergence(p, q, eps=1e-12):
    """
    Compute KL Divergence D_KL(P || Q).
    
    p : array-like, shape (N,)
    q : array-like, shape (N,)
    eps : small constant for numerical stability
    
    Returns:
        float KL divergence
    """
    
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    
    if p.shape != q.shape:
        raise ValueError("p and q must have the same shape")
    
    # Add epsilon to q for stability
    q = q + eps
    
    # Only compute where p > 0 (since 0 * log(0/q) = 0)
    mask = p > 0
    
    kl = np.sum(p[mask] * np.log(p[mask] / q[mask]))
    
    return float(kl)
    pass