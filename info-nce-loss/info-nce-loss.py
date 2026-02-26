import numpy as np

def info_nce_loss(Z1, Z2, temperature=0.1):
    """
    Compute InfoNCE Loss.
    
    Z1: (N, D)
    Z2: (N, D)
    temperature: scalar > 0
    
    Returns:
        scalar mean loss
    """
    
    Z1 = np.asarray(Z1, dtype=float)
    Z2 = np.asarray(Z2, dtype=float)
    
    if Z1.shape != Z2.shape:
        raise ValueError("Z1 and Z2 must have the same shape")
    
    if temperature <= 0:
        raise ValueError("temperature must be > 0")
    
    # Similarity matrix (N, N)
    S = np.dot(Z1, Z2.T) / temperature
    
    # Numerically stable softmax
    S_max = np.max(S, axis=1, keepdims=True)
    S_stable = S - S_max
    
    exp_S = np.exp(S_stable)
    denom = np.sum(exp_S, axis=1)
    
    # Positive similarities are diagonal elements
    pos_exp = np.diag(exp_S)
    
    # Compute loss
    loss = -np.log(pos_exp / denom)
    
    return float(np.mean(loss))
    pass