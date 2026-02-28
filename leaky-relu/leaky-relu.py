import numpy as np

def leaky_relu(x, alpha=0.01):
    """
    Leaky ReLU activation:
    
    f(x) = x        if x >= 0
         = alpha*x  if x < 0
         
    Returns:
        np.ndarray
    """
    
    x = np.asarray(x, dtype=float)
    return np.where(x >= 0, x, alpha * x)
    pass