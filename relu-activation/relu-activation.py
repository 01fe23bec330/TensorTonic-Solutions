import numpy as np

def relu(x):
    """
    ReLU activation:
    ReLU(x) = max(0, x)
    
    Returns:
        np.ndarray of floats
    """
    
    x = np.asarray(x, dtype=float)
    return np.maximum(0.0, x)
    pass