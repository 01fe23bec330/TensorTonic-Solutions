import numpy as np

def swish(x):
    """
    Swish activation: x * sigmoid(x)
    Fully vectorized and numerically stable.
    
    Returns:
        np.ndarray of floats
    """
    
    x = np.asarray(x, dtype=float)
    
    # Numerically stable sigmoid
    sigmoid = np.where(
        x >= 0,
        1.0 / (1.0 + np.exp(-x)),
        np.exp(x) / (1.0 + np.exp(x))
    )
    
    return x * sigmoid
    pass