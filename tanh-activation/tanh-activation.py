import numpy as np

def tanh(x):
    """
    Compute tanh activation using:
    tanh(x) = (e^x - e^{-x}) / (e^x + e^{-x})
    
    Returns:
        np.ndarray of floats
    """
    
    x = np.asarray(x, dtype=float)
    
    exp_pos = np.exp(x)
    exp_neg = np.exp(-x)
    
    return (exp_pos - exp_neg) / (exp_pos + exp_neg)
    pass