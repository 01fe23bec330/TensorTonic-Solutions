import numpy as np

def mean_squared_error(y_pred, y_true):
    """
    Compute Mean Squared Error (MSE).
    
    y_pred: array-like, shape (N,)
    y_true: array-like, shape (N,)
    
    Returns:
        float (MSE) or None if shapes mismatch
    """
    
    y_pred = np.asarray(y_pred, dtype=float)
    y_true = np.asarray(y_true, dtype=float)
    
    # Shape validation
    if y_pred.shape != y_true.shape:
        return None
    
    return float(np.mean((y_pred - y_true) ** 2))
    pass
