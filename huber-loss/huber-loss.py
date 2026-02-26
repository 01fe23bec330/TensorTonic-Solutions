import numpy as np

def huber_loss(y_true, y_pred, delta=1.0):
    """
    Compute mean Huber Loss.
    
    y_true : array-like
    y_pred : array-like
    delta  : positive float (threshold)
    
    Returns:
        float (mean Huber loss)
    """
    
    if delta <= 0:
        raise ValueError("delta must be positive")
    
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    
    if y_true.shape != y_pred.shape:
        raise ValueError("Shapes of y_true and y_pred must match")
    
    # Error
    e = y_true - y_pred
    abs_e = np.abs(e)
    
    # Piecewise definition
    quadratic = 0.5 * e**2
    linear = delta * (abs_e - 0.5 * delta)
    
    loss = np.where(abs_e <= delta, quadratic, linear)
    
    return float(np.mean(loss))
    pass