import math

def elu(x, alpha=1.0):
    """
    Apply ELU activation element-wise.
    
    x: list of values
    alpha: non-negative float
    
    Returns: list of floats
    """
    
    if alpha < 0:
        raise ValueError("alpha must be >= 0")
    
    result = []
    
    for v in x:
        if v > 0:
            result.append(float(v))
        else:
            result.append(float(alpha * (math.exp(v) - 1.0)))
    
    return result