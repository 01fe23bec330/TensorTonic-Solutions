import math

def selu(x):
    """
    Apply SELU activation element-wise.
    Returns a list of floats.
    """
    
    # Exact constants (must use these)
    lam = 1.0507009873554804934193349852946
    alpha = 1.6732632423543772848170429916717
    
    result = []
    
    for v in x:
        if v > 0:
            result.append(lam * v)
        else:
            result.append(lam * alpha * (math.exp(v) - 1.0))
    
    return result
    pass
