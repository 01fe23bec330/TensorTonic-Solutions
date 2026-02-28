import numpy as np
import math

def gelu(x):
    """
    Exact GELU using math.erf (vectorized safely).
    """
    
    x = np.asarray(x, dtype=float)
    
    # Apply math.erf elementwise (fully vectorized)
    erf_values = np.array([math.erf(v / math.sqrt(2.0)) for v in x.flat])
    erf_values = erf_values.reshape(x.shape)
    
    return 0.5 * x * (1.0 + erf_values)
    pass