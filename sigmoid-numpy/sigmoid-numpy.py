import numpy as np

def sigmoid(x):
    """
    Vectorized sigmoid function.
    Works for scalars, lists, and NumPy arrays.
    Returns a NumPy array of floats.
    """
    x = np.array(x, dtype=float)   # Ensure NumPy array
    return 1 / (1 + np.exp(-x))
   
    # Write code here
    pass