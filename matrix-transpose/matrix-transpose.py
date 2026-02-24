import numpy as np

def matrix_transpose(A):
    """
    Return the transpose of matrix A (swap rows and columns).
    Does NOT modify original matrix.
    """
    
    A = np.array(A)  # Ensure NumPy array
    
    N, M = A.shape
    
    # Create new matrix with swapped shape
    result = np.empty((M, N), dtype=A.dtype)
    
    # Manual transpose
    for i in range(N):
        for j in range(M):
            result[j, i] = A[i, j]
    
    return result
  
    # Write code here
    pass
