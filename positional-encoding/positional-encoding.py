import numpy as np

def positional_encoding(seq_len, d_model, base=10000.0):
    """
    Return PE of shape (seq_len, d_model) using sin/cos formulation.
    Odd d_model -> last column is sin.
    """
    # Positions: (seq_len, 1)
    pos = np.arange(seq_len, dtype=float)[:, np.newaxis]
    
    # Even dimension indices: 0,2,4,...
    even_dims = np.arange(0, d_model, 2, dtype=float)
    
    # Compute denominator term: base^(2i/d_model)
    denom = base ** (even_dims / d_model)
    
    # Angles matrix: (seq_len, ceil(d_model/2))
    angles = pos / denom
    
    # Initialize output
    pe = np.zeros((seq_len, d_model), dtype=float)
    
    # Fill even indices with sin
    pe[:, 0::2] = np.sin(angles)
    
    # Fill odd indices with cos (only if exist)
    if d_model > 1:
        pe[:, 1::2] = np.cos(angles[:, :pe[:, 1::2].shape[1]])
    
    return pe# Write code here
    pass