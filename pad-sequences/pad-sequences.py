import numpy as np

def pad_sequences(seqs, pad_value=0, max_len=None):
    """
    Returns: np.ndarray of shape (N, L) where:
      N = len(seqs)
      L = max_len if provided else max(len(seq) for seq in seqs) or 0
    """
    
    # If empty input
    if len(seqs) == 0:
        return np.zeros((0, 0), dtype=int)
    
    # Determine target length
    if max_len is None:
        max_len = max(len(seq) for seq in seqs)
    
    N = len(seqs)
    
    # Initialize output with pad_value
    output = np.full((N, max_len), pad_value, dtype=int)
    
    # Fill with truncated sequences
    for i, seq in enumerate(seqs):
        truncated = seq[:max_len]
        output[i, :len(truncated)] = truncated
    
    return output
    pass