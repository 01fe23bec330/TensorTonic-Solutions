import numpy as np

def cosine_embedding_loss(x1, x2, label, margin=0.0):
    """
    Compute Cosine Embedding Loss for a single pair.
    
    x1, x2 : array-like vectors (same length)
    label  : 1 (similar) or -1 (dissimilar)
    margin : non-negative float
    
    Returns:
        float loss value
    """
    
    x1 = np.asarray(x1, dtype=float)
    x2 = np.asarray(x2, dtype=float)
    
    # Cosine similarity
    dot = np.dot(x1, x2)
    norm1 = np.linalg.norm(x1)
    norm2 = np.linalg.norm(x2)
    
    cos_sim = dot / (norm1 * norm2)
    
    # Loss computation
    if label == 1:
        loss = 1.0 - cos_sim
    elif label == -1:
        loss = max(0.0, cos_sim - margin)
    else:
        raise ValueError("label must be 1 or -1")
    
    return float(loss)
    # Write code here