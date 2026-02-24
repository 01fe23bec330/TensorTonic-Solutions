import numpy as np

def contrastive_loss(a, b, y, margin=1.0, reduction="mean"):
    """
    Compute Contrastive Loss for Siamese networks.
    
    a, b : embeddings (D,) or (N, D)
    y    : labels (N,) or scalar (0 or 1)
    margin : float
    reduction : "mean" or "sum"
    """
    
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    y = np.asarray(y, dtype=float)
    
    # Ensure (N, D) shape
    if a.ndim == 1:
        a = a[np.newaxis, :]
        b = b[np.newaxis, :]
    
    # Validate labels
    if not np.all((y == 0) | (y == 1)):
        raise ValueError("y must contain only 0 or 1")
    
    # Compute L2 distance
    diff = a - b
    d = np.linalg.norm(diff, axis=1)
    
    # Positive loss: y * d^2
    pos_loss = y * (d ** 2)
    
    # Negative loss: (1-y) * max(0, margin - d)^2
    neg_part = np.maximum(0, margin - d)
    neg_loss = (1 - y) * (neg_part ** 2)
    
    loss = pos_loss + neg_loss
    
    if reduction == "mean":
        return np.mean(loss)
    elif reduction == "sum":
        return np.sum(loss)
    else:
        raise ValueError("reduction must be 'mean' or 'sum'")
    pass