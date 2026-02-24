import numpy as np

def cross_entropy_loss(y_true, y_pred):
    """
    Compute average multi-class cross-entropy loss.
    
    y_true: shape (N,)  -> class indices
    y_pred: shape (N, K) -> predicted probabilities
    
    Returns: float
    """
    
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    N = y_true.shape[0]
    
    # Select probability of correct class for each sample
    correct_class_probs = y_pred[np.arange(N), y_true]
    
    # Compute average negative log likelihood
    loss = -np.mean(np.log(correct_class_probs))
    
    return loss
    pass