import numpy as np

def wasserstein_critic_loss(real_scores, fake_scores):
    """
    Compute Wasserstein critic loss for WGAN.
    
    real_scores : np.ndarray (critic outputs for real samples)
    fake_scores : np.ndarray (critic outputs for fake samples)
    
    Returns:
        float loss
    """
    
    real_scores = np.asarray(real_scores, dtype=float)
    fake_scores = np.asarray(fake_scores, dtype=float)
    
    loss = np.mean(fake_scores) - np.mean(real_scores)
    
    return float(loss)
    pass