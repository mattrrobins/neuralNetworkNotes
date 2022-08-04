#! python3

import numpy as np

LOSS_FUNCTIONS = {"lse": lse, "log_loss": log_loss, "cross_entropy": cross_entropy}

## Loss functions

# The least-squared-error function
def lse(a, y):
    """
    Parameters:
    -----------
    a : array_like
    y : array_like
        a.shape == y.shape

    Returns:
    --------
    loss : array_like
    rloss : array_like
        rloss.shape == a.shape
    """
    loss = ((a - y) ** 2) / 2
    rloss = a - y
    return loss, rloss


# The log-loss function for binary classification
def log_loss(a, y):
    """
    Parameters:
    -----------
    a : array_like
    y : array_like
        a.shape == y.shape

    Returns:
    --------
    loss : array_like
    rloss : array_like
        rloss.shape == a.shape
    """
    loss = -1 * (y * np.log(a) + (1 - y) * np.log(1 - a))
    rloss = -(y / a) + (1 - y) / (1 - a)
    return loss, rloss


## The cross-entropy loss function
def cross_entropy(a, y, eps=1e-8):
    """
    Parameters:
    -----------
    a : array_like
    y : array_like
        a.shape == y.shape
    eps : float
        Default = 10^{-8} # For stability

    Returns:
    --------
    loss : array_like
    r_loss : array_like
        rloss.shape == a.shape
    """
    assert a.shape == y.shape, "a and y have different shapes"

    a = np.clip(a, eps, 1 - eps)
    loss = -1 * np.sum(y * np.log(a), axis=0)
    rloss = -1 * y / a
    return loss, rloss
