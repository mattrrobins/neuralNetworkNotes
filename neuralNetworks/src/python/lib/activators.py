import numpy as np

ACTIVATORS = ['relu', 'sigmoid', 'tanh', 'linear', 'softmax']

## Activator functions
# The (leaky-)ReLU function
def relu(z, beta=0.0):
    """
    Parameters
    ----------
    z : array_like
    beta : float

    Returns
    -------
    r : array_like
        The (broadcasted) ReLU function when beta=0, the leaky-ReLU otherwise.
    dr : array_like
        The (broadcasted) derivative of the (leaky-)ReLU function
    """
    # Change scalar to array if needed
    z = np.array(z)
    # Compute value of ReLU(z)
    r = np.maximum(z, beta * z)
    # Compute differential ReLU'(z)
    dr = ((~(z < 0)) * 1) + ((z < 0) * beta)
    return r, dr

# The sigmoid function
def sigmoid(z):
    """
    Parameters
    ----------
    z : array_like

    Returns
    -------
    sigma : array_like
        The (broadcasted) value of the sigmoid function evaluated at z
    dsigma : array_like
        The (broadcasted) derivative of the sigmoid function evaluate at z
    """
    # Compute value of sigmoid
    sigma = (1 / (1 + np.exp(-z)))
    # Compute differential of sigmoid
    dsigma = sigma * (1 - sigma)
    return sigma, dsigma

# The hyperbolic tangent function
def tanh(z):
    """
    Parameters
    ----------
    z : array_like

    Returns
    phi : array_like
        The (broadcasted) value of the hyperbolic tangent function evaluated at z
    dphi : array_like
        The (broadcasted) derivative of hyperbolic tangent function evaluated at z
    """
    # Compute value of tanh
    phi = np.tanh(z)
    # Compute differential of tanh
    dphi = 1 - (phi * phi)
    return phi, dphi

# The linear activator function
def linear(z):
    """
    Parameters
    ----------
    z : array_like

    Returns
    -------
    id : array_like
    d_id
    """
    id = z
    d_id = np.ones(z.shape)
    return id, d_id
