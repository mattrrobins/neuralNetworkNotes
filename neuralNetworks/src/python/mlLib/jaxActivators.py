#! python3

import jax.numpy as jnp
import jax
from jax import jit



@jit
def linear(z):
    """
    Parameters
    ----------
    z : array_like

    Returns
    -------
    z : array_like
    """
    return z


@jit
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
    """
    z = jnp.array(z)
    r = jnp.maximum(z, beta * z)
    return r

## This might be the preferred implementation for sigmoid for autodifferentiation?s
@jit
def sigmoid_better(z):
    """
    Parameters
    ----------
    z : array_like

    Returns
    -------
    sigma : array_like
        The (broadcasted) value of the sigmoid function evaluated at z
    """
    return 0.5 * (jnp.tanh(z / 2) + 1)

## This is a bad implementation for autodifferentiation purposes... avoid if possible
@jit
def sigmoid_bad(z):
    """
    Parameters
    ----------
    z : array_like

    Returns
    -------
    sigma : array_like
        The (broadcasted) value of the sigmoid function evaluated at z
    """
    z = jnp.array(z)
    sigma = 1 / (1 + jnp.exp(-z))
    return sigma


@jit
def softmax(z):
    pass





## Activator fixed dictionary
ACTIVATORS = {'relu' : relu,
              'sigmoid' : jax.nn.sigmoid,
              'tanh' : jnp.tanh,
              'linear' : linear,
              'softmax' : softmax}





if __name__ == '__main__':
    z = [-1, 1, -2, 2, -3, 3, -4, 4, -5, 5]
    z = jnp.array(z)
    for k, v in ACTIVATORS.items():
        if k == 'relu':
            for beta in [0.0, 0.5, 1, 1.5, 2]:
                print(v(z, beta))
        print(v(z))



