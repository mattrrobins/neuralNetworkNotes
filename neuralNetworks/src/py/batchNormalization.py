#! python3
import copy

import numpy as np

import utils
import activators
from activators import ACTIVATORS


def init_batch_norm_params(layers):
    """
    Parameters
    ----------
    layers : List[int]
        layers[l] = # nodes in layer l
    Returns
    -------
    params : Dict[Dict]
        params['w'][l] : array_like
        params['gamma'][l] : array_like
        params['beta'][l] : array_like
        params['b'] : array_like
    """
    L = len(layers) - 1
    w = {}
    gamma = {}
    beta = {}
    for l in range(1, L):
        w[l] = np.random.randn(layers[l], layers[l - 1])
        gamma[l] = np.ones((layers[l], 1))
        beta[l] = np.zeros((layers[l], 1))

    w[L] = np.random.randn(layers[L], layers[L - 1])
    b = np.zeros((layers[L], 1))

    params = {'w' : w, 'gamma' : gamma, 'beta' : beta, 'b' : b}
    return params


def linear_batch_activation_forward(a_prev, activator, w, gamma, beta):
    """
    Parameters
    ----------

    Returns
    -------
    """
    assert activator in ACTIVATORS, f'{activator} is not a valid activator.'

    z = w @ a_prev

    mu = np.mean(z, axis=1, keepdims=True)
    sigma2 = np.var(z, axis=1, keepdims=True)
    z_norm = (z - mu) / np.sqrt(sigma2 + 1e-8)
    z_batch = gamma * z_norm + beta

    assert(z_batch.shape == z.shape)

    if activator == 'relu':
        a, _ = activators.relu(z_batch)
    elif activator == 'sigmoid':
        a, _ = activators.sigmoid(z_batch)
    elif activator == 'tanh':
        a, _ = activators.tanh(z_batch)

    cache = {'a' : a,
             'z_batch' : z_batch,
             'z_norm' : z_norm,
             'z' : z,
             'mu' : mu,
             'sigma2' : sigma2}

    return  cache

def forward_propagation_batch(x, params, activators):
    """
    Parameters:
    -----------
    Returns
    -------
    """
    # Retrieve parameters
    w = params['w']
    b = params['b']
    gamma = params['gamma']
    beta = params['beta']

    L = len(w)
    n = x.shape[1]

    # Set empty caches
    a = {}
    z = {}
    z_norm = {}
    z_batch = {}
    mu = {}
    sigma2 = {}

    # Initialize a
    a[0] = x
    for l in range(1, L):
        cache = linear_batch_activation_forward(
                    a[l - 1],
                    activators[l -1],
                    w[l],
                    gamma[l],
                    beta[l]
                    )
        a[l] = cache['a']
        z[l] = cache['z']
        z_norm[l] = cache['z_norm']
        z_batch[l] = cache['z_batch']
        mu[l] = cache['mu']
        sigma2[l] = cache['sigma2']

    z[L], a[L] = utils.linear_activation_forward(a[L-1], w[L], b, activators[L-1])

    cache = {'a' : a,
             'z' : z,
             'z_norm' : z_norm,
             'z_batch' : z_batch,
             'mu' : mu,
             'sigma2' : sigma2}
    return cache


def backward_propagation_batch(x, y, params, cache, activators):
    """
    Parameters
    ----------
    Returns
    -------
    """
    ## Retrieve parameters
    a = cache['a']
    z = cache['z']
    mu = cache['mu']
    sigma2 = cache['sigma2']
    z_batch = cache['z_batch']
    z_norm = cache['z_norm']
    w = params['w']
    gamma = params['gamma']
    n = x.shape[1]
    L = len(z)

    ## Compute deltas
    delta = {}
    delta[L] = a[L] - y
    dN = {}
    for l in reversed(range(1, L)):
        delta[l] = utils.linear_activation_backward(delta[l + 1], z_batch[l], w[l + 1], activators[l])
        dN[l] = ((1 - 1 / n)) / np.sqrt(sigma2[l] + 1e-8) - (1 / n) * ((z[l] - mu[l]) ** 2 ) / (sigma2[l] + 1e-8) ** (-3 / 2)

    ## Compute gradients
    dw = {}
    dgamma = {}
    dbeta = {}
    db = (1 / n) * np.sum(delta[L], axis=1, keepdims=True)
    dw[L] = (1 / n) * (delta[L] @ a[L - 1].T)
    for l in range(1, L):
        dw[l] = gamma[l] * (dN[l] * (delta[l] @ a[l - 1]))
        dbeta[l] = (1 / n) * np.sum(delta[l], axis=1, keepdims=True)
        dgamma[l] = (1 / n) * np.sum(z_norm[l] * delta[l], axis=1, keepdims=True)

    grads = {'w' : dw, 'b' : db, 'gamma' : dgamma, 'beta' : dbeta}
    return grads


def update_batch_parameters(params, grads, learning_rate=0.01):
    """
    """
    ## Retrieve parameters
    w = copy.deepcopy(params['w'])
    b = copy.deepcopy(params['b'])
    gamma = copy.deepcopy(params['gamma'])
    beta = copy.deepcopy(params['beta'])
    L = len(w)

    ## Retrieve gradients
    dw = grads['w']
    db = grads['b']
    dgamma = grads['gamma']
    dbeta = grads['beta']

    ## Perform update
    b = b - learning_rate * db
    w[L] = w[L] - learning_rate * dw[L]
    for l in range(1, L):
        w[l] = w[l] - learning_rate * dw[l]
        beta[l] = beta[l] - learning_rate * dbeta[l]
        gamma[l] = gamma[l] - learning_rate *dgamma[l]

    params = {'w' : w, 'b' : b, 'gamma' : gamma, 'beta' : beta}
    return params






def model(x, y,
        hidden_layer_sizes,
        activators,
        batch_size,
        num_iters=10000,
        print_cost=False):
    """
    Parameters
    ----------
    Returns
    -------
    """
    n, layers = utils.dim_retrieval(x, y, hidden_layer_sizes)
    params = init_batch_norm_params(layers)

    mu_cum = {}
    sigma2_cum = {}
    # main loop
    for i in range(num_iters):
        batches = utils.get_batches(x, y, batch_size)
        cost = 0
        for batch in batches:
            x = batch['x']
            y = batch['y']
            forward_cache = forward_propagation_batch(x, params, activators)
            cost += utils.compute_cost(y, params, forward_cache)
            grads = backward_propagation_batch(x, y, params, forward_cache, activators)
            params = update_batch_parameters(params, grads)

            mu = forward_cache['mu']
            sigma2 = forward_cache['sigma2']
            for l in mu.keys():
                mu_cum.setdefault(l, [])
                sigma2_cum.setdefault(l, [])
                mu_cum[l].append(mu[l])
                sigma2_cum[l].append(sigma2[l])

        if print_cost and i % 1000 == 0:
            print(f'Cost after iteration {i}: {cost}')
    mu = {}
    sigma2 = {}
    for l in range(1, L):
        mu[l] = np.mean(mu_cum[l])
        sigma2[l] = batch_size / (batch_size - 1) * np.mean(sigma2_cum)

    return params, cost, mu, sigma2

def eval_model(x, y, params, mu, sigma2):
    """
    """
    w = params['w']
    b = params['b']
    gamma = params['gamma']
    beta = params['beta']
    L = len(w)
    a = {}
    a[0] = x
    for l in range(L):
        z = w[l] @ a[l-1]
        z = (z - mu[l]) / np.sqrt(sigma2[l] + 1e-8)
        z = gamma[l] * z + beta[l]
        a[l] = activators.relu(z)
    z = w[L] @ a[L - 1] + b
    a[L] = activators.sigmooid(z)

    print(f'Predict: {a[L]}|| Actual: {y}')




def main():
    x = np.random.rand(7, 10000)
    y = np.random.rand(1, 10000)
    hidden_layer_sizes = [4, 3, 3]
    activators = ['relu'] * 3 + ['sigmoid']
    batch_size = 2 ** 6

    params, cost, mu, sigma2 = model(x, y, hidden_layer_sizes, activators, batch_size, print_cost=True)

    eval_model(x[:,1], y[:, 1], params, mu, sigma2)


if __name__ == '__main__':
    main()
