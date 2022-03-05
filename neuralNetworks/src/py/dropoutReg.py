import copy

import numpy as np

def dropout_matrices(layer_dims, num_ex, keep_prob):
    """
    Parameters
    ----------
    layer_dims : List[int]
        layer_dims[l] = number of nodes in layer l
    num_ex : int
        The number of training examples
    keep_prob : List[float]
        keep_prob[l] = The probabilty of keeping a node in layer l

    Returns
    -------
    D : Dict[array_like]
        D[l].shape = (layer_dims[l], num_ex)
        D[l] = a Boolean array
    """
    np.random.seed(1)
    L = len(layer_dims)
    D = {}
    for l in range(L-1):
        D[l] = np.random.rand((layer_dims[l], num_ex))
        D[l] = D[l] < keep_prob[l]
        assert(D[l].shape == (layer_dims[l], num_ex))
    return D

def model(x, y,
            hidden_sizes,
            keep_prob,
            activators,
            num_iters=2500,
            learning_rate=0.1,
            print_cost=False):
    """
    Parameters
    ----------
    Parameters
    ----------
    x : array_like
        x.shape = (layer_dims[0], n)
    y : array_like
        y.shape = (layers_dims[-1], n)
    hidden_sizes : List[int]
        The number nodes layer l = hidden_sizes[l-1]
    activators : List[function]
        activators[l] = activation function of layer l+1
    num_iters : int
        Number of iterations with which our model performs gradient descent
    learning_rate : float
        The learning rate for gradient descent
    print_cost : Boolean
        If True, print the cost every 1000 iterations

    Returns
    -------
    params : Dict[Dict]
        params['w'][l] : array_like
            w[l].shape = (layers[l], layers[l-1])
        params['b'][l] : array_like
            b[l].shape = (layers[l], 1)
    cost : float
        The final cost value for the optimized parameters returned
    """
    n, layers = dim_retrieval(x, y, hidden_sizes)
    params = initialize_parameters(layers)
    for i in range(num_iters):
        D = dropout_matrices(layers, n, keep_prob)
        cache = forward_propagation(params, activators, D, x)
        cost = compute_cost(cache, y)
        grads = backward_propagation(params, activators, cache, D, x, y)
        params = update_parameters(params, grads, learning_rate)

        if print_cost and i % 1000 == 0:
            print(f'Cost after iteration {i}: {cost}')

    return params, cost
