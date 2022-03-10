import copy

import numpy as np

import activators
from activators import ACTIVATORS

# Preliminary functions for our model
def dim_retrieval(x, y, hidden_sizes):
    """
    Parameters
    ----------
    x : array_like
        x.shape = (layers[0], n)
    y : array_like
        y.shape = (layers[L], n)
    hidden_sizes : List[int]
        hidden_sizes[i-1] = The number nodes layer i
    Returns
    -------
    n : int
        The number of training examples
    layers : List
        layer[l] = # nodes in layer l

    """
    m, n = x.shape
    assert(y.shape[1] == n)
    K = y.shape[0]
    layers = [m]
    layers.extend(hidden_sizes)
    layers.append(K)

    return n, layers

## Initialize parameters using the size of each layer
def initialize_parameters_random(layers):
    """
    Parameters
    ----------
    layers : List[int]
        layers[l] = # nodes in layer l
    Returns
    -------
    params : Dict[Dict]
        w[l] : array_like
            dwl.shape = (layers[l], layers[l-1])
        b[l] : array_like
            dbl.shape = (layers[l], 1)
    """
    w = {}
    b = {}
    for l in range(1, len(layers)):
        w[l] = np.random.randn(layers[l], layers[l - 1]) * 0.01
        b[l] = np.zeros((layers[l], 1))
    params = {'w' : w, 'b' : b}
    return params

def forward_propagation(x, params):
    """
    Parameters
    ----------
    x : array_like
        x.shape = (m_x, n)
    params : Dict[Dict]
        w[l] : array_like
            w[l].shape = (layers[l], layers[l-1])
        b[l] : array_like
            b[l].shape = (layers[l], 1)
    Returns
    -------
    a2 : array_like
        a2.shape = (m_y, n)
    cache : Dict
        cache['z1'] : array_like
            z1.shape = (m_h, n)
        cache['a1'] : array_like
            a1.shape = (m_h, n)
        cache['z2'] : array_like
            z2.shape = (m_y, n)
        cache['a2'] = a2
    """

    # Retrieve parameters
    w = params['w']
    b = params['b']
    w1 = w[1]
    b1 = b[1]
    w2 = w[2]
    b2 = b[2]

    # Auxiliary computations
    z1 = w1 @ x + b1
    a1, _1 = activators.tanh(z1)
    z2 = w2 @ a1 + b2
    a2, _2 = activators.sigmoid(z2)

    assert(a1.shape == (w1.shape[0], x.shape[1]))
    assert(a2.shape == (w2.shape[0], a1.shape[1]))

    cache = {'z1' : z1,
             'a1' : a1,
             'z2' : z2,
             'a2' : a2}

    return a2, cache

def compute_cost(a2, y):
    """
    Parameters
    ----------
    a2 : array_like
        a2.shape = (m_y, n)
    y : array_like
        y.shape = (m_y, n)
    Returns
    -------
    cost : float
        The cost evaluated at y and a2
    """
    n = y.shape[1]
    cost = (-1 / n) * (np.sum(y * np.log(a2)) + np.sum((1 - y) * np.log(1 - a2)))
    cost = float(np.squeeze(cost))  # Makes sure we return a float

    return cost

def backward_propagation(params, cache, x, y):
    """
    Parameters
    ----------
    params : Dict[Dict]
        w[l] : array_like
            dwl.shape = (layers[l], layers[l-1])
        b[l] : array_like
            dbl.shape = (layers[l], 1)
    cache : Dict
        cache['z1'] : array_like
            z1.shape = (m_h, n)
        cache['a1'] : array_like
            a1.shape = (m_h, n)
        cache['z2'] : array_like
            z2.shape = (m_y, n)
        cache['a2'] = a2
    x : array_like
        x.shape = (m_x, n)
    y : array_like
        y.shape = (m_y, n)
    Returns
    -------
    grads : Dict
        grads['dw2'] : array_like
            dw2.shape = (m_y, m_h)
        grads['db2'] : array_like
            db2.shape = (m_y, 1)
        grads['dw1'] : array_like
            dw1.shape = (m_h, m_x)
        grads['db1'] : array_like
            db1.shape = (m_h, 1)
    """
    # Retrieve parameters
    w = params['w']
    w1 = w[1]
    w2 = w[2]

    # Set dimensional constants
    m_x, n = x.shape
    m_y, m_h = w2.shape

    # Retrieve node outputs
    a1 = cache['a1']
    a2 = cache['a2']

    # Auxiliary Computations
    delta2 = a2 - y
    assert(delta2.shape ==(m_y, n))
    d_tanh = 1 - (a1 * a1)
    assert(d_tanh.shape == (m_h, n))
    delta1 = (w2.T @ delta2) * d_tanh
    assert(delta1.shape == (m_h, n))

    # Gradient computations
    dw = {}
    db = {}
    dw[2] = (1 / n) * delta2 @ a1.T
    db[2] = (1 / n) * np.sum(delta2, axis=1, keepdims=True)
    dw[1] = (1 / n) * delta1 @ x.T
    db[1] = (1 / n) * np.sum(delta1, axis=1, keepdims=True)

    # Combine and return dict
    grads = {'dw' : dw, 'db' : db}
    return grads

def update_parameters(params, grads, learning_rate=1.2):
    """
    Parameters
    ----------
    params : Dict
        params['w2'] : array_like
            w2.shape = (m_y, m_h)
        params['b2'] : array_like
            b2.shape = (m_y, 1)
        params['w1'] : array_like
            w1.shape = (m_h, m_x)
        params['b1'] : array_like
            b1.shape = (m_h, 1)
    grads : Dict
        grads['dw2'] : array_like
            dw2.shape = (m_y, m_h)
        grads['db2'] : array_like
            db2.shape = (m_y, 1)
        grads['dw1'] : array_like
            dw1.shape = (m_h, m_x)
        grads['db1'] : array_like
            db1.shape = (m_h, 1)
    learning_rate : float
        Default = 1.2
    Returns
    -------
    params : Dict
        params['w2'] : array_like
            w2.shape = (m_y, m_h)
        params['b2'] : array_like
            b2.shape = (m_y, 1)
        params['w1'] : array_like
            w1.shape = (m_h, m_x)
        params['b1'] : array_like
            b1.shape = (m_h, 1)
    """
    # Retrieve parameters
    w = copy.deepcopy(params['w'])
    b = params['b']

    # Retrieve gradients
    dw = grads['dw']
    db = grads['db']

    # Perform update
    w[2] = w[2] - learning_rate * dw[2]
    b[2] = b[2] - learning_rate * db[2]
    w[1] = w[1] - learning_rate * dw[1]
    b[1] = b[1] - learning_rate * db[1]

    # Combine and return dict
    params = {'w' : w, 'b' : b}
    return params


# The main neural network training model
def model(x, y, hidden_sizes, num_iters=10000, print_cost=False):
    """
    Parameters
    ----------
    x : array_like
        x.shape = (m_x, n)
    y : array_like
        y.shape = (m_y. n)
    hidden_sizes : int
        Number of nodes in the single hidden layer
    num_iters : int
        Number of iterations with which our model performs gradient descent
    print_cost : Boolean
        If True, print the cost every 1000 iterations
    Returns
    -------
    params : Dict[Dict[array_like]]
        params['w'][2] : array_like
            w[2].shape = (m_y, m_h)
        params['b'][2] : array_like
            b[2].shape = (m_y, 1)
        params['w'][1] : array_like
            w[1].shape = (m_h, m_x)
        params['b'][1] : array_like
            b[1].shape = (m_h, 1)
    """
    # Set dimensional constants
    n, layers = dim_retrieval(x, y, hidden_sizes)
    # initialize parameters
    params = initialize_parameters_random(layers)

    # main loop for gradient descent
    for i in range(num_iters):
        a2, cache = forward_propagation(x, params)
        cost = compute_cost(a2, y)
        grads = backward_propagation(params, cache, x, y)
        params = update_parameters(params, grads)

        if print_cost and i % 1000 == 0:
            print(f'Cost after iteration {i}: {cost}')

    return params

# Using our model to obtain predictions
def predict(params, x):
    """
    Parameters
    ----------
    params : Dict
        params['w2'] : array_like
            w2.shape = (m_y, m_h)
        params['b2'] : array_like
            b2.shape = (m_y, 1)
        params['w1'] : array_like
            w1.shape = (m_h, m_x)
        params['b1'] : array_like
            b1.shape = (m_h, 1)
    x : array_like
        x.shape = (m_x, n)

    Returns
    -------
    predictions : array_like
        predictions.shape = (m_y, n)
    """
    a2, _ = forward_propagation(x, params)
    predictions = np.zeros(a2.shape)
    predictions[~(a2 < 0.5)] = 1

    return predictions






def main():
    x = np.random.rand(10, 200)
    y = np.random.rand(1, 200)
    hidden_sizes = [7]
    params = model(x, y, hidden_sizes)
    print(params)



if __name__ == '__main__':
    main()
