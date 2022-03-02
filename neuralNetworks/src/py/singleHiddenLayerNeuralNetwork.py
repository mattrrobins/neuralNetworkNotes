import copy

import numpy as np

# Activator functions

def sigmoid(z):
    """
    Parameters
    ----------
    z : array_like

    Returns
    -------
    sigma : array_like
        The value of the sigmoid function evaluated at z
    ds : array_like
        The differential of the sigmoid function evaluate at z
    """
    # Compute value of sigmoid
    sigma = (1 / (1 + np.exp(-z)))
    # Compute differential of sigmoid
    ds = sigma * (1 - sigma)
    return sigma, ds

# Preliminary functions for our model
def layer_shapes(x, y, hidden_layer_size):
    """
    Parameters
    ----------
    x : array_like
        x.shape = (m_x, n)
    y : array_like
        y.shape = (m_y, n)
    hidden_layer_size : int
        The number nodes in the hidden layer
    Returns
    -------
    n : int
        The number of training examples
    m_x : int
        The number of input features
    m_h : The number of nodes in the hidden layer
    m_y : The number of nodes in the output layer
    """
    m_x, n = x.shape
    assert(y.shape[1] == n)
    m_y = y.shape[0]
    m_h = hidden_layer_size
    return n, m_x, m_h, m_y



def initialize_parameters(m_x, m_h, m_y):
    """
    Parameters
    ----------
    m_x : int
        The number of input features
    m_h : int
        The number of nodes in the hidden layer
    m_y : int
        The number of nodes in the output layer

    Returns
    -------
    params : Dict
        w1 : array_like
            w1.shape = (m_h, m_x)
        b1 : array_like
            b1.shape = (m_h, 1)
        w2 : array_like
            w2.shape= (m_y, m_h)
        b2 : array_like
            b2.shape = (m_y, 1)
    """
    w1 = np.random.randn(m_h, m_x) * 0.01
    b1 = np.zeros((m_h, 1))
    w2 = np.random.randn(m_y, m_h) * 0.01
    b2 = np.zeros((m_y, 1))

    params = {'w1' : w1,
              'b1' : b1,
              'w2' : w2,
              'b2' : b2}

    return params

def forward_propagation(x, params):
    """
    Parameters
    ----------
    x : array_like
        x.shape = (m_x, n)
    params : Dict
        params['w1'] : array_like
            w1.shape = (m_h, m_x)
        params['b1'] : array_like
            b1.shape = (m_h, 1)
        params['w2'] : array_like
            w2.shape = (m_y, m_h)
        params['b2'] : array_like
            b2.shape = (m_y, 1)
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
    w1 = params['w1']
    b1 = params['b1']
    w2 = params['w2']
    b2 = params['b2']

    # Auxiliary computations
    z1 = w1 @ x + b1
    a1 = np.tanh(z1)
    z2 = w2 @ a1 + b2
    a2 = sigmoid(z2)

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
    params : Dict
        params['w2'] : array_like
            w2.shape = (m_y, m_h)
        params['b2'] : array_like
            b2.shape = (m_y, 1)
        params['w1'] : array_like
            w1.shape = (m_h, m_x)
        params['b1'] : array_like
            b1.shape = (m_h, 1)
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
    w1 = params['w1']
    w2 = params['w2']

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
    delta1 = (w2.T @ delta1) * d_tanh
    assert(delta1.shape == (m_h, n))

    # Gradient computations
    dw2 = (1 / n) * delta2 @ a1.T
    db2 = (1 / n) * np.sum(delta2, axis=1, keepdims=True)
    dw1 = (1 / n) * delta1 @ x.T
    db1 = (1 / n) * np.sum(delta1, axis=1, keepdims=True)

    # Combine and return dict
    grads = {'dw2' : dw2,
             'db2' : db2,
             'dw1' : dw1,
             'db1' : db1}
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
    w2 = copy.deepcopy(params['w2'])
    b2 = params['b2']
    w1 = copy.deepcopy(params['w1'])
    b1 = params['b1']

    # Retrieve gradients
    dw2 = grads['dw2']
    db2 = grads['db2']
    dw1 = grads['dw1']
    db1 = grads['db1']

    # Perform update
    w2 = w2 - learning_rate * dw2
    b2 = b2 - learning_rate * db2
    w1 = w1 - learning_rate * dw1
    b1 = b1 - learning_rate * db1

    # Combine and return dict
    params = {'w2' : w2,
              'b2' : b2,
              'w1' : w1,
              'b1' : b1}
    return params


# The main neural network training model
def model(x, y, num_hidden_layer, num_iters=10000, print_cost=False):
    """
    Parameters
    ----------
    x : array_like
        x.shape = (m_x, n)
    y : array_like
        y.shape = (m_y. n)
    num_hidden_layer : int
        Number of nodes in the single hidden layer
    num_iters : int
        Number of iterations with which our model performs gradient descent
    print_cost : Boolean
        If True, print the cost every 1000 iterations
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
    # Set dimensional constants
    n, m_x, m_h, m_y = layer_shapes(x, y, num_hidden_layer)
    # initialize parameters
    params = initialize_parameters(m_x, m_h, m_y)

    # main loop for gradient descent
    for i in range(num_iters):
        a2, cache = forward_propagation(X, params)
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


















###### Extras #######
def reshape_labels(num_labels, y):
    """
    Parameters
    ----------
    num_labels : int
        The number of possible labels the output y may take
    y : array_like
        y.size = n
        y[i] takes values in {1,2,...,num_labels}
    Returns
    Y : array_like
        Y.shape = (num_lables, n)
        Y[i][j] = 1 if y[j] = i, Y[i][j] = 0 otherwise
    -------
    """

    if num_labels <= 2:
        return y
    else:
        omega = []
        for i in range(num_labels):
            omega.append(np.eye(1, num_labels, i))  # the standard i-th basis vector in \mathbb{R}^{num_labels}

        Y = np.concatenate([omega[i] for i in y], axis=0).T
        for i in range(num_labels):
            for j in range(n):
                if y[j] == i:
                    assert Y[i][j] == 1
                else:
                    assert Y[i][j] == 0
        return Y

def relu(z, beta=0.0):
    """
    Parameters
    ----------
    z : array_like
    beta : float

    Returns
    -------
    r : array_like
        The ReLU function when beta=0, the leaky-ReLU otherwise.
    dr : array_like
        The differential of the ReLU function
    """
    # Change scalar to array if needed
    z = np.array(z)
    # Compute value of ReLU(z)
    r = np.maximum(z, beta * z)
    # Compute differential ReLU'(z)
    dr = np.zeros(z.size)
    dr[z < 0] = beta
    dr[~(z < 0)] = 1
    return r, dr

def cost_function(params,
                  input_layer_size,
                  hidden_layer_size,
                  num_labels,
                  x, y, lambda_=0.0):
    """
    Parameters
    ----------
    params : array_like
        Our parameters flattened into a single rank 1 array
    input_layer_size : int
        The number of features for our input layer
    hidden_layer_size : int
        The number of nodes for our hidden layer
    num_labels : int
        The number of classifcation labels for our target output
    x : array_like
        x.shape = (input_layer_size, n) where n is the number of training examples
    y : array_like
        y.shape = (num_lables, n)
    lambda_ : float
        Default: 0.0 - Represents a model without regularization
        The regularization parameter to be trained on a cross-validation set

    Returns
    -------
    d : Dict
        cost : float
            The value of the cost function evaulated at w1, b1, w2, b2
        dw1 : array_like
            dw1.shape = (hidden_layer_size, input_layer_size)
            The gradient of J with respect to w1
        db1 : array_like
            db1.shape = (hidden_layer_size, 1)
            The gradient of J with respect to b1
        dw2 : array_like
            dw2.shape = (num_labels, hidden_layer_size)
            The gradient of J with respect to w2
        db2: array_like
            db2.shape = (num_labels, 1)
            The gradient of J with respect to b2
    """
    # Specialization for binary classification since the second activator
    # a2[2] = 1 - a2[1], there is no loss by only using one.
    if num_labels == 2:
        num_lables = 1

    # Set dimensions, parameters and labels
    n = x.shape[1]

    d = reshape_params(params, input_layer_size, hidden_layer_size, num_labels)
    w1, w2, b1, b2 = d['w1'], d['w2'], d['b1'], d['b2']
    assert w1.shape == (hidden_layer_size, input_layer_size)
    assert w2.shape == (num_labels, hidden_layer_size)
    assert b1.shape == (hidden_layer_size, 1)
    assert b2.shape == (num_labels, 1)

    y = reshape_labels(num_labels, y)
    assert y.shape == (num_labels, n)

    # Auxiliary computations for J
    z1 = w1 @ x + b1
    assert z1.shape == (hidden_layer_size, n)
    a1, dg1 = relu(z1)
    assert a1.shape == (hidden_layer_size, n)
    z2 = w2 @ a1 + b2
    assert z2.shape == (num_labels, n)
    a2, _ = sigmoid(z2)
    assert a2.shape == (num_labels, n)

    # Compute J
    J = (-1 / n) * (np.sum(y * np.log(a2)) + np.sum((1 - y) * np.log(1 - a2)) \
        + (lambda_ / (2 * n)) * (np.sum(w1 * w1) + np.sum(w2 * w2))

    # Auxiliary computations for grad J
    delta2 = a2 - y
    delta1 = (delta2 * (w2 @ dg1)).T

    # Compute gradients
    dw2 = (1 / n) * np.sum(delta2 * a1.T)
    db2 = (1 / n) * np.sum(delta2)
    dw1 = (1 / n) * np.sum(delta1 @ x.T)
    db1 = (1 / n) * np.sum(delta1)

    d = {'cost' : J,
         'dw2' : dw2,
         'db2' : db2,
         'dw1' : dw1,
         'db1' : db1}

    return d

def main():
    x = np.random.random((4,3))
    sigma, d_sig = sigmoid(x)

    print(f'x={x}')
    print(f'sigma={sigma}')
    print(f'dsigma={d_sig}')


if __name__ == '__main__':
    main()
