import copy

import numpy as np

def sigmoid(z):
    """
    Parameters
    ----------
    z : array_like

    Returns
    -------
    sigma : array_like
        The value of the sigmoid function evaluated at z
    """
    sigma = (1 / (1 + np.exp(-z)))
    return sigma

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
    """
    r = np.maximum(z, beta * z)
    return r

def reshape_params(params, input_layer_size, hidden_layer_size, num_labels=2):
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
        Default: 2 - Represents binary classification
        The number of classifcation labels for our target output

    Returns
    -------
    d : Dict
        d['w1'] : array_like
            d['w1'].shape = (hidden_layer_size, input_layer_size)
        d['w2'] : array_like
            d['w2'].shape = (num_labels, hidden_layer_size)
        d['b1'] : array_like
            d['b1'].shape = (hidden_layer_size, 1)
        d['b2'] : array_like
            d['b2'].shape = (num_labels, 1)
    """
    pass

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
    J : float
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
    a1 = relu(z1)
    assert a1.shape == (hidden_layer_size, n)
    z2 = w2 @ a1 + b2
    assert z2.shape == (num_labels, n)
    a2 = sigmoid(z2)
    assert a2.shape == (num_labels, n)

    # Compute J
    #J = (-1 / n) * (np.sum(y * np.log(a2)) + np.sum((1 - y) * np.log(1 - a2)) \
    #    + (lambda_ / (2 * n)) * (np.sum(w1 * w1) + np.sum(w2 * w2))

    return 2









def main():
    x = np.random.random((4,3))
    sigma, d_sig = sigmoid(x)

    print(f'x={x}')
    print(f'sigma={sigma}')
    print(f'dsigma={d_sig}')


if __name__ == '__main__':
    main()
