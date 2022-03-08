#! python3

import numpy as np
from sklearn.utils import shuffle

## Usefule printing function
def print_array_dict(D):
    """
    Parameters
    ----------
    D : Dict[array_like]
    Returns
    -------
    None
    """
    txt = "Array {0} has shape {1}\n{2}"
    for k, v in D.items():
        print(txt.format(str(k), v.shape, v))


## Partition data into training, development, and test sets
def partition_data(x, y, train_ratio):
    """
    Parameters
    ----------
    x : array_like
        x.shape = (m, N)
    y : array_like
        y.shape = (k, N)
    train_ratio : float
        0<=train_ratio<=1

    Returns
    -------
    train : Tuple[array_like]
    dev : Tuple[array_like]
    test : Tuple[array_like]
    """
    ## Shuffle the data
    x, y = shuffle(x.T, y.T) #
    x = x.T
    y = y.T

    ## Get the size of partitions
    N = x.shape[1]
    N_train = int(train_ratio * N)
    N_mid = (N - N_train) // 2

    ## Create partitions
    train = (x[:,:N_train], y[:,:N_train])
    dev = (x[:,N_train:N_train+N_mid], y[:,N_train:N_train+N_mid])
    test = (x[:,N_train+N_mid:], y[:,N_train+N_mid:])

    assert(x.all() == np.concatenate([train[0], dev[0], test[0]], axis=1).all())
    assert(y.all() == np.concatenate([train[1], dev[1], test[1]], axis=1).all())

    return train, dev, test


## Initialize parameters using the size of each layer
def initialize_parameters(layers):
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

## Forward and Backward Linear Activations
def linear_activation_forward(a_prev, w, b, activator):
    """
    Parameters
    ----------
    a_prev : array_like
        a_prev.shape = (layers[l], n)
    w : array_like
        w.shape = (layers[l+1], layers[l])
    b : array_like
        b.shape = (layers[l+1], 1)
    activator : str
        activator = 'relu', 'sigmoid', 'tanh', 'softmax'

    Returns
    -------
    z : array_like
        z.shape = (layer_dims[l+1], n)
    a : array_like
        a.shape = (layer_dims[l+1], n)
    """
    z = w @ a_prev + b
    if activator = 'relu':
        a, _ = utils.relu(z)
    elif activator = 'sigmoid':
        a, _ = utils.sigmoid(z)
    else:
        print("Activation function doesn't match ReLu or sigmoid.")
    return z, a

def linear_activation_backward(delta_next, z, w, activator):
    """
    Parameters
    ----------
    delta_next : array_like
        delta_next.shape = (layers[l+1], n)
    z : array_like
        z.shape = (layers[l+1], n)
    w : array_like
        w.shape = (layers[l+1], layers[l])
    activator : str
        activator = 'relu', 'sigmoid', 'tanh', 'softmax'

    Returns
    -------
    delta : array_like
        delta.shape = (layers[l])
    """
    n = delta_next.shape[1]

    if activator = 'relu':
        _, dg = relu(z)
    elif activator = 'sigmoid':
        _, dg = sigmoid(z)
    else:
        print("Activation function doesn't match ReLu or sigmoid.")

    da = w.T @ delta_next
    assert(da.shape == (w.shape[0], n))
    delta = da * dg
    assert(delta.shape == (w.shape[0], n))
    return delta


## Compute the cost
def compute_cost(cache, y):
    """
    Parameters
    ----------
    cache : Dict[Dict]
        cache['z'][l] : array_like
            z[l].shape = (layers[l], n)
        cache['a'][l] : array_like
            a[l].shape = (layers[l], n)
    y : array_like
        y.shape = (layers[-1], n)
    -------
    cost : float
        The cost evaluated at y and aL
    """
    ## Retrieve parameters
    n = y.shape[1]
    a = cache['a']
    L = len(a)
    aL = a[L - 1]

    cost = (-1 / n) * (np.sum(y * np.log(aL)) + np.sum((1 - y) * np.log(1 - aL)))
    cost = float(np.squeeze(cost))

    return cost


## Update parameters via gradient descent
def update_parameters(params, grads, learning_rate=0.01):
    """
    Parameters
    ----------
    params : Dict[Dict]
        params['w'][l] : array_like
            w[l].shape = (layers[l], layers[l-1])
        params['b'][l] : array_like
            b[l].shape = (layers[l], 1)
    grads : Dict[Dict]
        grads['dw'][l] : array_like
            dw[l].shape = w[l].shape
        grads['db'][l] : array_like
            db[l].shape = b[l].shape
    learning_rate : float
        Default: 0.01
        The learning rate for gradient descent

    Returns
    -------
    params : Dict[Dict]
        params['w'][l] : array_like
            w[l].shape = (layers[l], layers[l-1])
        params['b'][l] : array_like
            b[l].shape = (layers[l], 1)
    """
    ## Retrieve parameters
    w = copy.deepcopy(params['w'])
    b = copy.deepcopy(params['b'])
    L = len(w)

    ## Retrieve gradients
    dw = grads['dw']
    db = grads['db']

    ## Perform update
    for l in range(1, L + 1):
        w[l] = w[l] - learning_rate * dw[l]
        b[l] = b[l] - learning_rate * db[l]

    params = {'w' : w, 'b' : b}
    return params


## Activator functions
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
    dr = ((~(z < 0)) * 1) + ((z < 0) * beta)
    return r, dr

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
