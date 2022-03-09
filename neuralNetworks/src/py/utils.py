#! python3

import numpy as np
from sklearn.utils import shuffle

import activators
from activators import ACTIVATORS

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


##### General Neural Network Model #####

## Retrieve number of examples and layer dimensions
def dim_retrieval(x, y, hidden_sizes):
    """
    Parameters
    ----------
    x : array_like
        x.shape = (layers[0], n)
    y : array_like
        y.shape = (layers[L], n)
    hidden_sizes : List[int]
        The number nodes layer i = hidden_sizes[i-1]
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

def dropout_matrices(layers, num_examples, keep_prob):
    """
    Parameters
    ----------
    layers : List[int]
        layers[l] = number of nodes in layer l
    num_examples : int
        The number of training examples
    keep_prob : List[float]
        keep_prob[l] = The probabilty of keeping a node in layer l

    Returns
    -------
    D : Dict[array_like]
        D[l].shape = (layers[l], num_ex)
        D[l] = a Boolean array
    """
    np.random.seed(1)
    L = len(layers)
    D = {}
    for l in range(L - 1):
        D[l] = np.random.rand(layers[l], num_examples)
        D[l] = (D[l] < keep_prob[l]).astype(int)
        assert(D[l].shape == (layers[l], num_examples))
    return D

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
        activator = 'relu', 'sigmoid', or 'tanh'

    Returns
    -------
    z : array_like
        z.shape = (layer_dims[l+1], n)
    a : array_like
        a.shape = (layer_dims[l+1], n)
    """
    assert activator in ACTIVATORS, f'{activator} is not a valid activator.'

    z = w @ a_prev + b
    if activator == 'relu':
        a, _ = activators.relu(z)
    elif activator == 'sigmoid':
        a, _ = activators.sigmoid(z)
    elif activator == 'tanh':
        a, _ = activators.tanh(z)
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
        activator = 'relu', 'sigmoid', or 'tanh'

    Returns
    -------
    delta : array_like
        delta.shape = (layers[l])
    """
    assert activator in ACTIVATORS, f'{activator} is not a valid activator.'

    n = delta_next.shape[1]

    if activator == 'relu':
        _, dg = activators.relu(z)
    elif activator == 'sigmoid':
        _, dg = activators.sigmoid(z)
    elif activator == 'tanh':
        _, dg = activators.tanh(z)

    da = w.T @ delta_next
    assert(da.shape == (w.shape[0], n))
    delta = da * dg
    assert(delta.shape == (w.shape[0], n))
    return delta


def forward_propagation(x, params, activators, D, keep_prob):
    """
    Parameters
    ----------
    x : array_like
        x.shape = (layers[0] n)
    params : Dict[Dict]
        params['w'][l] : array_like
            wl.shape = (layers[l], layers[l-1])
        params['b'][l] : array_like
            bl.shape = (layers[l], 1)
    activators : List[str]
        activators[l] = activation function of layer l+1
    D : Dict[array_like]
        D[l].shape = (layer_dims[l], num_ex)
        D[l] = a Boolean array astype(int)
    keep_prob : List[float]
        keep_prob[l] = The probabilty of keeping a node in layer l

    Returns
    -------
    cache : Dict[Dict]
        cache['z'][l] : array_like
            z[l].shape = (layers[l], n)
        cache['a'][l] : array_like
            a[l].shape = (layers[l], n)
    """
    # Retrieve parameters
    w = params['w']
    b = params['b']
    L = len(w) # Number of layers excluding output layer
    n = x.shape[1]
    # Set empty caches
    a = {}
    z = {}
    # Dropout on layer 0
    a[0] = x
    a[0] = a[0] * D[0]
    a[0] /= keep_prob[0]
    # Loop through hidden layers
    for l in range(1, L):
        zl, al = linear_activation_forward(a[l - 1], w[l], b[l], activators[l - 1])
        al = al * D[l]
        al /= keep_prob[l]
        z[l] = zl
        a[l] = al
    # Output layer
    z[L], a[L] = linear_activation_forward(a[L - 1], w[L], b[L], activators[-1])

    cache = {'z' : z, 'a' : a}
    return cache

def backward_propagation(x, y, params, cache, activators, D, keep_prob):
    """
    Parameters
    ----------
    x : array_like
        x.shape = (layers[0], n)
    y : array_like
        y.shape = (layers[-1], n)
    params : Dict[Dict[array_like]]
        params['w'][l] : array_like
            w[l].shape = (layers[l], layers[l-1])
        params['b'][l] : array_like
            b[l].shape = (layers[l], 1)
    cache : Dict[Dict[array_like]]
        cache['a'][l] : array_like
            a[l].shape = (layers[l], n)
        cache['z'][l] : array_like
            z[l].shape = (layers[l], n)
    activators : List[str]
        activators[l] = activation function of layer l+1
    D : Dict[array_like]
        D[l].shape = (layer_dims[l], num_ex)
        D[l] = a Boolean array astype(int)
    keep_prob : List[float]
        keep_prob[l] = The probabilty of keeping a node in layer l

    Returns
    -------
    grads : Dict[Dict]
        grads['dw'][l] : array_like
            dw[l].shape = w[l].shape
        grads['db'][l] : array_like
            db[l].shape = b[l].shape
    """
    ## Retrieve parameters
    a = cache['a']
    z = cache['z']
    w = params['w']
    n = x.shape[1]
    L = len(z)

    ## Compute deltas
    delta = {}
    delta[L] = a[L] - y
    for l in reversed(range(1, L)):
        delta = linear_activation_backward(delta[l + 1], z[l], w[l], activators[l])
        delta = delta * D[l]
        delta /= keep_prob[l]

    ## Compute gradients
    dw = {}
    db = {}

    for l in range(1, L + 1):
        db[l] = (1 / n) * np.sum(delta[l], axis=1, keepdims=True)
        assert(db[l].shape == (w[l].shape[0], 1))
        dw[l] = (1 / n) * delta[l] * a[l - 1].T
        assert(dw[l].shape == w[l].shape)
    grads = {'dw' : dw, 'db' : db}
    return grads

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

def model_nn(x, y,
          hidden_layer_sizes,
          activators,
          keep_prob=1.0,
          num_iters=10000,
          print_cost=False):
    """
    Parameters
    ----------
    x : array_like
        x.shape = (layers[0], n)
    y : array_like
        y.shape = (layers[-1], n)
    hidden_layer_sizes : List[int]
        The number nodes layer l = hidden_layer_sizes[l-1]
    activators : List[str]
        activators[l] = activation function of layer l+1
    keep_prob : List[float] | float
        keep_prob[l] = The probabilty of keeping a node in layer l
        keep_prob = The same probability for all input and hidden layers
    num_iters : int
        Number of iterations with which our model performs gradient descent
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
    ## Set dimensions and Initialize parameters
    n, layers = dim_retrieval(x, y, hidden_layer_sizes)
    params = initialize_parameters_random(layers)

    ## Expand keep_prob to a list if it's a single float
    if isinstance(keep_prob, float):
        keep_prob = [keep_prob] * (len(layers) - 1)

    # main gradient descent loop
    for i in range(num_iters):
        D = dropout_matrices(layers, n, keep_prob)
        cache = forward_propagation(x, params, activators, D, keep_prob)
        cost = compute_cost(cache, y)
        grads = backward_propagation(x, y, params, cache, activators, D, keep_prob)
        params = update_parameters(params, grads)

        if print_cost and i % 1000 == 0:
            print(f'Cost after iteration {i}: {cost}')

    return params, cost







########## TESTING ##########
def test_dropout_nn():
    x = np.random.rand(4, 500)
    y = np.random.rand(1, 500)
    hidden_layer_sizes = [4, 5, 4]
    activators = ['relu', 'relu', 'relu', 'sigmoid']
    keep_prob = 1.0
    params, cost = model_nn(x, y, hidden_layer_sizes, activators, keep_prob)
    print(params)





#######
if __name__ == '__main__':
    test_dropout_nn()
