import copy

import numpy as np
from sklearn.utils import shuffle

import utils

def get_batches(x, y, b):
    """
    Parameters
    ----------
    x : array_like
        x.shape = (m, n)
    y : array_like
        y.shape = (k, n)
    b : int

    Returns
    -------
    batches : List[Dict]
        batches[i]['x'] : array_like
            x.shape = (m, b) # except last batch
            y.shape = (k, b) # except last batch

    """
    m, n = x.shape
    ## Shuffle the data
    x, y = shuffle(x.T, y.T) # Only shuffles rows, so transpose is needed
    x = x.T
    y = y.T

    B = int(np.ceil(n / b))
    batches = []
    for i in range(B):
        x_temp = x[:,(b * i):(b * (i + 1))]
        y_temp = y[:,(b * i):(b * (i + 1))]
        batches.append({'x' : x_temp, 'y' : y_temp})
    # Slicing automatically ends at the end of
    # the list if the stop is outside the index
    return batches

def initialize_momenta(layers):
    """
    Parameters
    ----------
    layers : List[int]
        layers[l] = # nodes in layer l
    Returns
    -------
    v : Dict[Dict[array_like]]
    s : Dict[Dict[array_like]]
    """
    vw = {}
    vb = {}
    sw = {}
    sb = {}
    for l in range(1, len(layers)):
        vw[l] = np.zeros((layers[l], layers[l - 1]))
        sw[l] = np.zeros((layers[l], layers[l - 1]))
        vb[l] = np.zeros((layers[l], 1))
        sb[l] = np.zeros((layers[l], 1))

    v = {'w' : vw, 'b' : vb}
    s = {'w' : sw, 'b' : sb}

    return v, s

def learning_rate_decay(epoch, learning_rate, decay_rate=0.95):
    """
    Parameters
    ----------
    learning_rate : float
    decay_rate : float
        Default: 0.95

    Returns
    -------
    alpha : float
    """
    alpha = (1 / (1 + epoch * decay_rate)) * learning_rate
    return alpha

def corrected_momentum(v, grads, update_iter, beta1=0.0):
    """
    Parameters
    ----------
    v : Dict[Dict[array_like]]
        v['w'][l].shape = w[l].shape
        v['b'][l].shape = b[l].shape
    grads : Dict[Dict]
        grads['w'][l] : array_like
            dw[l].shape = w[l].shape
        grads['b'][l] : array_like
            db[l].shape = b[l].shape
    update_iter : int
    beta1 : float
        Default: 0.0 - Returns grads
        Usual: 0.9

    Returns
    -------
    velocities : Dict[Dict[array_like]]
        velocities['w'][l].shape = dw[l].shape
        velocities['b'][l].shape = db[l].shape
    """
    ## Retrieve velocities and gradients
    vw = v['w']
    vb = v['b']
    dw = grads['w']
    db = grads['b']
    L = len(dw)

    for l in range(1, L + 1):
        vw[l] = beta1 * vw[l] + (1 - beta1) * dw[l]
        vw[l] /= (1 - beta1 ** update_iter)
        assert(vw[l].shape == dw[l].shape)
        vb[l] = beta1 * vb[l] + (1 - beta1) * db[l]
        vb[l] /= (1 - beta1 ** update_iter)
        assert(vb[l].shape == db[l].shape)

    velocities = {'w' : vw, 'b' : vb}
    return velocities

def corrected_rmsprop(s, grads, update_iter, beta2=0.999):
    """
    Parameters
    ----------
    s : Dict[Dict[array_like]]
        s['w'][l].shape = w[l].shape
        s['b'][l].shape = b[l].shape
    grads : Dict[Dict]
        grads['w'][l] : array_like
            dw[l].shape = w[l].shape
        grads['b'][l] : array_like
            db[l].shape = b[l].shape
    update_iter : int
    beta2 : float
        Default: 0.999

    Returns
    -------
    accelerations : Dict[Dict[array_like]]
        accelerations['w'][l].shape = w[l].shape
        accelerations['b'][l].shape = b[l].shape
    """
    ## Retrieve accelerations and gradients
    sw = s['w']
    sb = s['b']
    dw = grads['w']
    db = grads['b']
    L = len(dw)

    for l in range(1, L + 1):
        sw[l] = beta2 * sw[l] + (1 - beta2) * (dw[l] * dw[l])
        sw[l] /= (1 - beta2 ** update_iter)
        assert(sw[l].shape == dw[l].shape)
        sb[l] = beta2 * sb[l] + (1 - beta2) * (db[l] * db[l])
        sb[l] /= (1 - beta2 ** update_iter)
        assert(sb[l].shape == db[l].shape)

    accelerations = {'w' : sw, 'b' : sb}
    return accelerations


def update_parameters_adam(params, grads, epoch, batch_iter, velocities, accelerations, momenta=[1e-8, 0.9, 0.999], learning_rate=0.01, decay_rate = 0.95):
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
    epoch : int
    batch_iter : int
    learning_rate : float
        Default: 0.01
    momenta : List[float]
        momenta[0] = epsilon
            Default: 10^{-8}
        momenta[1] = beta_1
            Default: 0.9
        momenta[2] = beta_2
            Default: 0.999

    Returns
    -------
    params : Dict[Dict]
        params['w'][l] : array_like
            w[l].shape = (layers[l], layers[l-1])
        params['b'][l] : array_like
            b[l].shape = (layers[l], 1)
    """
    update_iter = epoch + batch_iter
    ## Retrieve parameters
    w = copy.deepcopy(params['w'])
    b = copy.deepcopy(params['b'])
    L = len(w)

    ## Update velocites and accelerations
    v = corrected_momentum(velocities, grads, update_iter, momenta[1])
    vw = v['w']
    vb = v['b']
    s = corrected_rmsprop(accelerations, grads, update_iter, momenta[2])
    sw = s['w']
    sb = s['b']

    ## Update learning rate
    alpha = learning_rate_decay(epoch, learning_rate, decay_rate)

    ## Perform update
    for l in range(1, L + 1):
        w[l] = w[l] - alpha * vw[l] / (np.sqrt(sw[l]) + momenta[0])
        b[l] = b[l] - alpha * vb[l] / (np.sqrt(sb[l]) + momenta[0])

    params = {'w' : w, 'b' : b}
    return params

def model(x, y,
        hidden_layer_sizes,
        activators,
        batch_size,
        lambda_=0.0,
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
    batch_size : int
    lambda_ : float
        The regularization parameter
        Default: 0.0
    num_iters : int
        Number of iterations with which our model performs gradient descent
        Default: 10000
    print_cost : Boolean
        If True, print the cost every 1000 iterations
        Default: False

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
    n, layers = utils.dim_retrieval(x, y, hidden_layer_sizes)
    params = utils.initialize_parameters_random(layers)
    v, s = initialize_momenta(layers)


    ## main descent loop
    for i in range(num_iters):
        batches = get_batches(x, y, batch_size)
        ## batch loop
        batch_iter = 1
        cost = 0
        for batch in batches:
            x = batch['x']
            y = batch['y']
            cache = utils.forward_propagation(x, params, activators)
            cost += utils.compute_cost(y, params, cache)
            grads = utils.backward_propagation(x, y, params, cache, activators)
            params =  update_parameters_adam(params,
                                       grads,
                                       i,
                                       batch_iter,
                                       v,
                                       s,
                                       momenta=[1e-8, 0.9, 0.999],
                                       learning_rate=0.01,
                                       decay_rate = 0.0)
            batch_iter += 1

        if print_cost and i % 1000 == 0:
            print(f'Cost after iteration {i}: {cost}')

    return params, cost











def main():
    x = np.random.rand(7,10000)
    y = np.random.rand(1,10000)
    hidden_layer_sizes = [4, 3, 3]
    activators = ['relu'] * 3 + ['sigmoid']
    batch_size = 2 ** 6

    params, cost = model(x, y, hidden_layer_sizes, activators, batch_size, print_cost=True)
    utils.print_array_dict(params['w'])
    utils.print_array_dict(params['b'])

if __name__ == '__main__':
    main()
