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

    ## main descent loop
    for i in range(num_iters):
        batches = get_batches(x, y, batch_size)
        ## batch loop
        for batch in batches:
            x = batch['x']
            y = batch['y']
            cache = utils.forward_propagation(x, params, activators)
            cost = utils.compute_cost(y, params, cache)
            grads = utils.backward_propagation(x, y, params, cache, activators)
            params = utils.update_parameters(params, grads, learning_rate=0.005)

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
