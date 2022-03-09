import numpy as np

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
    B = int(np.ceil(n / b))
    batches = []
    for i in range(B):
        x_temp = x[:,(b * i):(b * (i + 1))]
        y_temp = y[:,(b * i):(b * (i + 1))]
        batches.append({'x' : x_temp, 'y' : y_temp})
    # Slicing automatically ends at the end of the list regardless of the stop
    return batches


def model(x, y,
        batch_size,
        hidden_layer_sizes,
        learning_rate, activators,
        num_iters=10000,
        print_cost=False):
    """
    Parameters
    ----------
    Returns
    -------
    """
    n, layers = utils.dim_retrieval(x, y, hidden_layer_sizes)
    params = utils.initialize_parameters(layers)
    batches = get_batches(x, y, batch_size)
    B = len(batches)

    ## main descent loop
    for i in range(num_iters):
        ## batch loop
        for batch in batches:
            x = batch['x']
            y = batch['y']
            cache = utils.forward_propagation(params, x, activators)
            cost = utils.compute_cost(cache, y)
            grads = utils.backward_propagation(params, cache, activators, x, y)
            params = utils.update_parameters(params, grads, 0.1)

def main():
    x = np.random.rand(7,10000)
    y = np.random.rand(1,10000)
    batch_size = 2 ** 6
    batches = get_batches(x, y, batch_size)
    for batch in batches:
        x = batch['x']
        y = batch['y']
        print(x.shape)
        print(y.shape)

if __name__ == '__main__':
    main()
