#! python3

import numpy as np

from mlLib.utils import LinearParameters, apply_activation

class ShuffleBatchData():
    def __init__(self, data, batch_size, seed=10101):
        """
        Parameters:
        -----------
        data : Dict[array_like]
            data['x'] : array_like
            data['y'] : array_like
        batch_size : int
        seed : int
            Default: 10101

        Returns:
        None
        """
        self.data = data
        self.batch_size = batch_size
        self.seed = seed
        self.idx = np.arange(data['x'].shape[1])
        self.__N = data['x'].shape[1]

        np.random.seed(seed)
    
    def get_batches(self):
        """
        Parameters:
        -----------
        None

        Returns:
        --------
        None
        """
        np.random.shuffle(self.idx)
        x_shuffled = self.data['x'][:, self.idx]
        y_shuffled = self.data['y'][:, self.idx]

        B = int(np.ceil(self.__N / self.batch_size))

        batches = []
        for i in range(B):
            x_aux = x_shuffled[:, (self.batch_size * i):(self.batch_size * (i + 1))]
            y_aux = y_shuffled[:, (self.batch_size * i):(self.batch_size * (i + 1))]
            batches.append({'x' : x_aux, 'y' : y_aux})

        return batches

class NeuralNetwork():
    def __init__(self, config):
        """
        Parameters:
        -----------
        config : Dict
            config['lp_reg'] = 0,1,2
            config['batch_size'] = 2 ** p # p in {5, 6, 7, 8, 9, 10}
            config['nodes'] = List[int]
            config['bias'] = List[Boolean]
            config['activators'] = List[str]
            config['keep_probs'] = List[float]

        Returns:
        --------
        None
        """
        self.config = config
        self.lp_reg = config['lp_reg']
        self.batch_size = config['batch_size']
        self.nodes = config['nodes']
        self.bias = config['bias']
        self.activators = config['activators']
        self.keep_probs = config['keep_probs']
        self.L = len(config['nodes']) - 1

    def init_dropout(self, num_examples, seed=101011):
        """
        Parameters:
        -----------
        num_examples : int
        seed : int
            Default: 1 # For reproducability

        Returns:
        --------
        D : Dict[layer : array_like]
        """
        np.random.seed(seed)
        D = {}
        for l in range(self.L + 1):
            D[l] = np.random.rand(self.nodes[l], num_examples)
            D[l] = (D[l] < self.keep_probs[l]).astype(int)
            D[l] = D[l] / self.keep_probs[l]
            assert (D[l].shape == (self.nodes[l], num_examples)), "Dropout matrices are the wrong shape"
        
        return D

    def forward_propagation(self, params, x, dropout=None):
        """
        Parameters:
        -----------
        params : Dict[class[Parameters]]
            params[l].w = Weights
            params[l].bias = Boolean
            params[l].b = Bias
        x : array_like

        Returns:
        --------
        cache = Dict[array_like]
            cache['a'] = a
            cache['dg'] = dg

        """
        # Initialize dictionaries
        a = {}
        dg = {}

        a[0], dg[0] = apply_activation(x, self.activators[0])
        if dropout != None:
            a[0] = dropout[0] * a[0]

        for l in range(1, self.L + 1):
            z = params[l].forward(a[l - 1])
            a[l], dg[l] = apply_activation(z, self.activators[l])
            if dropout != None:
                a[l] = dropout[l] * a[l]

        cache = {'a': a, 'dg': dg}
        return cache

    def cost_function(self, params, a, y, lambda_=0.01, eps=1e-8):
        """
        Parameters:
        -----------
        params: Dict[LinearParameters]
        a: array_like
        y: array_like
        lambda_: float
            Default: 0.01
        eps: float
            Default: 1e-8

        Returns:
        --------
        cost: float
        """
        n = y.shape[1]
        if self.lp_reg == 0:
            lambda_ = 0.0

        # Compute regularization term
        R = 0
        for param in params.values():
            R += np.sum(np.abs(param.w) ** self.lp_reg)
        R *= (lambda_ / (2 * n))

        # Compute unregularized cost
        a = np.clip(a, eps, 1 - eps)    # Bound a for stability
        J = (-1 / n) * (np.sum(y * np.log(a) + (1 - y) * np.log(1 - a)))

        cost = float(np.squeeze(J + R))

        return cost

    def backward_propagation(self, params, cache, y, dropout):
        """
        Parameters:
        -----------
        params : Dict[LinearParameters]
            params[l].w = Weights
            params[l].bias = Boolean
            params[l].b = Bias
        cache : Dict[array_like]
            cache['a'] : array_like
            cache['dg'] : array_like
        y : array_like
        
        Returns:
        --------
        None
        """

        # Retrieve cache
        a = cache['a']
        dg = cache['dg']

        # Initialize differentials along the network
        delta = {}
        delta[self.L] = ((a[self.L] - y) / y.shape[1]) * dropout[self.L]

        for l in reversed(range(1, self.L + 1)):
            delta[l - 1] = dg[l - 1] * params[l].backward(delta[l], a[l - 1]) * dropout[l - 1]

    def update_parameters(self, params, learning_rate=0.1):
        """
        Parameters:
        -----------
        params : Dict[LinearParameters]
            params[l].w = Weights
            params[l].bias = Boolean
            params[l].b = Bias
        learning_rate : float
            Default : 0.01

        Returns:
        --------
        None
        """
        for param in params.values():
            param.update(learning_rate)

    def fit(self, data, learning_rate=0.1, lambda_=0.01, num_iters=10000):
        """
        Parameters:
        -----------
        data : Dict[array_like]
            data['x'] : array_like
            data['y'] : array_like
        learning_rate : float
            Default : 0.1
        lambda_ : float
            Default : 0.0
        num_iters : int
            Default : 10000

        Returns:
        --------
        costs : List[floats]
        params : class[LinearParameters]
        """
        # Initialize parameters per layer
        params = {}
        for l in range(1, self.L + 1):
            params[l] = LinearParameters(
                (self.nodes[l], self.nodes[l - 1]), self.bias[l])

        # Initialize batching
        batching = ShuffleBatchData(data, self.batch_size)

        costs = []
        for i in range(num_iters):
            batches = batching.get_batches()
            for batch in batches:
                x = batch['x']
                y = batch['y']
                dropout = self.init_dropout(x.shape[1])
                cache = self.forward_propagation(params, x, dropout)
                cost = self.cost_function(params, cache['a'][self.L], y, lambda_)
                costs.append(cost)
                self.backward_propagation(params, cache, y, dropout)
                self.update_parameters(params, learning_rate)

            if i % 100 == 0:
                print(f'Cost after iteration {i}: {cost}')

        return params

    def evaluate(self, params, x):
        """
        Parameters:
        -----------
        params : Dict[LinearParameters]
        x : array_like

        Returns:
        --------
        y_hat : array_like
        """
        cache = self.forward_propagation(params, x)
        a = cache['a'][self.L]
        y_hat = (~(a < 0.5)).astype(int)
        return y_hat

    def accuracy(self, params, data):
        """
        Parameters:
        -----------
        params : Dict[LinearParameters]
        data : Dict[array_like]
            data['x'] : array_like
            data['y'] : array_like

        Returns:
        --------
        accuracy : float
        """
        x = data['x']
        y = data['y']

        y_hat = self.evaluate(params, x)
        acc = np.sum(y_hat == y) / y.shape[1]

        return acc

if __name__ == '__main__':
    from pathlib import Path

    import pandas as pd

    from mlLib.utils import ProcessData

    csv = Path('neuralNetworks/src/python/data/housepricedata.csv')
    df = pd.read_csv(csv)
    dataset = df.values
    x = dataset[:, :10]
    y = dataset[:, 10].reshape(-1, 1)
    data = ProcessData(x, y, 0.15, 0.15, seed=1, feat_as_col=False)
    

    config = {
        'lp_reg': 0,
        'batch_size' : 2 ** 5,
        'nodes': [10, 32, 8, 1],
        'bias': [False, True, True, True],
        'activators': ['linear', 'relu', 'relu','sigmoid'],
        'keep_probs' : [1, 0.85, 0.85, 1]
    }

    model = NeuralNetwork(config)
    params = model.fit(data.train, 0.01, 0.1, num_iters=2000)

    train_acc = model.accuracy(params, data.train)
    print(f'Training Accuracy: {train_acc}')
    dev_acc = model.accuracy(params, data.dev)
    print(f'Dev Accuracy: {dev_acc}')
    test_acc = model.accuracy(params, data.test)
    print(f'Test Accuracy: {test_acc}')
