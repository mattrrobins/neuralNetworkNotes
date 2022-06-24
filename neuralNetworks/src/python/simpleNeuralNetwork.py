#! python3

from pathlib import Path

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow import keras
from keras import Model, Input
from keras.layers import Dense

import mlLib.npActivators as npActivators
from mlLib.utils import ProcessData

def apply_activation(z, activator):
    """
    Parameters:
    -----------
    z : array_like
    activator : str

    Returns:
    --------
    a : array_like
    dg : array_like
    """
    if activator == 'relu':
        a, dg = npActivators.relu(z)
    elif activator == 'sigmoid':
        a, dg = npActivators.sigmoid(z)
    elif activator == 'tanh':
        a, dg = npActivators.tanh(z)
    elif activator == 'linear':
        a, dg = npActivators.linear(z)

    assert (a.shape == z.shape)
    assert (dg.shape == z.shape)
    return a, dg

class Parameters():
    def __init__(self, dims, bias=True, seed=1):
        """
        Parameters:
        -----------
        dims : tuple(int, int)
        bias : Boolean
            Default : True
        seed : int
            Default : 1
        
        Returns:
        --------
        None
        """
        np.random.seed(seed)
        self.dims = dims
        self.bias = bias
        self.w = np.random.randn(*dims) * 0.01
        if bias:
            self.b = np.zeros((dims[0], 1))

    def forward(self, a):
        """
        Parameters:
        -----------
        a : array_like

        Returns:
        --------
        z : array_like
        """
        z = np.einsum('ij,jk', self.w, a)
        if self.bias:
            z += self.b
        
        return z

    def backward(self, din, a_prev):
        """
        Parameters:
        -----------
        din : array_like
        a_prev : array_like

        Returns:
        --------
        dout : array_like
        """
        if self.bias:
            self.db = np.sum(din, axis=1, keepdims=True)
            assert (self.db.shape == self.b.shape)
        
        self.dw = np.einsum('ij,kj', din, a_prev)
        assert (self.dw.shape == self.w.shape)

        dout = np.einsum('ij,ik', self.w, din)
        assert (dout.shape == a_prev.shape)

        return dout

    def update(self, learning_rate=0.01):
        """
        Parameters:
        -----------
        learning_rate : float
            Default : 0.01
        
        Returns:
        --------
        None
        """
        w = self.w - learning_rate * self.dw
        self.w = w

        if self.bias:
            b = self.b - learning_rate * self.db
            self.b = b

class NeuralNetwork():
    def __init__(self, config):
        """
        Parameters:
        -----------
        config : Dict
            config['lp_reg'] = 0,1,2
            config['nodes'] = List[int]
            config['bias'] = List[Boolean]
            config['activators'] = List[str]

        Returns:
        --------
        None
        """
        self.config = config
        self.lp_reg = config['lp_reg']
        self.nodes = config['nodes']
        self.bias = config['bias']
        self.activators = config['activators']
        self.L = len(config['nodes']) - 1

    def compute_cost(self, y, a, params, lambda_=0.0, eps=1e-8):
        """
        Parameters:
        -----------
        y: array_like
        a: array_like
        params: class[Parameters]
        lambda_: float
            Default: 0.0
        eps: float
            Default: 1e-8

        Returns:
        --------
        cost: float
        """
        n = y.shape[1]

        # Compute regularization term
        R = 0
        for param in params.values():
            R += np.sum(np.abs(param.w) ** self.lp_reg)
        R *= (lambda_ / (2 * n))

        # Compute unregularized cost
        J = (-1 / n) * (np.sum(y * np.log(a + eps)) + np.sum((1 - y) * np.log(1 - a + eps)))

        cost = float(np.squeeze(J + R))

        return cost

    def forward_propagation(self, x, params):
        """
        Parameters:
        -----------
        x : array_like
        params : Dict[class[Parameters]]
            params[l].w = Weights
            params[l].bias = Boolean
            params[l].b = Bias

        Returns:
        --------
        cache = Dict[arrau_like]
            cache['a'] = a
            cache['dg'] = dg

        """
        # Initialize dictionaries
        a = {}
        dg = {}
        
        a[0], dg[0] = apply_activation(x, self.activators[0])

        for l in range(1, self.L + 1):
            z = params[l].forward(a[l - 1])
            a[l], dg[l] = apply_activation(z, self.activators[l])
        
        cache = {'a' : a, 'dg' : dg}
        return cache

    def backward_propagation(self, y, params, cache):
        """
        Parameters:
        -----------
        y : array_like
        params : Dict[class[Parameters]]
            params[l].w = Weights
            params[l].bias = Boolean
            params[l].b = Bias
        cache : Dict[array_like]
            cache['a'] : array_like
            cache['dg'] : array_like
        
        Returns:
        --------
        None
        """

        # Retrieve cache
        a = cache['a']
        dg = cache['dg']

        # Initialize differentials along the network
        delta = {}
        delta[self.L] = (a[self.L] - y) / y.shape[1]

        for l in reversed(range(1, self.L + 1)):
            delta[l - 1] = dg[l- 1] * params[l].backward(delta[l], a[l - 1])

    def update_parameters(self, params, learning_rate=0.01):
        """
        Parameters:
        -----------
        params : Dict[class[Parameters]]
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

    def fit(self, x, y, learning_rate=0.01, lambda_=0.0, num_iters=10000):
        """
        Parameters:
        -----------
        x : array_like
        y : array_like
        learning_rate : float
            Default : 0.1
        lambda_ : float
            Default : 0.0
        num_iters : int
            Default : 10000

        Returns:
        --------
        costs : List[floats]
        params : class[Parameters]
        """
        # Initialize parameters per layer
        params = {}
        for l in range(1, self.L + 1):
            params[l] = Parameters((self.nodes[l], self.nodes[l - 1]), self.bias[l])


        costs = []
        for i in range(num_iters):
            cache = self.forward_propagation(x, params)
            cost = self.compute_cost(y, cache['a'][self.L], params, lambda_)
            costs.append(cost)
            self.backward_propagation(y, params, cache)
            self.update_parameters(params, learning_rate)

            if i % 1000 == 0:
                print(f'Cost after iteration {i}: {cost}')

        return costs, params

    def evaluate(self, x, params):
        """
        Parameters:
        -----------
        x : array_like
        params : class[Parameters]

        Returns:
        --------
        predictions : array_like
        """
        cache = self.forward_propagation(x, params)
        a = cache['a'][self.L]
        predictions = (~(a <0.5)).astype(int)
        return predictions

    def accuracy(self, x, y, params):
        """
        Parameters:
        -----------
        x : array_like
        y : array_like
        params : class[Parameters]

        Returns:
        --------
        accuracy : float
        """
        predictions = self.evaluate(x, params)
        aux = np.abs(predictions - y)
        accuracy = 1 - np.sum(aux) / y.shape[1]

        return accuracy


def tf_main(csv):
    df = pd.read_csv(csv)
    dataset = df.values
    x, y = dataset[:, :-1], dataset[:, -1]
    y = y.reshape(y.size, 1)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
    train = {'x' : x_train, 'y' : y_train}
    test = {'x' : x_test, 'y' : y_test}
    mu = np.mean(train['x'], axis=0, keepdims=True)
    var = np.var(train['x'], axis=0, keepdims=True)
    eps = 1e-8
    train['x'] = (train['x'] - mu) / np.sqrt(var + eps)
    test['x'] = (test['x'] - mu) / np.sqrt(var + eps)

    ## Define network layout
    input_layer = Input(shape=(10,))
    hidden_layer = Dense(
        32, 
        activation='relu', 
        kernel_initializer='he_normal',
        bias_initializer='zeros'
    )(input_layer)
    output_layer = Dense(
        1, 
        activation='sigmoid',
        kernel_initializer='he_normal',
        bias_initializer='zeros'
    )(hidden_layer)
    model = Model(inputs=input_layer, outputs=output_layer)
    model.summary()

    ## Compile desired model
    model.compile(
        loss = 'binary_crossentropy',
        optimizer = 'adam',
        metrics = ['accuracy']
    )

    ## Train the model
    hist = model.fit(
        train['x'],
        train['y'],
        batch_size=32,
        epochs=150,
        validation_split=0.2
    )

    ## Evaluate the model
    test_scores = model.evaluate(test['x'], test['y'], verbose=2)
    print(f'Test Loss: {test_scores[0]}')
    print(f'Test Accuracy: {test_scores[1]}')

def main():
    csv = Path('neuralNetworks/src/python/data/housepricedata.csv')
    df = pd.read_csv(csv)
    dataset = df.values
    x = dataset[:, 0:10]
    y = dataset[:, 10].reshape(-1, 1)
    data = ProcessData(x.T, y.T, 0.15, 0.15)

    config = {
        'lp_reg' : 2,
        'nodes' : [10, 32, 1],
        'bias' : [False, True, True],
        'activators' : ['linear', 'relu', 'sigmoid']
    }

    model = NeuralNetwork(config)
    costs, params = model.fit(data.train['x'], data.train['y'], 0.01, 0.1)
    dev_acc = model.accuracy(data.dev['x'], data.dev['y'], params)
    print(f'The dev accuracy: {dev_acc}')
    test_acc = model.accuracy(data.test['x'], data.test['y'], params)
    print(f'The test accuracy: {test_acc}')

    tf_main(csv)
    
if __name__ == '__main__':
    main()


