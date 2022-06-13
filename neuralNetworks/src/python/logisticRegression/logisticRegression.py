#! python3

from pathlib import Path

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

def sigmoid(z):
    """
    Parameters
    ----------
    z : array_like

    Returns
    -------
    sigma : array_like
        The (broadcasted) value of the sigmoid function evaluated at z
    dsigma : array_like
        The (broadcasted) derivative of the sigmoid function evaluate at z
    """
    # Compute value of sigmoid
    sigma = (1 / (1 + np.exp(-z)))
    # Compute differential of sigmoid
    dsigma = sigma * (1 - sigma)
    return sigma, dsigma

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

    def forward(self, x):
        """
        Parameters:
        -----------
        x : array_like

        Returns:
        --------
        z : array_like
        """
        z = np.einsum('ij,jk', self.w, x)
        if self.bias:
            z += self.b
        
        return z

    def backward(self, dz, x):
        """
        Parameters:
        -----------
        dz : array_like
        x : array_like

        Returns:
        --------
        None
        """
        if self.bias:
            self.db = np.sum(dz, axis=1, keepdims=True)
            assert (self.db.shape == self.b.shape)
        
        self.dw = np.einsum('ij,kj', dz, x)
        assert (self.dw.shape == self.w.shape)

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

class Model():
    def __init__(self, lp_reg):
        """
        Parameters:
        lp_reg : int
            2 : L_2 Regularization is imposed
            1 : L_1 Regularization is imposed

        Returns:
        --------
        None
        """
        self.lp = lp_reg

    def cost_function(self, y, params, a, lambda_=0.0, eps=1e-8):
        """
        Parameters:
        -----------
        y : array_like
        params : class[Parameters]
        a : array_like
        lambda_ : float
            Default : 0.0
        eps : float
            Default : 1e-8

        Returns:
        --------
        cost : float
        """
        n = y.shape[1]

        R = np.sum(np.abs(params.w) ** self.lp)
        R *= (lambda_ / (2 * n))

        J = (-1 / n) * (np.sum(y * np.log(a + eps)) + np.sum((1 - y) * np.log(1 - a + eps)))

        cost = float(np.squeeze(J + R))

        return cost

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
        dims = (y.shape[0], x.shape[0])
        n = x.shape[1]
        params = Parameters(dims, True)

        costs = []
        for i in range(num_iters):
            z = params.forward(x)
            a, dg = sigmoid(z)
            cost = self.cost_function(y, params, a, lambda_)
            costs.append(cost)
            dz = (a - y) / n
            params.backward(dz, x)
            params.update(learning_rate)

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
        z = params.forward(x)
        a, _ = sigmoid(z)
        prediction = (~(a < 0.5)).astype(int)

        return prediction

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

class ProcessData():
    def __init__(self, x, y, dev_perc, test_perc):
        """
        Parameters:
        -----------
        x : array_like
            x.shape = (features, examples)
        y : array_like
            y.shape = (label, examples)
        dev_perc : float
        test_perc : float

        Returns:
        --------
        None
        """
        self.x = x.T
        self.y = y.T
        self.dev_perc = dev_perc
        self.test_perc = test_perc

        self.split()
        self.normalize()


    def split(self):
        """
        Parameters:
        -----------
        None

        Returns:
        --------
        None
        """
        percent = self.dev_perc + self.test_perc
        x_train, x_aux, y_train, y_aux = train_test_split(self.x, self.y, test_size=percent)
        self.train = {'x' : x_train.T, 'y' : y_train.T}
        new_percent = self.test_perc / percent
        x_dev, x_test, y_dev, y_test = train_test_split(x_aux, y_aux, test_size=new_percent)
        self.dev = {'x' : x_dev.T, 'y' : y_dev.T}
        self.test = {'x' : x_test.T, 'y' : y_test.T}

    def normalize(self, z=None, eps=1e-8):
        """
        Parameters:
        -----------
        z : array_like
            Default : None - For initialization
        eps : float
            Default 1e-8 - For stability

        Returns:
        z_scale : array_like
        """
        if z == None: 
            x = self.train['x']
            self.mu = np.mean(x, axis=1, keepdims=True)
            self.var = np.var(x, axis=1, keepdims=True)
            self.theta = 1 / np.sqrt(self.var + eps)
            self.train['x'] = self.theta * (x - self.mu)
            self.dev['x'] = self.theta * (self.dev['x'] - self.mu)
            self.test['x'] = self.theta * (self.test['x'] - self.mu)
        else:
            z_scale = self.theta * (z - self.mu)
            return z_scale

def main_scratch():
    csv = Path('neuralNetworks/src/python/data/housepricedata.csv')
    df =pd.read_csv(csv)
    dataset = df.values
    x = dataset[:, 0:10]
    y = dataset[:, 10].reshape(-1, 1)

    data = ProcessData(x.T, y.T, 0.15, 0.15)
    model = Model(2)
    costs, params = model.fit(data.train['x'], data.train['y'], 0.01, 0.01)
    dev_acc = model.accuracy(data.dev['x'], data.dev['y'], params)
    print(f'The accuracy on the dev set: {dev_acc}.')
    test_acc = model.accuracy(data.test['x'], data.test['y'], params)
    print(f'The accuracy on the test set: {test_acc}.')
 
def main_imports():
    csv = Path('neuralNetworks/src/python/data/housepricedata.csv')
    df = pd.read_csv(csv)
    dataset = df.values
    x = dataset[:, 0:10]
    y = dataset[:, 10].reshape(-1, 1)

    data = ProcessData(x.T, y.T, 0.15, 0.15)
    x_train = data.train['x'].T
    y_train = data.train['y'].reshape(-1)
    x_dev = data.dev['x'].T
    y_dev = data.dev['y'].reshape(-1)
    x_test = data.test['x'].T
    y_test = data.test['y'].reshape(-1)

    log_reg = LogisticRegression()

    log_reg.fit(x_train, y_train)
    dev_acc = log_reg.score(x_dev, y_dev)
    print(f'The accuracy on the dev set: {dev_acc}.')
    test_acc = log_reg.score(x_test, y_test)
    print(f'The accuracy on the test set: {test_acc}.')

if __name__ == '__main__':
    main_scratch()
    main_imports()

"""
Terminal Output:
Cost after iteration 0: 0.6969560526003767
Cost after iteration 1000: 0.28991745368387156
Cost after iteration 2000: 0.2726522372917119
Cost after iteration 3000: 0.26662849120350673
Cost after iteration 4000: 0.2634303377416742
Cost after iteration 5000: 0.26146337689032983
Cost after iteration 6000: 0.260153625195817
Cost after iteration 7000: 0.2592344557940492
Cost after iteration 8000: 0.25856572665860506
Cost after iteration 9000: 0.258066497005197
The accuracy on the dev set: 0.8767123287671232.
The accuracy on the test set: 0.9315068493150684.
The accuracy on the dev set: 0.9041095890410958.
The accuracy on the test set: 0.908675799086758.
"""