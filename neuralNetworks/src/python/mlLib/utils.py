#! python3

import numpy as np
from sklearn.model_selection import train_test_split

import mlLib.npActivators as npActivators

## Shuffle, split and normalize data
class ProcessData():
    def __init__(self, x, y, test_percent, *dev_percents, seed=1, shuffle=True, feat_as_col=True):
        """
        Parameters:
        -----------
        x : array_like
            x.shape = (examples, features)
        y : array_like
            y.shape = (examples, labels)
        test_percent : float
        dev_pervents : Tuple(floats)
        seed : int
            Default = 1
        shuffle : Boolean
            Default = True
        feat_as_col : Boolean
            Default = True
        
        Returns:
        --------
        None
        """
        self.x = x
        self.y = y
        self.test_percent = test_percent
        self.dev_percent = list(dev_percents)
        self.k_fold = len(self.dev_percent)
        self.seed = seed
        self.shuffle = shuffle
        self.feat_as_col = feat_as_col

        self.split()
        self.normalize()
        
        print(f"x_train.shape: {self.train['x'].shape}")
        print(f"y_train.shape: {self.train['y'].shape}")
        print(f"x_test.shape: {self.test['x'].shape}")
        print(f"y_test.shape: {self.test['y'].shape}")
        for k in range(self.k_fold):
            print(f"x_dev[{k}].shape: {self.dev['x'][k].shape}")
            print(f"y_dev[{k}].shape: {self.dev['y'][k].shape}")

    def split(self):
        """
        Parameters:
        -----------
        None

        Returns:
        --------
        None
        """
        x_aux, x_test, y_aux, y_test = train_test_split(self.x, self.y, test_size=self.test_percent, random_state=self.seed, shuffle=self.shuffle)
        left_over = 1 - self.test_percent
        x_dev = []
        y_dev = []
        for perc in self.dev_percent:
            aux_perc = perc / left_over
            x_aux, x_d, y_aux, y_d = train_test_split(x_aux, y_aux, test_size=aux_perc, random_state=self.seed)
            x_dev.append(x_d)
            y_dev.append(y_d)
            left_over -= perc
        
        if self.feat_as_col:
            self.train = {'x' : x_aux, 'y' : y_aux}
            self.test = {'x' : x_test, 'y' : y_test}
            self.dev = {'x' : x_dev, 'y' : y_dev}
        else:
            self.train = {'x' : x_aux.T, 'y' : y_aux.T}
            self.test = {'x' : x_test.T, 'y' : y_test.T}
            x_dev = [cv.T for cv in x_dev]
            y_dev = [cv.T for cv in y_dev]
            self.dev = {'x' : x_dev, 'y' : y_dev}

    def normalize(self, z=None, eps=0.0):
        """
        Parameters:
        -----------
        z : array_like
            Default : None - For initialization
        eps : float
            Default 0.0 - For stability

        Returns:
        z_scale : array_like
        """
        if z == None:
            x = self.train['x']
            axis = 0 if self.feat_as_col else 1
            self.mu = np.mean(x, axis=axis, keepdims=True)
            self.var = np.var(x, axis=axis, keepdims=True)
            self.theta = 1 / np.sqrt(self.var + eps)
            self.train['x'] = self.theta * (x - self.mu)
            self.test['x'] = self.theta * (self.test['x'] - self.mu)
            for k in range(self.k_fold):
                self.dev['x'][k] = self.theta * (self.dev['x'][k] - self.mu)

        else:
            z_scale = self.theta * (z - self.mu)
            return z_scale

class LinearParameters():
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

        dout = np.einsum('ij,ik->jk', self.w, din)
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


## Functions
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



if __name__ == '__main__':
    import pandas as pd
    from pathlib import Path

    csv = Path('neuralNetworks/src/python/data/housepricedata.csv')
    df = pd.read_csv(csv)
    dataset = df.values
    x = dataset[:, :10]
    y = dataset[:, 10].reshape(-1, 1)
    print(x.shape[0])
    data = ProcessData(x, y, 0.2, 0.2, feat_as_col=True)
    
    