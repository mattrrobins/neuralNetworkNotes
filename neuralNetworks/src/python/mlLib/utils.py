#! python3

import numpy as np
from sklearn.model_selection import train_test_split

import mlLib.npActivators as npActivators

## Classes

## Shuffle, split and normalize full dataset
class ProcessData():
    def __init__(self, x, y, test_percent, dev_percent=0.0, seed=101, shuffle=True, feat_as_col=True):
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
        self.dev_percent = dev_percent
        self.seed = seed
        self.shuffle = shuffle
        self.feat_as_col = feat_as_col

        self.split()
        self.normalize()
        
        print(f"x_train.shape: {self.train['x'].shape}")
        print(f"y_train.shape: {self.train['y'].shape}")
        print(f"x_test.shape: {self.test['x'].shape}")
        print(f"y_test.shape: {self.test['y'].shape}")
        if self.dev_percent > 0.0:
            print(f"x_dev.shape: {self.dev['x'].shape}")
            print(f"y_dev.shape: {self.dev['y'].shape}")

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
        aux_perc = self.dev_percent / left_over
        x_train, x_dev, y_train, y_dev = train_test_split(x_aux, y_aux, test_size=aux_perc, random_state=self.seed)
        
        if self.feat_as_col:
            self.train = {'x' : x_train, 'y' : y_train}
            self.test = {'x' : x_test, 'y' : y_test}
            self.dev = {'x' : x_dev, 'y' : y_dev}
        else:
            self.train = {'x' : x_train.T, 'y' : y_train.T}
            self.test = {'x' : x_test.T, 'y' : y_test.T}
            self.dev = {'x' : x_dev.T, 'y' : y_dev.T}

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
            self.dev['x'] = self.theta * (self.dev['x'] - self.mu)

        else:
            z_scale = self.theta * (z - self.mu)
            return z_scale

## Shuffle and create mini-batches during training
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


## Gradient descent with exponentially moving averages
class Momentum:
    def __init__(self, param, bias, beta1=0.9):
        """
        Parameters:
        -----------
        param : LinearParameters
        bias : Bool
        beta1 : float
            Default = 0.9

        Returns:
        --------
        None
        """
        self.bias = bias
        self.beta1 = beta1
        self.w = np.zeros(param.w.shape)
        if self.bias:
            self.b = np.zeros(param.b.shape)

    def update(self, param, learning_rate, iter, update_params=True):
        """
        Parameters:
        -----------
        param : LinearParameter
        learning_rate : float
        iter : int
        update_params : Bool
            Default = True  - Dictates return type

        Returns:
        --------
        None OR v : Dict[array_like]
        """
        self.w = self.beta1 * self.w + (1 - self.beta1) * param.dw
        vw_corrected = self.w / (1 - self.beta1**iter)
        if update_params:
            param.w = param.w - learning_rate * vw_corrected
        if self.bias:
            self.b = self.beta1 * self.b + (1 - self.beta1) * param.db
            vb_corrected = self.b / (1 - self.beta1**iter)
            if update_params:
                param.b = param.b - learning_rate * vb_corrected
        if not update_params:
            v = {}
            v["w"] = vw_corrected
            if self.bias:
                v["b"] = vb_corrected
            return v

## Gradient descent with root mean squared propagation
class RMSProp:
    def __init__(self, param, bias, beta2=0.9, eps=1e-8):
        """
        Parameters:
        -----------
        params : LinearParameters
        bias : Bool
        beta2 : float
            Default = 0.9
        eps : float
            Default = 10^{-8}

        Returns:
        None
        """
        self.bias = bias
        self.beta2 = beta2
        self.eps = eps
        self.w = np.zeros(param.w.shape)
        if self.bias:
            self.b = np.zeros(param.b.shape)

    def update(self, param, learning_rate, iter, update_params=True):
        """
        Parameters:
        -----------
        params : LinearParameters
        learning_rate : float
        iter : int
        update_params : Boolean
            Default = True

        Returns:
        None OR v : Dict[array_like]
        """
        self.w = self.beta2 * self.w + (1 - self.beta2) * (param.dw**2)
        sw_corrected = self.w / (1 - self.beta2**iter)
        if update_params:
            param.w = param.w - learning_rate * (
                param.dw / (np.sqrt(sw_corrected) + self.eps)
            )
        if self.bias:
            self.b = self.beta2 * self.b + (1 - self.beta2) * (param.db**2)
            sb_corrected = self.b / (1 - self.beta2**iter)
            if update_params:
                param.b = param.b - learning_rate * (
                    param.db / (np.sqrt(sb_corrected) + self.eps)
                )
        if not update_params:
            s = {}
            s["w"] = sw_corrected
            if self.bias:
                s["b"] = sb_corrected
            return s

class Adam:
    def __init__(self, param, bias, beta1=0.9, beta2=0.999, eps=1e-8):
        """
        Parameters:
        -----------
        param : LinearParameters
        bias : Bool
        beta1 : float
            Default = 0.9
        beta2 : float
            Default = 0.999
        eps : float
            Default = 10^{-8}

        Returns:
        None
        """
        self.bias = bias
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps

        self.mom = Momentum(param, self.bias, self.beta1)
        self.rmsprop = RMSProp(param, self.bias, self.beta2, self.eps)

    def update(self, param, learning_rate, iter):
        """
        Parameters:
        -----------
        params : LinearParameters
        learning_rate : float
        iter : int

        Returns:
        None
        """
        v = self.mom.update(param, learning_rate, iter, False)
        s = self.rmsprop.update(param, learning_rate, iter, False)

        param.w = param.w - learning_rate * v["w"] / (np.sqrt(s["w"]) + self.eps)
        if self.bias:
            param.b = param.b - learning_rate * v["b"] / (np.sqrt(s["b"]) + self.eps)

## Initializing, utilizing and updating the weight and bias parameters
class LinearParameters():
    def __init__(self, dims, bias=True, seed=1011):
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


###################################################

## Functions

## Applying the activator function after an affine-linear transformation
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
    
    