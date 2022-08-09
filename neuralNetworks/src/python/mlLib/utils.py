#! python3

import time

import numpy as np
from sklearn.model_selection import train_test_split

import mlLib.npActivators as npActivators

## Classes
## Timing Epoch
class EpochRuntime():
    def __init__(self):
        self.current = time.time()
    
    def elapsed_time(self):
        elapsed = time.time() - self.current
        mins, secs = elapsed // 60, elapsed % 60
        txt = 'Elapsed time for the most recent epoch: {0} minutes and {1:0.3f}seconds'.format(mins, secs)
        print(txt)
        self.current = time.time()
  

## Shuffle, split and normalize full dataset
class ProcessData():
    def __init__(self, x, y, test_percent, dev_percent=0.0, seed=101, shuffle=True, feat_as_col=True, norm=True):
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
        if norm:
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

    def backward(self, din, a_prev,):
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

        return dout

    def update(self, learning_rate):
        """
        Parameters:
        -----------
        learning_rate : float
        
        Returns:
        --------
        None
        """
        w = self.w - learning_rate * self.dw
        self.w = w

        if self.bias:
            b = self.b - learning_rate * self.db
            self.b = b


class BatchNormParameters:
    def __init__(self, dim, eps=1e-8):
        """
        Parameters:
        -----------
        dim = int
        eps : float
            Default = 10^{-8}

        Returns:
        --------
        None
        """
        self.dims = (dim, 1)
        self.eps = eps
        self.gamma = np.ones(self.dims)
        self.beta = np.zeros(self.dims)

        self.running_mean = np.zeros(self.dims)
        self.running_var = np.zeros(self.dims)

    def normalize(self, u, momentum=0.9):
        """
        Parameters:
        -----------
        u : array_like
            u.shape == (n, N)
        momentum : float
            Default = 0.9   # Calculating running statistics

        Returns:
        uhat : array_like
            uhat.hape == (n, N)
        ruhat : array_like
            ruhat.shape == (n, N, n, N)
        """
        # Compute normalization
        mu = np.mean(u, axis=1, keepdims=True)
        sigma2 = np.var(u, axis=1, keepdims=True)
        theta = 1 / np.sqrt(sigma2 + self.eps)
        uhat = theta * (u - mu)

        # Update running mean and variance
        self.running_mean = momentum * self.running_mean + (1 - momentum) * mu
        self.running_var = momentum * self.running_var + (1 - momentum) * sigma2

        # Compute reverse differential
        m, n = u.shape
        duhat = np.zeros((m, n, m, n))
        I_m = np.eye(m)
        I_n = np.eye(n)
        for alpha in range(m):
            for beta in range(n):
                for i in range(m):
                    for j in range(n):
                        duhat[alpha, beta, i, j] = (
                            I_m[alpha, i]
                            * theta[alpha, 0]
                            * (
                                I_n[j, beta]
                                - (1 + uhat[alpha, j] * uhat[alpha, beta]) / n
                            )
                        )

        ruhat = np.einsum("ijkl->klij", duhat)

        return uhat, ruhat

    def forward(self, u):
        """
        Parameters:
        -----------
        u : array_like
            u.shape == (n, N)

        Returns:
        z : array_like
            z.shape == (n, N)
        """
        self.norm, self.dnorm = self.normalize(u)
        z = self.gamma * self.norm + self.beta
        return z

    def backward(self, d_in):
        """
        Parameters:
        -----------
        d_in : array_like
            d_in.shape == (n, N)
        """
        self.dbeta = np.sum(d_in, axis=1, keepdims=True)
        self.dgamma = np.sum(self.norm * d_in, axis=1, keepdims=True)

        return np.einsum("ijkl,kl", self.dnorm, d_in)

    def update(self, learning_rate):
        """
        Parameters:
        -----------
        learning_rate : float

        Returns:
        --------
        None
        """
        self.gamma = self.gamma - learning_rate * self.dgamma
        self.beta = self.beta - learning_rate * self.dbeta

    def evaluate(self, u):
        """
        Parameters:
        -----------
        u : array_like
            u.shape == (n, N)

        Returns:
        z : array_like
            z.shape == (n, N)
        """
        z = (u - self.running_mean) / np.sqrt(self.running_var + self.eps)
        z = self.gamma * z + self.beta
        return z

###################################################

## Functions

## Map the labels {0,1,2,...,C-1} to basis vectors {e_1,...,e_C}
def encode_labels(y, C):
    """
    Parameters:
    ----------
    y : array_like
        y.shape == (N,)
    C : int

    Returns:
    Y : array_like
        Y.shape == (C, N)
    """
    N = y.size
    Y = np.zeros((C, N))
    for i in range(C):
        for j in range(N):
            if y[j] == i:
                Y[i, j] = 1

    return Y


## Map the one-hot encoded vecors to labels {0,1,...,C-1}
def decode_labels(Y):
    """
    Parameters:
    Y : array_like
        Y.shape == (C, N)

    Returns:
    --------
    y : array_like
        y.shape == (N,)
        y[0,j]
    """
    N = Y.shape[1]
    y = np.zeros(N)
    labels, col_index = np.nonzero(Y)
    y[col_index] = labels
    return y

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



## Loss functions

# The least-squared-error function
def lse(a, y):
    """
    Parameters:
    -----------
    a : array_like
    y : array_like
        a.shape == y.shape

    Returns:
    --------
    loss : array_like
    rloss : array_like
        rloss.shape == a.shape
    """
    loss = ((a - y)**2) / 2
    rloss = a - y
    return loss, rloss


# The log-loss function for binary classification
def log_loss(a, y):
    """
    Parameters:
    -----------
    a : array_like
    y : array_like
        a.shape == y.shape

    Returns:
    --------
    loss : array_like
    rloss : array_like
        rloss.shape == a.shape
    """
    loss = -1 * (y * np.log(a) + (1 - y) * np.log(1 - a))
    rloss = -(y / a) + (1 - y) / (1 - a)
    return loss, rloss

## The cross-entropy loss function
def cross_entropy(a, y, eps=1e-8):
    """
    Parameters:
    -----------
    a : array_like
    y : array_like
        a.shape == y.shape
    eps : float
        Default = 10^{-8} # For stability

    Returns:
    --------
    loss : array_like
    r_loss : array_like
        rloss.shape == a.shape
    """
    assert a.shape == y.shape, "a and y have different shapes"

    a = np.clip(a, eps, 1 - eps)
    loss = -1 * np.sum(y * np.log(a), axis=0)
    rloss = -1 * y / a
    return loss, rloss


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
    
    