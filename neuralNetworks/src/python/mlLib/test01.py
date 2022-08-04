#! python3

import numpy as np

from mlLib.npLossFunctions import LOSS_FUNCTIONS


class NeuralNetwork:
    def __init__(self, config):
        """
        Parameters:
        -----------
        config : Dict
            config['batch_size'] = 2^p # p in {5, 6, 7, 8, 9, 10}
            config['nodes'] = List[int]
            config['bias'] = List[Bool]
            config['batch_norm'] = List[Bool]
            config['activators'] = List[str] # str in ACTIVATORS
            config['keep_probs'] = List[float]
            congif['loss'] = str # str in ['log_loss', 'cross_entropy', 'lse']
            config['optimizer'] = str

        Returns:
        --------
        None
        """
        self.batch_size = config["batch_size"]
        self.nodes = config["nodes"]
        self.bias = config["bias"]
        self.batch_norm = config["batch_norm"]
        self.activators = config["activators"]
        self.keep_probs = config["keep_probs"]
        self.loss = LOSS_FUNCTIONS[config["loss"]]

    def fit(self, data, learning_rate, lambda_, epochs, print_cost_epoch):
        """
        Parameters:
        -----------
        data : Dict[array_like]
            data['x'] : array_like
            data['x'].shape == (features, examples)
            data['y'] : array_like
            data['y'].shape == (targets, examples)
        learning_rate : float
        lambda_ : float
        num_epochs : int
        print_cost_epoch : int # 0 Doesn't print costs

        Returns:
        --------
        costs : List[floats]
        """
        ## Initialize parameters in dictionary with layer as key


class Momentum:
    def __init__(self, param, beta1=0.9):
        self.param = param
        self.beta1 = beta1


class Parameters:
    def __init__(self, dims, batch_norm=True, bias=False):
        """
        Parameters:
        -----------

        Returns:
        --------
        """
        assert (
            batch_norm != bias
        ), "Comment out this assertion if you desire a bias parameter with batch normalization during this layer"

        self.n = dims[0]
        self.batch_norm = batch_norm
        self.bias = bias

        self.w = np.random.randn(*dims) * 0.01
        if self.bias:
            self.b = np.zeros((self.n, 1))
        if self.batch_norm:
            self.eps = 1e-8
            self.gamma = np.ones((self.n, 1))
            self.beta = np.zeros((self.n, 1))
            self.running_mean = np.zeros((self.n, 1))
            self.running_var = np.zeros((self.n, 1))

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

    def train_forward(self, a, running_momentum=0.9):
        """
        Parameters:
        -----------

        Returns:
        --------
        """
        z = self.w @ a
        if self.bias:
            z += self.b
        if self.batch_norm:
            self.norm, self.rnorm = self.normalize(z, running_momentum)
            z = self.gamma * self.norm + self.beta
        
        return z

    def backward(self, delta_in):
        """
        Parameters:
        -----------

        Returns:
        --------
        """
        if self.batch_norm:
            self.dbeta = np.sum(delta_in, axis=1, keepdims=True)
            self.dgamma = np.sum(self.norm * delta_in, axis=1, keepdims=True)
        if self.bias:
            ###self.db = np.
            #### fix batch normalization again.  We're missing gamma when moving in the network direction

