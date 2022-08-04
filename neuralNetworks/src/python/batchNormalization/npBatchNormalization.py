#! python3

import numpy as np

from mlLib.utils import LinearParameters, ShuffleBatchData, EpochRuntime
from mlLib.utils import apply_activation


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

    def normalize(self, u):
        """
        Parameters:
        -----------
        u : array_like
            u.shape == (n, N)
        iter : int

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
        momentum = 0.9
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


class NeuralNetwork:
    def __init__(self, config):
        """
        Parameters:
        -----------
        config : Dict
            config['lp_reg'] = 0,1,2
            config['batch_size'] = 2 ** p # p in {5, 6, 7, 8, 9, 10}
            config['nodes'] = List[int]
            config['bias'] = List[Bool]
            config['batch_norm'] = List[Bool]
            config['activators'] = List[str]
            config['keep_probs'] = List[float]

        Returns:
        --------
        None
        """
        self.config = config
        self.lp_reg = config["lp_reg"]
        self.batch_size = config["batch_size"]
        self.nodes = config["nodes"]
        self.bias = config["bias"]
        self.batch_norm = config["batch_norm"]
        self.activators = config["activators"]
        self.keep_probs = config["keep_probs"]
        self.L = len(config["nodes"]) - 1

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
            assert D[l].shape == (
                self.nodes[l],
                num_examples,
            ), "Dropout matrices are the wrong shape"

        return D

    def forward_propagation(self, x, dropout=None):
        """
        Parameters:
        -----------
        x : array_like
        dropout : Dict[array_like]
            Default = None

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
            a[0] *= dropout[0]

        for l in range(1, self.L + 1):
            z = self.lin_params[l].forward(a[l - 1])
            if self.batch_norm[l]:
                z = self.bn_params[l].forward(z)
            a[l], dg[l] = apply_activation(z, self.activators[l])
            if dropout != None:
                a[l] *= dropout[l]

        cache = {"a": a, "dg": dg}
        return cache

    def cost_function(self, a, y, lambda_=0.01, eps=1e-8):
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
        for param in self.lin_params.values():
            R += np.sum(np.abs(param.w) ** self.lp_reg)
        R *= lambda_ / (2 * n)

        # Compute unregularized cost
        a = np.clip(a, eps, 1 - eps)  # Bound a for stability
        J = (-1 / n) * (np.sum(y * np.log(a) + (1 - y) * np.log(1 - a)))

        cost = float(np.squeeze(J + R))

        return cost

    def backward_propagation(self, cache, y, dropout):
        """
        Parameters:
        -----------
        cache : Dict[array_like]
            cache['a'] : array_like
            cache['dg'] : array_like
        y : array_like
        dropout : Dict[array_like]

        Returns:
        --------
        None
        """

        # Retrieve cache
        a = cache["a"]
        dg = cache["dg"]

        # Initialize differentials along the network
        delta = {}
        delta[self.L] = ((a[self.L] - y) / y.shape[1]) * dropout[self.L]

        for l in reversed(range(1, self.L + 1)):
            delta[l - 1] = (
                dg[l - 1]
                * self.lin_params[l].backward(delta[l], a[l - 1])
                * dropout[l - 1]
            )
            if self.batch_norm[l - 1]:
                delta[l - 1] = self.bn_params[l - 1].backward(delta[l - 1])

    def update_parameters(self, learning_rate):
        """
        Parameters:
        -----------
        learning_rate : float

        Returns:
        --------
        None
        """
        for l in range(1, self.L + 1):
            self.lin_params[l].update(learning_rate)
            if self.batch_norm[l]:
                self.bn_params[l].update(learning_rate)

    def fit(
        self,
        data,
        learning_rate,
        lambda_=0.01,
        num_epochs=10000,
        print_cost_epoch=1000,
    ):
        """
        Parameters:
        -----------
        data : Dict[array_like]
            data['x'] : array_like
            data['y'] : array_like
        learning_rate : float
        lambda_ : float
            Default = 0.01
        num_epochs : int
            Default = 10000
        print_cost_epoch : int
            Default = 1000   # 0 Doesn't print costs

        Returns:
        --------
        costs : List[floats]
        params : Dict[LinearParameters]
        """
        # Initialize parameters per layer
        self.lin_params = {}
        self.bn_params = {}
        for l in range(1, self.L + 1):
            self.lin_params[l] = LinearParameters(
                (self.nodes[l], self.nodes[l - 1]), self.bias[l]
            )
            if self.batch_norm[l]:
                self.bn_params[l] = BatchNormParameters(self.nodes[l])

        # Initialize batching
        batching = ShuffleBatchData(data, self.batch_size)

        costs = []
        time = EpochRuntime()
        for epoch in range(num_epochs):
            batches = batching.get_batches()
            B = len(batches)
            k = 1
            cost = 0
            for batch in batches:
                x = batch["x"]
                y = batch["y"]
                dropout = self.init_dropout(x.shape[1])
                cache = self.forward_propagation(x, dropout)
                batch_cost = self.cost_function(cache["a"][self.L], y, lambda_)
                cost += x.shape[1] * batch_cost
                self.backward_propagation(cache, y, dropout)
                self.update_parameters(learning_rate)
                k += 1
            cost /= data["x"].shape[1]
            costs.append(cost)
            time.elapsed_time()

            if (print_cost_epoch != 0) and (epoch % print_cost_epoch == 0):
                print(f"Cost after epoch {epoch}: {cost}")

        return costs

    def evaluate(self, x):
        """
        Parameters:
        -----------
        x : array_like

        Returns:
        --------
        y_hat : array_like
        """
        a = {}
        a[0], _ = apply_activation(x, self.activators[0])
        for l in range(1, self.L + 1):
            z = self.lin_params[l].forward(a[l - 1])
            if self.batch_norm[l]:
                z = self.bn_params[l].evaluate(z)
            a[l], _ = apply_activation(z, self.activators[l])

        y_hat = (~(a[self.L] < 0.5)).astype(int)
        return y_hat

    def accuracy(self, data):
        """
        Parameters:
        -----------
        data : Dict[array_like]
            data['x'] : array_like
            data['y'] : array_like

        Returns:
        --------
        accuracy : float
        """
        x = data["x"]
        y = data["y"]

        y_hat = self.evaluate(x)
        acc = np.sum(y_hat == y) / y.shape[1]

        return acc


if __name__ == "__main__":
    from pathlib import Path

    import pandas as pd

    from mlLib.utils import ProcessData

    csv = Path("neuralNetworks/src/python/data/housepricedata.csv")
    # csv = Path("neuralNetworks/src/python/data/pima-indians-diabetes.csv")
    df = pd.read_csv(csv)
    dataset = df.values
    x = dataset[:, :-1]
    y = dataset[:, -1].reshape(-1, 1)
    data = ProcessData(x, y, 0.1, 0.1, seed=1, feat_as_col=False)

    config = {
        "lp_reg": 2,
        "batch_size": 2**5,
        "nodes": [data.train["x"].shape[0], 32, 16, data.train["y"].shape[0]],
        "bias": [False, False, False, True],
        "batch_norm": [False, True, True, False],
        "activators": ["linear", "relu", "relu", "sigmoid"],
        "keep_probs": [1, 1, 1, 1],
    }

    model = NeuralNetwork(config)
    costs = model.fit(
        data.train,
        learning_rate=0.1,
        lambda_=0.0,
        num_epochs=3,
        print_cost_epoch=1,
    )

    train_acc = model.accuracy(data.train)
    print(f"Training Accuracy: {train_acc}")
    dev_acc = model.accuracy(data.dev)
    print(f"Dev Accuracy: {dev_acc}")
    test_acc = model.accuracy(data.test)
    print(f"Test Accuracy: {test_acc}")

    import matplotlib.pyplot as plt

    plt.plot(costs, "ro")
    plt.ylabel("Cost")
    plt.xlabel("Epochs")
    plt.show()
