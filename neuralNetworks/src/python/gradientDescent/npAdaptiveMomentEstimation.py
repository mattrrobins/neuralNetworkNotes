#! python3

import numpy as np

from mlLib.utils import LinearParameters, ShuffleBatchData, Momentum, RMSProp
from mlLib.utils import apply_activation


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


class NeuralNetwork:
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
        self.lp_reg = config["lp_reg"]
        self.batch_size = config["batch_size"]
        self.nodes = config["nodes"]
        self.bias = config["bias"]
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

        cache = {"a": a, "dg": dg}
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
        R *= lambda_ / (2 * n)

        # Compute unregularized cost
        a = np.clip(a, eps, 1 - eps)  # Bound a for stability
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
        a = cache["a"]
        dg = cache["dg"]

        # Initialize differentials along the network
        delta = {}
        delta[self.L] = ((a[self.L] - y) / y.shape[1]) * dropout[self.L]

        for l in reversed(range(1, self.L + 1)):
            delta[l - 1] = (
                dg[l - 1] * params[l].backward(delta[l], a[l - 1]) * dropout[l - 1]
            )

    def update_parameters(self, params, adams, learning_rate, iter):
        """
        Parameters:
        -----------
        params : Dict[LinearParameters]
            params[l].w = Weights
            params[l].b = Bias
        adams : Dict[Adam]
        learning_rate : float
        iter : int

        Returns:
        --------
        None
        """
        for l in params.keys():
            adams[l].update(params[l], learning_rate, iter)

    def fit(
        self,
        data,
        learning_rate=0.1,
        lambda_=0.01,
        num_epochs=10000,
        print_cost_iter=1000,
    ):
        """
        Parameters:
        -----------
        data : Dict[array_like]
            data['x'] : array_like
            data['y'] : array_like
        learning_rate : float
            Default : 0.1
        lambda_ : float
            Default : 0.01
        num_epochs : int
            Default : 10000
        print_cost_iter : int
            Default: 1000   # 0 Doesn't print costs

        Returns:
        --------
        costs : List[floats]
        params : Dict[LinearParameters]
        """
        # Initialize parameters and optimzer per layer
        params = {}
        adams = {}
        for l in range(1, self.L + 1):
            params[l] = LinearParameters(
                (self.nodes[l], self.nodes[l - 1]), self.bias[l]
            )
            adams[l] = Adam(params[l], self.bias[l])

        # Initialize batching
        batching = ShuffleBatchData(data, self.batch_size)

        costs = []
        for epoch in range(num_epochs):
            batches = batching.get_batches()
            B = len(batches)
            k = 1
            cost = 0
            for batch in batches:
                iter = (epoch * B) + k
                x = batch["x"]
                y = batch["y"]
                dropout = self.init_dropout(x.shape[1])
                cache = self.forward_propagation(params, x, dropout)
                batch_cost = self.cost_function(params, cache["a"][self.L], y, lambda_)
                cost += x.shape[1] * batch_cost
                self.backward_propagation(params, cache, y, dropout)
                self.update_parameters(params, adams, learning_rate, iter)
                k += 1
            cost /= data["x"].shape[1]
            costs.append(cost)

            if (print_cost_iter != 0) and (epoch % print_cost_iter == 0):
                print(f"Cost after epoch {epoch}: {cost}")

        return params, costs

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
        a = cache["a"][self.L]
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
        x = data["x"]
        y = data["y"]

        y_hat = self.evaluate(params, x)
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
        "bias": [False, True, True, True],
        "activators": ["linear", "relu", "relu", "sigmoid"],
        "keep_probs": [1, 1, 1, 1],
    }

    model = NeuralNetwork(config)
    params, costs = model.fit(
        data.train,
        learning_rate=0.0001,
        lambda_=0.5,
        num_epochs=150,
        print_cost_iter=20,
    )

    train_acc = model.accuracy(params, data.train)
    print(f"Training Accuracy: {train_acc}")
    dev_acc = model.accuracy(params, data.dev)
    print(f"Dev Accuracy: {dev_acc}")
    test_acc = model.accuracy(params, data.test)
    print(f"Test Accuracy: {test_acc}")

    import matplotlib.pyplot as plt

    plt.plot(costs, "ro")
    plt.ylabel("Cost")
    plt.xlabel("Epochs")
    plt.show()
