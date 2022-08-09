#! python3

from tkinter import N
import numpy as np

from mlLib.npLossFunctions import LOSS_FUNCTIONS, log_loss
from mlLib.utils import ShuffleBatchData
from mlLib.utils import apply_activation


class Parameters:
    def __init__(
        self,
        dims,
        batch_norm=True,
        bias=False,
        optimizer="sgd",
        beta1=0.9,
        beta2=0.999,
        eps=1e-8,
    ):
        """
        Parameters:
        -----------

        Returns:
        --------
        """
        assert (
            batch_norm != bias
        ), "Bias in a layer with batch normalization is redundant"

        self.n = dims[0]
        self.batch_norm = batch_norm
        self.bias = bias

        self.optimizer = optimizer
        self.cond_mom = optimizer in ["momentum", "adam"]
        self.cond_rmsprop = optimizer in ["rmsprop", "adam"]

        if self.cond_mom:
            self.v = {}
            self.beta1 = beta1
        if self.cond_rmsprop:
            self.s = {}
            self.beta2 = beta2
            self.eps = eps

        self.w = np.random.randn(*dims) * 0.01
        if self.cond_mom:
            self.v["w"] = np.zeros(self.w.shape)
        if self.cond_rmsprop:
            self.s["w"] = np.zeros(self.w.shape)

        if self.bias:
            self.b = np.zeros((self.n, 1))
            if self.cond_mom:
                self.v["b"] = np.zeros(self.b.shape)
            if self.cond_rmsprop:
                self.s["b"] = np.zeros(self.b.shape)
        if self.batch_norm:
            self.gamma = np.ones((self.n, 1))
            self.beta = np.zeros((self.n, 1))
            self.running_mean = np.zeros((self.n, 1))
            self.running_var = np.zeros((self.n, 1))
            if self.cond_mom:
                self.v["gamma"] = np.zeros(self.gamma.shape)
                self.v["beta"] = np.zeros(self.beta.shape)
            if self.cond_rmsprop:
                self.s["gamma"] = np.zeros(self.gamma.shape)
                self.s["beta"] = np.zeros(self.beta.shape)

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
        self.theta = 1 / np.sqrt(sigma2 + 1e-8)
        uhat = self.theta * (u - mu)

        # Update running mean and variance
        self.running_mean = momentum * self.running_mean + (1 - momentum) * mu
        self.running_var = momentum * self.running_var + (1 - momentum) * sigma2

        self.norm = uhat

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
            self.normalize(z, running_momentum)
            z = self.gamma * self.norm + self.beta

        return z

    def test_forward(self, a):
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
            z = (z - self.running_mean) / np.sqrt(self.running_var + 1e-8)
            z = self.gamma * z + self.beta

        return z

    def backward(self, delta_in, a_prev):
        """
        Parameters:
        -----------

        Returns:
        --------
        """
        if self.batch_norm:
            self.dbeta = np.sum(delta_in, axis=1, keepdims=True)
            self.dgamma = np.sum(self.norm * delta_in, axis=1, keepdims=True)
            delta_in = self.gamma * delta_in
            delta_in = self.theta * delta_in - self.theta * (
                np.mean(delta_in, axis=1, keepdims=True)
                + self.norm * np.mean(self.norm * delta_in, axis=1, keepdims=True)
            )
        if self.bias:
            self.db = np.sum(delta_in, axis=1, keepdims=True)
        self.dw = np.einsum("ij,kj", delta_in, a_prev)

        delta_out = np.einsum("ij,ik->jk", self.w, delta_in)
        return delta_out

    def update(self, learning_rate, lambda_N, iter=None):
        """
        Parameters:
        -----------

        Returns:
        --------
        """
        self.dw += lambda_N * self.w
        if self.cond_mom:
            self.v["w"] = self.beta1 * self.v["w"] + (1 - self.beta1) * self.dw
            vw_corrected = self.v["w"] / (1 - self.beta1**iter)
            if self.cond_rmsprop:
                self.s["w"] = self.beta2 * self.s["w"] + (1 - self.beta2) * (
                    self.dw**2
                )
                sw_corrected = self.s["w"] / (1 - self.beta2**iter)
                self.w = self.w - learning_rate * vw_corrected / (
                    np.sqrt(sw_corrected) + self.eps
                )
            else:
                self.w = self.w - learning_rate * vw_corrected
        if self.cond_rmsprop:
            self.s["w"] = self.beta2 * self.s["w"] + (1 - self.beta2) * (self.dw**2)
            sw_corrected = self.s["w"] / (1 - self.beta2**iter)
            self.w = self.w - learning_rate * (
                self.dw / (np.sqrt(sw_corrected) + self.eps)
            )
        if self.optimizer == "sgd":
            self.w = self.w - learning_rate * self.dw

        if self.bias:
            if self.cond_mom:
                self.v["b"] = self.beta1 * self.v["b"] + (1 - self.beta1) * self.db
                vb_corrected = self.v["b"] / (1 - self.beta1**iter)
                if self.cond_rmsprop:
                    self.s["b"] = self.beta2 * self.s["b"] + (1 - self.beta2) * (
                        self.db**2
                    )
                    sb_corrected = self.s["b"] / (1 - self.beta2**iter)
                    self.b = self.b - learning_rate * vb_corrected / (
                        np.sqrt(sb_corrected) + self.eps
                    )
                else:
                    self.b = self.b - learning_rate * vb_corrected
            if self.cond_rmsprop:
                self.s["b"] = self.beta2 * self.s["b"] + (1 - self.beta2) * (
                    self.db**2
                )
                sb_corrected = self.s["b"] / (1 - self.beta2**iter)
                self.b = self.b - learning_rate * (
                    self.db / (np.sqrt(sb_corrected) + self.eps)
                )
            if self.optimizer == "sgd":
                self.b = self.b - learning_rate * self.db

        if self.batch_norm:
            if self.cond_mom:
                self.v["gammma"] = (
                    self.beta1 * self.v["gamma"] + (1 - self.beta1) * self.dgamma
                )
                vgamma_corrected = self.v["gamma"] / (1 - self.beta1**iter)
                self.v["beta"] = (
                    self.beta1 * self.v["beta"] + (1 - self.beta1) * self.dbeta
                )
                vbeta_corrected = self.v["beta"] / (1 - self.beta1**iter)
                if self.cond_rmsprop:
                    self.s["gamma"] = self.beta2 * self.s["gamma"] + (
                        1 - self.beta2
                    ) * (self.dgamma**2)
                    sgamma_corrected = self.s["gamma"] / (1 - self.beta2**iter)
                    self.gamma = self.gamma - learning_rate * vgamma_corrected / (
                        np.sqrt(sgamma_corrected) + self.eps
                    )
                    self.s["beta"] = self.beta2 * self.s["beta"] + (1 - self.beta2) * (
                        self.dbeta**2
                    )
                    sbeta_corrected = self.s["beta"] / (1 - self.beta2**iter)
                    self.beta = self.beta - learning_rate * vbeta_corrected / (
                        np.sqrt(sbeta_corrected) + self.eps
                    )
                else:
                    self.gamma = self.gamma - learning_rate * vgamma_corrected
                    self.beta = self.beta - learning_rate * vbeta_corrected
            if self.cond_rmsprop:
                self.s["gamma"] = self.beta2 * self.s["gamma"] + (1 - self.beta2) * (
                    self.dgamma**2
                )
                sgamma_corrected = self.s["gamma"] / (1 - self.beta2**iter)
                self.gamma = self.gamma - learning_rate * self.dgamma / (
                    np.sqrt(sgamma_corrected) + self.eps
                )
                self.s["beta"] = self.beta2 * self.s["beta"] + (1 - self.beta2) * (
                    self.dbeta**2
                )
                sbeta_corrected = self.s["beta"] / (1 - self.beta2**iter)
                self.beta = self.beta - learning_rate * self.dbeta / (
                    np.sqrt(sbeta_corrected) + self.eps
                )
            if self.optimizer == "sgd":
                self.gamma = self.gamma - learning_rate * self.dgamma
                self.beta = self.beta - learning_rate * self.dbeta


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
        self.L = len(self.nodes) - 1
        self.optimizer = config["optimizer"][0]
        self.optimizer_hps = config["optimizer"][1]

    def forward_propagation(self, x, train=True):
        """
        Parameters:
        -----------

        Returns:
        --------
        """
        # Initialize cache
        a = {}
        dg = {}

        a[0], dg[0] = apply_activation(x, self.activators[0])
        for l in range(1, self.L + 1):
            if train:
                z = self.params[l].train_forward(a[l - 1])
            else:
                z = self.params[l].test_forward(a[l - 1])
            a[l], dg[l] = apply_activation(z, self.activators[l])

        cache = {"a": a, "dg": dg}
        return cache

    def cost_function(self, a, y, lambda_=0.0):
        """
        Parameters:
        -----------

        Returns:
        --------
        """
        loss, rloss = self.loss(a, y)

        # L^2-Regularization
        R = 0
        for param in self.params.values():
            R += np.linalg.norm(param.w) ** 2
        R *= lambda_ / (2 * y.shape[1])

        J = np.mean(loss)

        rcost = rloss / y.shape[1]

        cost = float(np.squeeze(J + R))
        return cost, rcost

    def backward_propagation(self, cache, rcost, y):
        """
        Parameters:
        -----------

        Returns:
        --------
        """
        a = cache["a"]
        dg = cache["dg"]

        # Initialize differentials along the network direction
        delta = {}
        delta[self.L] = dg[self.L] * rcost

        for l in reversed(range(1, self.L + 1)):
            delta[l - 1] = dg[l - 1] * self.params[l].backward(delta[l], a[l - 1])

    def update(self, learning_rate, lambda_N, iter=None):
        """
        Parameters:
        -----------

        Returns:
        --------
        """
        for param in self.params.values():
            param.update(learning_rate, lambda_N, iter)

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
        self.params = {}
        for l in range(1, self.L + 1):
            self.params[l] = Parameters(
                (self.nodes[l], self.nodes[l - 1]),
                self.batch_norm[l],
                self.bias[l],
                self.optimizer,
                *self.optimizer_hps,
            )

        ## Initialize batching
        batching = ShuffleBatchData(data, self.batch_size)
        N = data["x"].shape[1]

        ## Main training loop
        costs = []
        for epoch in range(epochs):
            batches = batching.get_batches()
            B = len(batches)
            k = 1
            cost = 0
            for batch in batches:
                t = epoch * B + k
                x = batch["x"]
                y = batch["y"]
                cache = self.forward_propagation(x, train=True)
                batch_cost, rcost = self.cost_function(cache["a"][self.L], y, lambda_)
                cost += x.shape[1] * batch_cost
                self.backward_propagation(cache, rcost, y)
                self.update(learning_rate, lambda_ / x.shape[1], t)
                k += 1
            costs.append(cost / N)

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
        cache = self.forward_propagation(x, False)
        a = cache["a"][self.L]

        y_hat = (~(a < 0.5)).astype(int)
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
        "batch_size": 2**7,
        "nodes": [data.train["x"].shape[0], 32, 16, data.train["y"].shape[0]],
        "bias": [False, False, False, True],
        "batch_norm": [False, True, True, False],
        "activators": ["linear", "relu", "relu", "sigmoid"],
        "keep_probs": [1, 1, 1, 1],
        "loss": "log_loss",
        "optimizer": ("adam", [0.9, 0.999, 1e-8]),
    }

    model = NeuralNetwork(config)
    costs = model.fit(
        data.train, learning_rate=0.1, lambda_=0.1, epochs=500, print_cost_epoch=50
    )

    import matplotlib.pyplot as plt

    plt.plot(costs, "ro")
    plt.ylabel("Cost")
    plt.xlabel("Epochs")
    plt.show()
