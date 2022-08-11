#! python3

import numpy as np

from mlLib.utils import apply_activation


class LinearParameters:
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
        z = np.einsum("ij,jk", self.w, x)
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
            assert self.db.shape == self.b.shape

        self.dw = np.einsum("ij,kj", dz, x)
        assert self.dw.shape == self.w.shape

    def update(self, learning_rate, lambda_n):
        """
        Parameters:
        -----------
        learning_rate : float
            Default : 0.01

        Returns:
        --------
        None
        """
        dw = self.dw + lambda_n * self.w
        w = self.w - learning_rate * dw
        self.w = w

        if self.bias:
            b = self.b - learning_rate * self.db
            self.b = b


class LogisticRegression:
    def __init__(self, lp_reg):
        """
        Parameters:
        lp_reg : int
            2 : L_2 Regularization is imposed
            1 : L_1 Regularization is imposed
            0 : No regulariation is imposed

        Returns:
        --------
        None
        """
        self.lp_reg = lp_reg

    def predict(self, params, x):
        """
        Parameters:
        -----------
        params : class[LinearParameters]
        x : array_like

        Returns:
        --------
        a : array_like
        dg : array_like
        """
        z = params.forward(x)
        a, dg = apply_activation(z, "sigmoid")
        return a, dg

    def cost_function(self, params, x, y, lambda_=0.01, eps=1e-8):
        """
        Parameters:
        -----------
        params : class[LinearParameters]
        x : array_like
        y : array_like
        lambda_ : float
            Default : 0.01
        eps : float
            Default : 1e-8

        Returns:
        --------
        cost : float
        """
        n = y.shape[1]

        R = np.sum(np.abs(params.w) ** self.lp_reg)
        R *= lambda_ / (2 * n)

        a, _ = self.predict(params, x)
        a = np.clip(a, eps, 1 - eps)

        J = (-1 / n) * (np.sum(y * np.log(a) + (1 - y) * np.log(1 - a)))

        cost = float(np.squeeze(J + R))

        return cost

    def fit(self, x, y, learning_rate=0.1, lambda_=0.01, seed=1, num_iters=10000):
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
        params = LinearParameters(dims, True, seed)

        if self.lp_reg == 0:
            lambda_ = 0.0

        costs = []
        for i in range(num_iters):
            a, _ = self.predict(params, x)
            cost = self.cost_function(params, x, y, lambda_)
            costs.append(cost)
            dz = (a - y) / n
            params.backward(dz, x)
            params.update(learning_rate, lambda_ / n)

            if i % 1000 == 0:
                print(f"Cost after iteration {i}: {cost}")

        return params

    def evaluate(self, params, x):
        """
        Parameters:
        -----------
        params : class[Parameters]
        x : array_like

        Returns:
        --------
        y_hat : array_like
        """
        a, _ = self.predict(params, x)
        y_hat = (~(a < 0.5)).astype(int)

        return y_hat

    def accuracy(self, params, x, y):
        """
        Parameters:
        -----------
        params : class[Parameters]
        x : array_like
        y : array_like

        Returns:
        --------
        accuracy : float
        """
        y_hat = self.evaluate(params, x)

        accuracy = np.sum(y_hat == y) / y.shape[1]

        return accuracy


if __name__ == "__main__":
    from pathlib import Path

    import pandas as pd

    from mlLib.utils import ProcessData

    csv = Path("neuralNetworks/src/python/data/housepricedata.csv")
    df = pd.read_csv(csv)
    dataset = df.values
    x = dataset[:, :10]
    y = dataset[:, 10].reshape(-1, 1)
    data = ProcessData(x, y, 0.15, 0.15, seed=1, feat_as_col=False)

    model = LogisticRegression(2)
    params = model.fit(data.train["x"], data.train["y"], 0.1, 0.01)
    train_acc = model.accuracy(params, data.train["x"], data.train["y"])
    print(f"Training Accuracy: {train_acc}")
    dev_acc = model.accuracy(params, data.dev["x"], data.dev["y"])
    print(f"Dev Accuracy: {dev_acc}")
    test_acc = model.accuracy(params, data.test["x"], data.test["y"])
    print(f"Test Accuracy: {test_acc}")
