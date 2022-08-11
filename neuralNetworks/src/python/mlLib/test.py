#! python3

import numpy as np

from mlLib.npLossFunctions import lse
from mlLib.utils import ShuffleBatchData


class LinearRegression:
    def __init__(self):
        """ """

    def cost_function(self, a, y, lambda_):
        """ """
        loss, rloss = lse(a, y)

        R = np.linalg.norm(self.w) ** 2
        R *= lambda_ / (2 * y.shape[1])

        J = np.mean(loss, axis=1)

        cost = float(np.squeeze(J + R))
        rcost = rloss / y.shape[1]

        return cost, rcost

    def update(
        self,
        learning_rate,
    ):
        """ """
        self.w -= learning_rate * self.dw
        self.b -= learning_rate * self.db

    def fit(self, data, batch_size, learning_rate, lambda_, epochs, print_cost_epoch):
        """ """
        x = data["x"]
        y = data["y"]

        self.w = np.random.randn(y.shape[0], x.shape[0]) * 0.01
        self.b = np.zeros((y.shape[0], 1))

        batching = ShuffleBatchData(data, batch_size)

        N = x.shape[1]
        costs = []
        for epoch in range(epochs):
            batches = batching.get_batches()
            B = len(batches)
            k = 1
            cost = 0
            for batch in batches:
                x = batch["x"]
                y = batch["y"]
                n = x.shape[1]
                a = self.w @ x + self.b
                batch_cost, rcost = self.cost_function(a, y, lambda_)
                lambda_n = lambda_ / n
                cost += n * batch_cost
                self.dw = rcost @ x.T / n + lambda_n * self.w
                self.db = np.mean(rcost, axis=1, keepdims=True)
                self.update(learning_rate)
                k += 1
            costs.append(cost)

            if (print_cost_epoch != 0) and (epoch % print_cost_epoch == 0):
                print(f"Cost after epoch {epoch}: {cost}")

        return costs

    def evaluate(self, x):
        """ """
        return self.w @ x + self.b

    def accuracy(self, data):
        """ """
        x = data["x"]
        y = data["y"]
        a = self.evaluate(x)
        rs = np.mean(lse(a, y))
        var = np.var(y, axis=1, keepdims=True)
        r2 = float(np.squeeze(1 - (rs / var)))
        return r2


if __name__ == "__main__":

    import pandas as pd

    from mlLib.utils import ProcessData

    df = pd.read_csv("neuralNetworks/src/python/data/heart.data.csv")
    data = df.values
    x = data[:, :-1]
    y = data[:, -1].reshape(-1, 1)
    data = ProcessData(x, y, 0.05, 0.05, feat_as_col=False)

    model = LinearRegression()
    costs = model.fit(data.train, 2**5, 0.01, 0.1, 500, 50)

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

    x = data.test["x"]
    y = data.test["y"]
    print(model.evaluate(x))
    print(y)
