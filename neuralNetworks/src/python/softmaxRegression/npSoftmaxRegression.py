#! python3

import numpy as np

from mlLib.utils import LinearParameters, EpochRuntime, ShuffleBatchData

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


## The softmax function and differential
def softmax(z):
    """
    Parameters:
    -----------
    z : array_like

    Returns:
    --------
    y : array_like
        y.shape == z.shape
    dy : array_like
    """
    n = z.shape[0]
    u = np.exp(z - np.max(z, axis=0))
    u_sum = np.sum(u, axis=0, keepdims=True)
    y = u / u_sum

    dy = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i == j:
                dy[i, j] = y[i, 0] * (1 - y[j, 0])
            else:
                dy[i, j] = -y[i, 0] * y[j, 0]

    return y, dy


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
    loss : float
    r_loss : array_like
    rloss.shape == a.shape
    """
    assert a.shape == y.shape, "a and y have different shapes"

    a = np.clip(a, eps, 1 - eps)
    loss = -1 * np.sum(y * np.log(a), axis=0)
    rloss = -1 * y / a
    return loss, rloss


class SoftmaxRegression:
    def __init__(self, num_features, num_labels, bias=True, seed=1101):
        """
        Parameters:
        -----------

        Returns:
        --------
        None
        """
        self.n = num_features
        self.C = num_labels
        self.bias = bias

        self.params = LinearParameters((self.C, self.n), self.bias, seed)

    def forward(self, x):
        """
        Parameters:
        -----------
        x : array_like

        Returns:
        --------
        a : array_like
        da : array_like
        """
        z = self.params.forward(x)
        a, da = softmax(z)
        return a, da

    def cost_function(self, a, y, lambda_):
        """
        Parameters:
        -----------
        a : array_like
        y : array_like
        lambda_ : float

        Returns:
        cost : float
        """
        N = y.shape[1]
        loss, rloss = cross_entropy(a, y)

        R = 0
        R += np.linalg.norm(self.params.w) ** 2
        R *= lambda_ / (2 * N)

        J = np.sum(loss) / N
        cost = float(np.squeeze(J + R))

        rcost = np.einsum("ij->i", rloss) / N
        rcost = rcost.reshape(-1, 1)
        return cost, rcost

    def update_parameters(self, learning_rate, lambda_N):
        """
        Parameters:
        -----------
        learning_rate : float

        Returns:
        None
        """
        dw = self.params.dw
        dw += lambda_N * self.params.w
        self.params.dw = dw
        self.params.update(learning_rate)

    def fit(self, data, learning_rate, lambda_, batch_size, epochs, print_cost_epoch):
        """
        Parameters:
        -----------
        data : Dict[array_like]
            data['x'] : array_like
            data['y'] : array_like
        learning_rate : float
        lambda_ : float
        epochs : int
        print_cost_epoch : int

        Returns:
        costs : List[float]
        """
        batching = ShuffleBatchData(data, batch_size)

        costs = []
        # time = EpochRuntime()
        for epoch in range(epochs):
            batches = batching.get_batches()
            cost = 0
            for batch in batches:
                x = batch["x"]
                y = batch["y"]
                (
                    a,
                    ra,
                ) = self.forward(x)
                batch_cost, rcost = self.cost_function(a, y, lambda_)
                cost += x.shape[1] * batch_cost
                delta = ra @ rcost
                _ = self.params.backward(delta, x)
                lambda_N = lambda_ / x.shape[1]
                self.update_parameters(learning_rate, lambda_N)
            cost /= data["x"].shape[1]
            costs.append(cost)
            # time.elapsed_time()

            if (print_cost_epoch != 0) and (epoch % print_cost_epoch == 0):
                print(f"Cost after epoch {epoch}: {cost}")

        return costs

    def accuracy(self, data):
        """
        Parameters:
        -----------
        data : Dict[array_like]
            data['x'] : array_like
            data['y'] : array_like

        Returns:
        --------
        acc : float
        """
        x = data["x"]
        y = data["y"]
        a, _ = self.forward(x)
        yhat = np.argmax(a, axis=0)
        y = decode_labels(y)
        y = y.reshape(yhat.shape)
        acc = np.sum(yhat == y) / y.size
        return acc


if __name__ == "__main__":
    from mlLib.utils import ProcessData

    from scipy.io import loadmat
    from pathlib import Path

    """
    data = loadmat(Path("neuralNetworks/src/python/data/softmaxData.mat"))
    x, y = data["X"], data["y"].ravel()
    y[y == 10] = 0
    N = y.size
    y = y.reshape(-1, N)
    C = 10
    n = x.shape[1]
    y = encode_labels(y, C)
    y = y.reshape(N, -1)
    print(x.shape)
    print(y.shape)
    """

    from mlxtend.data import iris_data

    x, y = iris_data()
    N = y.size
    C = 3
    n = x.shape[1]
    y = encode_labels(y, C).T

    data = ProcessData(x, y, 0.05, 0.05, feat_as_col=False)

    model = SoftmaxRegression(n, C, True)
    costs = model.fit(data.train, 0.05, 0.8, 2**5, 500, 50)

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
