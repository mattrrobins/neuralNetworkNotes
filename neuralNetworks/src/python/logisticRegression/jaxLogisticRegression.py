#! python3

"""
This needs work.  Likely on understanding the JAX implementation.
The cost function leads to NaN's quickly and never recovers.
I thought it was the implementation of the activators causing this, but JAX's own 
implementation doesn't resolve the issue.  Will return at a later date.
"""


from functools import partial

import jax.numpy as jnp
from jax import random, jit, value_and_grad
from jax.nn import sigmoid

class LogisticRegression():
    def __init__(self, lp_reg):
        """
        """
        self.lp_reg = lp_reg

    def init_parameters(self, num_feats, seed=1, scale=0.01):
        """
        """
        key = random.PRNGKey(seed)
        key_w, key_b = random.split(key)
        w = random.normal(key_w, (1, num_feats)) * scale
        b = random.normal(key_b, (1, 1)) * scale
        params = {'w' : w, 'b' : b}
        return params

    @partial(jit, static_argnums=(0,))
    def predict(self, params, x):
        """
        """
        z = jnp.einsum('ij,jk', params['w'], x)
        z += params['b']
        a = sigmoid(z)
        return a

    @partial(jit, static_argnums=(0,), inline=True)
    def cost_function(self, params, x, y, lambda_=0.01, eps=1e-14):
        """
        """
        n = y.shape[1]

        # Compute penalty term
        R = jnp.sum(jnp.abs(params['w']) ** self.lp_reg)
        R *= (lambda_ / (2 * n))

        # Compute unregularized term
        a = self.predict(params, x)
        a = jnp.clip(a, eps, 1 - eps)
        J = (-1 / n) * jnp.sum(y * jnp.log(a) + (1 - y) * jnp.log(1 - a))

        # Compute regularized cost
        cost = J + R
        return cost

    def update_parameters(self, params, grads, learning_rate=0.1):
        """
        """
        params['w'] -= learning_rate * grads['w']
        params['b'] -= learning_rate * grads['b']

        return params

    def fit(self, x, y, learning_rate=0.1, lambda_=0.01, seed=1, num_iters=10000):
        """
        """
        m, n = x.shape
        # Initialize parameters
        params = self.init_parameters(m, seed, 0.01)

        if self.lp_reg == 0:
            lambda_ = 0.0

        # Main training loop
        costs = []
        for i in range(num_iters):
            cost, grads = value_and_grad(self.cost_function)(params, x, y, lambda_)
            cost = float(jnp.squeeze(cost))
            costs.append(cost)
            params = self.update_parameters(params, grads, learning_rate)

            if i % 1000 == 0:
                print(f'Cost after iteration {i}: {cost}')

        return params

    def evaluate(self, params, x):
        """
        Parameters:
        -----------
        params : Dict[array_like]
        x : array_like

        Returns:
        --------
        y_hat : array_like
        """
        a= self.predict(params, x)
        y_hat = (~(a < 0.5)).astype(int)

        return y_hat

    def accuracy(self, params, x, y):
        """
        Parameters:
        -----------
        params : Dict[array_like]
        x : array_like
        y : array_like

        Returns:
        --------
        accuracy : float
        """
        y_hat = self.evaluate(params, x)
        
        accuracy = jnp.sum(y_hat == y) / y.shape[1]

        return accuracy


if __name__ == '__main__':
    from pathlib import Path

    import pandas as pd

    from mlLib.utils import ProcessData

    csv = Path('neuralNetworks/src/python/data/housepricedata.csv')
    df = pd.read_csv(csv)
    dataset = df.values
    x = dataset[:, :10]
    y = dataset[:, 10].reshape(-1, 1)
    data = ProcessData(x, y, 0.15, 0.15, seed=1, feat_as_col=False)

    model = LogisticRegression(2)
    params = model.fit(data.train['x'], data.train['y'], 0.1, 0.01)
    train_acc = model.accuracy(params, data.train['x'], data.train['y'])
    print(f'Training Accuracy: {train_acc}')
    dev_acc = model.accuracy(params, data.dev['x'][0], data.dev['y'][0])
    print(f'Dev Accuracy: {dev_acc}')
    test_acc = model.accuracy(params, data.test['x'], data.test['y'])
    print(f'Test Accuracy: {test_acc}')
