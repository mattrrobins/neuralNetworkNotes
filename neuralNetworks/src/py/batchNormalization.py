#! python3
import copy

import numpy as np

import mlLib.npActivators as npActivators
from mlLib.npActivators import ACTIVATORS

class LinearParams():
    def __init__(self, dims, bias=True, seed=1):
        np.random.seed(seed)
        self.dims = dims
        self.bias = bias
        self.w = np.random.randn(dims[0], dims[1]) * 0.01
        if bias:
            self.b = np.zeros((dims[0], 1))

    def forward(self, a):
        """
        """
        z = np.einsum('ij,jk', self.w, a)
        if self.bias:
            z = z + self.b
        return z

    def backward(self, d_in, a_prev):
        """
        """
        if self.bias:
            self.db = np.sum(d_in, axis=1, keepdims=True)
            assert (self.db.shape == self.b.shape), "db has the wrong shape"

        self.dw = np.einsum('ij,kj', d_in, a_prev)
        assert (self.dw.shape == self.w.shape), "dw has the wrong shape"

        d_out = np.einsum('ij,ik', self.w, d_in)
        assert (d_out.shape == (self.w.shape[1], d_in.shape[1])), "d_out has the wrong shape"
        return d_out

    def update(self, learning_rate=0.01):
        """
        """
        w = self.w - learning_rate * self.dw
        if self.bias:
            b = copy.deepcopy(self.b)
            b = self.b - learning_rate * self.db

        self.w = w
        if self.bias:
            self.b = b

class BatchNormParams():
    def __init__(self, dims, eps=1e-8, seed=1):
        np.random.seed(seed)
        self.dims = dims
        self.epsilon = eps

        self.gamma = np.ones((dims[0], 1))
        self.beta = np.zeros((dims[0], 1))

        self.running_mean = np.zeros((dims[0], 1))
        self.running_var = np.zeros((dims[0], 1))

    def forward(self, x, *iter):
        """
        """
        self.mu = np.mean(x, axis=1, keepdims=True)
        self.var = np.var(x, axis=1, keepdims=True)
        self.theta = 1 / np.sqrt(self.var + self.epsilon)

        if len(iter) == 1:
            momentum = iter[0] / (iter[0] + 1)
            self.running_mean = momentum * self.running_mean + (1 - momentum) * self.mu
            self.running_var = momentum * self.running_var + (1 - momentum) * self.var

        self.norm = self.theta * (x - self.mu)
        assert (self.norm.shape == x.shape), "Normalization resulted in wrong dims"

        return self.gamma * self.norm + self.beta

    def backward(self, d_in):
        """
        """
        self.dbeta = np.sum(d_in, axis=1, keepdims=True)
        assert (self.dbeta.shape == self.beta.shape)

        aux = self.norm * d_in
        self.dgamma = np.sum(aux, axis=1, keepdims=True)
        assert (self.dgamma.shape == self.gamma.shape)

        ## Compute differential of normalization
        m, n = self.dims
        I_m = np.identity(m)
        I_n = np.identity(n)

        n_1 = np.ones((n, 1))

        v = np.einsum('ij,kl->ijkl', self.norm, self.norm)
        v = np.einsum('ijil->ijl', v)
        v = (1 + v) / n
        v = np.einsum('klj,pq->kljpq', v, I_m)
        v = np.einsum('kljki->jilk', v)

        I = np.einsum('ki,jl->jilk', I_m, I_n)

        rN = self.theta.T * (I - v)

        d_in_gamma = (self.gamma @ n_1.T) * d_in

        d_out = np.einsum('jilk,ij->kl', rN, d_in_gamma)
        assert (d_out.shape == (m, n)), "Batch Norm differential is incorrect"

        return d_out

    def update(self, learning_rate=0.01):
        gamma = self.gamma - learning_rate * self.dgamma
        beta = self.beta - learning_rate * self.dbeta

        self.gamma = gamma
        self.beta = beta

def apply_activation(z, activator):
    """
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

class Network():
    def __init__(self, config):
        self.config = config
        self.nodes = config['nodes']
        self.activators = config['activators']
        self.bias = config['bias']
        self.bn = config['batch_norm']
        self.lp = config['lp_regularization']
        self.L = len(config['nodes']) - 1

    def forward_propagation(self, x, lin_params, bn_params, *iter):
        """
        """
        z = {}
        a = {}
        dg = {}


        L = len(lin_params)
        nodes = self.config['nodes']
        activators = self.config['activators']

        a[0], dg[0] = apply_activation(x, activators[0])

        for l in range(1, L + 1):
            u = lin_params[l].forward(a[l - 1])
            if l in bn_params.keys():
                u = bn_params[l].forward(u, *iter)
            out, d_out = apply_activation(u, activators[l])
            z[l] = u
            a[l] = out
            dg[l] = d_out

        cache = {'z' : z, 'a' : a, 'dg' : dg}

        return cache

    def compute_cost(self, y, cache, lin_params, lambda_=0.0, eps=1e-8):
        """
        """
        n = y.shape[1]
        a = cache['a'][self.L]
        J = (-1 / n) * (np.sum(y * np.log(a + eps)) + np.sum((1 - y) * np.log(1 - a + eps)))

        # Compute regularization term
        R = 0
        for param in lin_params.values():
            R += np.sum(np.abs(param.w) ** self.lp)
        R *= (lambda_ / (2 * n))

        # Compute total cost
        cost = float(np.squeeze(J + R))
        return cost

    def backward_propagation(self, y, lin_params, bn_params, cache):
        """
        """
        z = cache['z']
        a = cache['a']
        dg = cache['dg']

        n = y.shape[1]

        delta = {}
        delta[self.L] = ((a[self.L] - y) / n)

        for l in reversed(range(1, self.L + 1)):
            if l in bn_params.keys():
                du = bn_params[l].backward(delta[l])
            else:
                du = delta[l]
            delta_temp = lin_params[l].backward(du, a[l - 1])
            delta[l - 1] = dg[l - 1] * delta_temp

    def update_parameters(self, linear_params, bn_params, learning_rate=0.01):
        """
        """
        for l in range(1, self.L + 1):
            linear_params[l].update(learning_rate)
            if l in bn_params.keys():
                bn_params[l].update(learning_rate)

    def train(self, x, y, learning_rate=0.01, num_iters=100, lambda_=0.0, seed=1):
        """
        """
        n = x.shape[1]
        # Initialize parameters
        lin_params = {}
        bn_params = {}

        for l in range(1, self.L + 1):
            if self.bn[l]:
                bn_params[l] = BatchNormParams((self.nodes[l], n), 1e-8, seed)
            dims = (self.nodes[l], self.nodes[l - 1])
            lin_params[l] = LinearParams(dims, self.bias[l], seed)

        # Gradient descent loop
        costs = []
        for i in range(num_iters):
            cache = self.forward_propagation(x, lin_params, bn_params, i)
            cost = self.compute_cost(y, cache, lin_params, lambda_)
            costs.append(cost)
            self.backward_propagation(y, lin_params, bn_params, cache)
            self.update_parameters(lin_params, bn_params, learning_rate)

            if i % 1000 == 0:
                print(f'Cost after iteration {i}: {cost}')

        return lin_params, bn_params, costs








def main():
    config = {
     "nodes" : [10, 32, 32, 1],
     "activators" : ["linear", "relu", "relu", "sigmoid"],
     "bias" : [False, False, False, True],
     "batch_norm" : [False, True, True, False],
     "lp_regularization" : 2
     }

    x = np.random.rand(10, 1000)
    y = np.random.randint(0, 2, size=(1, 1000))

    model = Network(config)
    lin_params, bn_params, costs = model.train(x, y)
    pprint(costs)




if __name__ == '__main__':
    main()
