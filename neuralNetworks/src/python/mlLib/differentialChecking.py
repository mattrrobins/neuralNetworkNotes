#! python3

## Reverse Differential Checking ##

import numpy as np
from numpy.linalg import norm


def differential_check(f, x, eps=1e-3):
    """
    Parameters:
    -----------
    f : function
    x : array_like
    eps : float
        Default = 10^{-3}

    Returns:
    --------
    error
    """
    y, rf = f(x)
    x = np.array(x)
    if len(x.shape) == 0:
        x = x.reshape(1, 1)
    elif len(x.shape) == 1:
        x = x.reshape(-1, 1)
        # if len(y.shape) == 0:
        # y = y.reshape(1, 1)
        # elif len(y.shape) == 1:
        # y = y.reshape(-1, 1)

    # k, l = y.shape
    m, n = x.shape
    # F = np.zeros((m, n, k, l))
    F = np.zeros((m, n, *y.shape))
    print(F.shape)
    # rf = rf.reshape(m, n, k, l)

    for i in range(m):
        for j in range(n):
            e = np.zeros((m, n))
            e[i, j] = 1
            x_plus = x + eps * e
            x_minus = x - eps * e
            f_plus, _ = f(x_plus)
            f_minus, _ = f(x_minus)
            # f_diff = f_plus.reshape(k, l) - f_minus.reshape(k, l)
            f_diff = f_plus - f_minus
            F[i, j] = f_diff

    F = F / (2 * eps)

    error = norm(F - rf) / (norm(F) + norm(rf))

    return error


def sigmoid(x):
    ## sigmoid: ℝ^n → ℝ^n ##
    # n = 1 is valid
    x = np.array(x)
    sigma = 1 / (1 + np.exp(-x))

    dsigma = np.diagflat(sigma * (1 - sigma))
    rsigma = dsigma.T
    return sigma, rsigma


def foo(x):
    ## f: ℝ^3 → ℝ^2 ##
    ## f(x, y, z) = (xy, z^2)

    y = np.zeros((2, 1))
    y[0] = x[0] * x[1]
    y[1] = x[2] ** 2

    J = np.zeros((2, 3))
    J[0, 0] = x[1]
    J[0, 1] = x[0]
    J[1, 2] = 2 * x[2]

    R = J.T
    return y, R


def bar(x):
    ## f: ℝ^{m×n} → ℝ^m ##
    ## f(x) = x@v
    np.random.seed(1)
    m, n = x.shape
    v = np.random.randn(n)
    f = np.einsum("ij, j", x, v)

    J = np.zeros((m, m, n))
    for mu in range(m):
        for i in range(m):
            for j in range(n):
                if mu == i:
                    J[mu, i, j] = v[j]

    R = np.einsum("kij->ijk", J)
    return f, R


def baz(x):
    ## f: ℝ^{m×n} → ℝ^{m×n} ##
    ## f(x) = x * x  # The Hadmard square
    m, n = x.shape
    f = np.einsum("ij,ij->ij", x, x)

    J = np.zeros((m, n, m, n))
    for mu in range(m):
        for nu in range(n):
            for i in range(m):
                for j in range(n):
                    if (mu == i) and (nu == j):
                        J[mu, nu, i, j] = 2 * x[i, j]

    R = np.einsum("ijkl->klij", J)
    return f, R


if __name__ == "__main__":

    def differential_check_01(f, x, eps=1e-3):
        """
        Parameters:
        -----------
        f : function
        x : array_like
        eps : float
            Default = 10^{-3}

        Returns:
        --------
        error
        """
        y, rf = f(x)
        x = np.array(x)
        if len(x.shape) == 0:
            x = x.reshape(1, 1)
        elif len(x.shape) == 1:
            x = x.reshape(-1, 1)
        if len(y.shape) == 0:
            y = y.reshape(1, 1)
        elif len(y.shape) == 1:
            y = y.reshape(-1, 1)

        k, l = y.shape
        m, n = x.shape
        F = np.zeros((m, n, k, l))
        rf = rf.reshape(m, n, k, l)

        for mu in range(k):
            for nu in range(l):
                for i in range(m):
                    for j in range(n):
                        e = np.zeros((m, n))
                        e[i, j] = 1
                        x_plus = x + eps * e
                        x_minus = x - eps * e
                        f_plus, _ = f(x_plus)
                        f_minus, _ = f(x_minus)
                        f_diff = f_plus.reshape(k, l) - f_minus.reshape(k, l)
                        F[i, j, mu, nu] = f_diff[mu, nu]

        F = F / (2 * eps)

        error = norm(F - rf) / (norm(F) + norm(rf))

        return error

    def fctn(x):
        x = np.array(x)
        f = np.exp(x)
        assert f.shape == x.shape
        return f, np.diagflat(f)

    m = 3
    n = 4
    np.random.seed(1)
    for _ in range(5):
        x = np.random.rand(m) * 10
        # x = np.random.randint(1, 100)
        print(x)
        # e = differential_check_euclidean(foo, x)
        # e = differential_check_matrix_to_euclidean(bar, x)
        e = differential_check(sigmoid, x)
        print(e)
