#! python3

import numpy as np


def foo_scalar(x):
    f = x * x
    df = 2 * x

    return f, df


def foo_vector(x):
    f = x * x
    n = x.size
    df = np.zeros((n, n))
    for mu in range(n):
        for i in range(n):
            if mu == i:
                df[mu, i] = 2 * x[i]

    return f, df


def foo_matrix(x):
    f = x * x
    m, n = x.shape
    df = np.zeros((m, n, m, n))
    for mu in range(m):
        for nu in range(n):
            for i in range(m):
                for j in range(n):
                    if (mu == i) and (nu == j):
                        df[mu, nu, i, j] = 2 * x[i, j]

    return f, df


if __name__ == "__main__":
    np.random.seed(1)
    for i in range(3):
        print(f"Iteration {i}")

        x = np.random.randn(1)
        f, df = foo_scalar(x)
        print(f)
        print(df)

        x = np.random.randn(5)
        f, df = foo_vector(x)
        print(f)
        print(df)

        x = np.random.randn(2, 3)
        f, df = foo_matrix(x)
        print(f)
        print(df)
