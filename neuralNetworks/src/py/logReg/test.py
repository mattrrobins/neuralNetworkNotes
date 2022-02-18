#! python3

from pprint import pprint

import numpy as np

from logRegCostFctn import *

def main():
    x = np.random.rand(2, 10)
    y = np.random.randint(2, size=(1, 10))
    w = np.random.rand(2, 1)
    b = np.random.rand(1)

    J, dw, db = cost_function(x, y, w, b)
    print(J)
    pprint(dw)
    pprint(db)

    J_list, w_opt, b_opt = grad_descent(x, y, w, b, 0.001, 50)
    pprint(J_list)
    pprint(dw)
    pprint(db)


if __name__ == '__main__':
    main()
