import cvxpy as cp
import numpy as np
import scipy.sparse as sp

def create(m, n, density=0.1):
    np.random.seed(0)

    mu = np.exp(0.01*np.random.randn(n))-1  # returns
    D = np.random.rand(n)/10;               # idiosyncratic risk
    F = sp.rand(n,m,density)                # factor model
    F.data = np.random.randn(len(F.data))/10
    gamma = 1
    B = 1

    x = cp.Variable(n)
    f = mu.T*x - gamma*(cp.sum_squares(F.T.dot(x)) +
                        cp.sum_squares(cp.mul_elemwise(D, x)))
    C = [cp.sum_entries(x) == B,
         x >= 0]

    return cp.Problem(cp.Maximize(f), C)
