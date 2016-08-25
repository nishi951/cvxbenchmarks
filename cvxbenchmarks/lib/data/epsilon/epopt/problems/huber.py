
import cvxpy as cp
import numpy as np
import scipy.sparse as sp

def create(m, n):
    np.random.seed(0)

    x0 = np.random.randn(n)
    A = np.random.randn(m,n)
    A = A*sp.diags([1 / np.sqrt(np.sum(A**2, 0))], [0])
    b = A.dot(x0) + np.sqrt(0.01)*np.random.randn(m)
    b = b + 10*np.asarray(sp.rand(m, 1, 0.05).todense()).ravel()

    x = cp.Variable(n)
    return cp.Problem(cp.Minimize(cp.sum_entries(cp.huber(A*x - b))))
