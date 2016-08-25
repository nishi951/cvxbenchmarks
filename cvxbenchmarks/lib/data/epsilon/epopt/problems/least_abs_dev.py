
import cvxpy as cp
import numpy as np
import scipy.sparse as sp

def create(m, n):
    np.random.seed(0)
    A = np.random.randn(m,n);
    A = A*sp.diags([1 / np.sqrt(np.sum(A**2, 0))], [0])
    b = A.dot(10*np.random.randn(n))

    k = max(m/50, 1)
    idx = np.random.randint(0, m, k)
    b[idx] += 100*np.random.randn(k)

    x = cp.Variable(n)
    return cp.Problem(cp.Minimize(cp.norm1(A*x - b)))
