
import cvxpy as cp
import numpy as np

def create(n):
    np.random.seed(0)
    k = max(int(np.sqrt(n)/2), 1)

    x0 = np.ones((n,1))
    idxs = np.random.randint(0, n, (k,2))
    idxs.sort()
    for a, b in idxs:
        x0[a:b] += 10*(np.random.rand()-0.5)
    b = x0 + np.random.randn(n, 1)

    lam = np.sqrt(n)
    x = cp.Variable(n)
    f = 0.5*cp.sum_squares(x-b) + lam*cp.norm1(x[1:]-x[:-1])

    return cp.Problem(cp.Minimize(f))
