import cvxpy as cp
import numpy as np
import scipy.sparse as sp

def create(n, r=10, density=0.1):
    np.random.seed(0)

    L1 = np.random.randn(n,r)
    L2 = np.random.randn(r,n)
    L0 = L1.dot(L2)

    S0 = sp.rand(n, n, density)
    S0.data = 10*np.random.randn(len(S0.data))
    M = L0 + S0
    lam = 0.1

    L = cp.Variable(n, n)
    S = cp.Variable(n, n)
    f = cp.norm(L, "nuc") + lam*cp.norm1(S)
    C = [L + S == M]

    return cp.Problem(cp.Minimize(f), C)
