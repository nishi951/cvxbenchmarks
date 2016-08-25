import cvxpy as cp
import numpy as np
import scipy.sparse as sp

def create(m, n, lam):
    np.random.seed(0)

    m = int(n)
    n = int(n)
    lam = float(lam)

    A = sp.rand(n,n, 0.01)
    A = np.asarray(A.T.dot(A).todense() + 0.1*np.eye(n))
    L = np.linalg.cholesky(np.linalg.inv(A))
    X = np.random.randn(m,n).dot(L.T)
    S = X.T.dot(X)/m
    W = np.ones((n,n)) - np.eye(n)

    Theta = cp.Variable(n,n)
    return cp.Problem(cp.Minimize(
        lam*cp.norm1(cp.mul_elemwise(W,Theta)) +
        cp.sum_entries(cp.mul_elemwise(S,Theta)) -
        cp.log_det(Theta)))
