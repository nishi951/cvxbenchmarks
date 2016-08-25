
import cvxpy as cp
import numpy as np
import scipy.sparse as sp

def hinge(x):
    return cp.sum_entries(cp.max_elemwise(x,0))

def normalized_data_matrix(m, n, mu):
    if mu == 1:
        # dense
        A = np.random.randn(m, n)
        A /= np.sqrt(np.sum(A**2, 0))
    else:
        # sparse
        A = sp.rand(m, n, mu)
        A.data = np.random.randn(A.nnz)
        N = A.copy()
        N.data = N.data**2
        A = A*sp.diags([1 / np.sqrt(np.ravel(N.sum(axis=0)))], [0])

    return A

def create_regression(m, n, k=1, rho=1, mu=1, sigma=0.05):
    """Create a random (multivariate) regression problem."""

    A = normalized_data_matrix(m, n, mu)
    X0 = sp.rand(n, k, rho)
    X0.data = np.random.randn(X0.nnz)

    if k == 1:
        x0 = sp.rand(n, 1, rho)
        x0.data = np.random.randn(x0.nnz)
        x0 = x0.toarray().ravel()
        b = A.dot(x0) + sigma*np.random.randn(m)
        return A, b
    else:
        X0 = sp.rand(n, k, rho)
        X0.data = np.random.randn(X0.nnz)
        X0 = X0.toarray()
        B = A.dot(X0) + sigma*np.random.randn(m,k)
        return A, B

def create_classification(m, n, rho=1, mu=1, sigma=0.05):
    """Create a random classification problem."""
    A = normalized_data_matrix(m, n, mu)
    x0 = sp.rand(n, 1, rho)
    x0.data = np.random.randn(x0.nnz)
    x0 = x0.toarray().ravel()

    b = np.sign(A.dot(x0) + sigma*np.random.randn(m))
    return A, b
