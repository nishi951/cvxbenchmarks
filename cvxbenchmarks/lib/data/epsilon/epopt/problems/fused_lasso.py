
import cvxpy as cp
import numpy as np
import numpy.linalg as LA
import scipy.sparse as sp
from epopt.problems import problem_util

def create(m, ni, k, rho=0.05, sigma=0.05):
    A = np.random.randn(m, ni*k)
    A /= np.sqrt(np.sum(A**2, 0))

    x0 = np.zeros(ni*k)
    for i in range(k):
        if np.random.rand() < rho:
            x0[i*ni:(i+1)*ni] = np.random.rand()
    b = A.dot(x0) + sigma*np.random.randn(m)

    lam = 0.1*sigma*np.sqrt(m*np.log(ni*k))
    x = cp.Variable(A.shape[1])
    f = cp.sum_squares(A*x - b) + lam*cp.norm1(x) + lam*cp.tv(x)
    return cp.Problem(cp.Minimize(f))
