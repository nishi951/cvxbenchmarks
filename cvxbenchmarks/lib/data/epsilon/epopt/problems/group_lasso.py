
import cvxpy as cp
import numpy as np
import numpy.linalg as LA
import scipy.sparse as sp
from epopt.problems import problem_util

def create(m, ni, K):
    np.random.seed(0)

    part = np.random.randint(1, ni, K)
    n = np.sum(part)
    p = 0.2

    # Each part is pa[i]:pb[i]
    pb = np.cumsum(part)
    pa = np.hstack((0, pb[:-1]))

    x0 = np.zeros(n)
    for i in xrange(K):
        if np.random.rand() < p:
            x0[pa[i]:pb[i]] = np.random.randn(part[i])

    A = problem_util.normalized_data_matrix(m, n, 1)
    b = A.dot(x0) + np.sqrt(0.001)*np.random.randn(m)
    lam = 0.1*max(LA.norm(A[:,pa[i]:pb[i]].T.dot(b)) for i in xrange(K))

    x = cp.Variable(n)
    f = (0.5*cp.sum_squares(A*x - b) +
         lam*sum(cp.norm2(x[pa[i]:pb[i]]) for i in xrange(K)))
    return cp.Problem(cp.Minimize(f))
