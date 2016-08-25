
import cvxpy as cp
import numpy as np
import scipy.sparse as sp
import scipy.linalg as la
from epopt.problems import problem_util

def create(m, n):
    mu = 1
    rho = 1
    sigma = 0.1

    A = problem_util.normalized_data_matrix(m, n, mu)
    x0 = sp.rand(n, 1, rho)
    x0.data = np.random.randn(x0.nnz)
    x0 = x0.toarray().ravel()

    b = np.sign(A.dot(x0) + sigma*np.random.randn(m))
    A[b>0,:] += 0.7*np.tile([x0], (np.sum(b>0),1))
    A[b<0,:] -= 0.7*np.tile([x0], (np.sum(b<0),1))

    P = la.block_diag(np.random.randn(n-1,n-1), 0)

    lam = 1
    x = cp.Variable(A.shape[1])

    # Straightforward formulation w/ no constraints
    # TODO(mwytock): Fix compiler so this works
    z0 = 1 - sp.diags([b],[0])*A*x + cp.norm1(P.T*x)
    f_eval = lambda: (lam*cp.sum_squares(x) + cp.sum_entries(cp.max_elemwise(z0, 0))).value

    # Explicit epigraph constraint
    t = cp.Variable(1)
    z = 1 - sp.diags([b],[0])*A*x + t
    f = lam*cp.sum_squares(x) + cp.sum_entries(cp.max_elemwise(z, 0))
    C = [cp.norm1(P.T*x) <= t]
    return cp.Problem(cp.Minimize(f), C), f_eval
