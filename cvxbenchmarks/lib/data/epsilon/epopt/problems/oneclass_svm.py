
from epopt.problems import problem_util
import cvxpy as cp
import epopt as ep
import numpy as np
import scipy.sparse as sp

def create(m, n):
    # Generate random points uniform over hypersphere
    A = np.random.randn(m, n)
    A /= np.sqrt(np.sum(A**2, axis=1))[:,np.newaxis]
    A *= (np.random.rand(m)**(1./n))[:,np.newaxis]

    # Shift points and add some outliers
    # NOTE(mwytock): causes problems for operator splitting, should fix
    #x0 = np.random.randn(n)
    x0 = np.zeros(n)
    A += x0

    k = max(m/50, 1)
    idx = np.random.randint(0, m, k)
    A[idx, :] += np.random.randn(k, n)
    lam = 1
    x = cp.Variable(n)
    rho = cp.Variable(1)

    # Straightforward expression
    #z = np.sum(A**2, axis=1) - 2*A*x + cp.sum_squares(x)  # z_i = ||a_i - x||^2
    #f = cp.sum_entries(cp.max_elemwise(z - rho, 0)) + lam*cp.max_elemwise(0, rho)

    # Explicit epigraph form
    t = cp.Variable(1)
    z = np.sum(A**2, axis=1) - 2*A*x + t  # z_i = ||a_i - x||^2
    z0 = np.sum(A**2, axis=1) - 2*A*x + cp.sum_squares(x)
    f_eval = lambda: ((1./n)*cp.sum_entries(cp.max_elemwise(z0 - rho, 0)) + cp.max_elemwise(0, rho)).value
    f = (1./n)*cp.sum_entries(cp.max_elemwise(z-rho, 0)) + lam*cp.sum_entries(cp.max_elemwise(rho, 0))
    C = [cp.sum_squares(x) <= t]

    return cp.Problem(cp.Minimize(f), C), f_eval
