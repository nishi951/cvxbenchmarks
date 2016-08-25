from epopt.problems import problem_util
import cvxpy as cp
import epopt as ep
import numpy as np
import scipy.sparse as sp

def create(**kwargs):
    m = kwargs["m"]
    n = kwargs["n"]
    k = 10
    A = [problem_util.normalized_data_matrix(m,n,1) for i in range(k)]
    B = problem_util.normalized_data_matrix(k,n,1)
    c = np.random.rand(k)

    x = cp.Variable(n)
    t = cp.Variable(k)
    f = cp.max_entries(t+cp.abs(B*x-c))
    C = []
    for i in range(k):
        C.append(cp.pnorm(A[i]*x, 2) <= t[i])

	t_eval = lambda: np.array([cp.pnorm(A[i]*x, 2).value for i in range(k)])
    f_eval = lambda: cp.max_entries(t_eval() + cp.abs(B*x-c)).value

    return cp.Problem(cp.Minimize(f), C), f_eval
