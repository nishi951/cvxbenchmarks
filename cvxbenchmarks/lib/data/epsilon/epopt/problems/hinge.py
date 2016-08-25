
from epopt.problems import problem_util
import cvxpy as cp
import epopt as ep
import numpy as np
import scipy.sparse as sp

def create(**kwargs):
    n = kwargs["n"]

    x = cp.Variable(n)
    u = np.random.rand(n)
    f =  cp.sum_squares(x-u)+cp.sum_entries(cp.max_elemwise(x, 0))
    return cp.Problem(cp.Minimize(f))
