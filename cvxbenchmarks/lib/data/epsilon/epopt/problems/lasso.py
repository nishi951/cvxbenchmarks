
import cvxpy as cp
import numpy as np
import numpy.linalg as LA
import scipy.sparse as sp
from epopt.problems import problem_util

def create(**kwargs):
    A, B = problem_util.create_regression(**kwargs)
    lambda_max = np.abs(A.T.dot(B)).max()
    lam = 0.5*lambda_max

    X = cp.Variable(A.shape[1], B.shape[1] if len(B.shape) > 1 else 1)
    f = cp.sum_squares(A*X - B) + lam*cp.norm1(X)
    return cp.Problem(cp.Minimize(f))
