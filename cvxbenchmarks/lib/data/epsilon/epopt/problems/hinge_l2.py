"""Standard SVM, i.e.. hinge loss w/ l2 regularization."""

from epopt.problems import problem_util
import cvxpy as cp
import epopt as ep
import numpy as np
import scipy.sparse as sp

def create(**kwargs):
    A, b = problem_util.create_classification(**kwargs)
    lam = 1

    x = cp.Variable(A.shape[1])
    f = ep.hinge_loss(x, A, b) + lam*cp.sum_squares(x)
    return cp.Problem(cp.Minimize(f))
