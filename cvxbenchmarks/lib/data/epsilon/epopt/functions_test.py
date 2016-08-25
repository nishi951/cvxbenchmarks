
import logging

import epopt as ep
import cvxpy as cp
import numpy as np

from cvxpy.settings import OPTIMAL
from nose.tools import assert_equal, assert_less

m = 10
n = 5
k = 3
theta = cp.Variable(n)
Theta = cp.Variable(n,k)
alphas = np.linspace(1./(k+1), 1-1./(k+1), k)

np.random.seed(0)
X = np.random.randn(m,n)
y_binary = np.random.randint(2, size=(m,))*2-1
y_multi = np.random.randint(k, size=(m,))

# TODO(mwytock): Need to handle axis=1 parameter
# lambda: ep.softmax_loss(Theta, X, y_multi),
# lambda: ep.multiclass_hinge_loss(Theta, X, y_multi),

FUNCTION_TESTS = [
    lambda: ep.hinge_loss(theta, X, y_binary),
    lambda: ep.logistic_loss(theta, X, y_binary),
    lambda: ep.poisson_loss(theta, X, y_multi),
]

def run_function(f):
    prob = cp.Problem(cp.Minimize(f()))
    obj_val0 = prob.solve()

    status, obj_val1 = ep.solve(prob)
    tol = 1e-2
    assert_equal(status, OPTIMAL)
    assert_less(abs(obj_val1-obj_val0)/(1+abs(obj_val0)), tol)

def test_functions():
    for f in FUNCTION_TESTS:
        yield run_function, f
