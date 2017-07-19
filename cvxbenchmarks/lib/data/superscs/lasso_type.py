# LASSO-type problem

# https://kul-forbes.github.io/scs/page_benchmarks.html

# Note that the methodology is the same, even if the exact matrices generated are not.

import numpy as np
import cvxpy as cp
import scipy as sp

import scipy.sparse as sps


# Variable declarations
n = 10000
m = 2000

s = np.ceil(n/10)
x_true = np.hstack((np.random.randn(s, 1), np.zeros(n-s, 1)))
x_true = np.random.permutation(x_true)

density = 0.1
rcA = 0.1
A = sps.random(m, n, density, data_rvs = np.random.randn)

b = A*x_true + 0.1*np.random.randn(m, 1)
mu = 1


# Problem construction
x = cp.Variable(n)

prob = cp.Problem(cp.Minimize(0.5*cp.sum_squares(A*x) + mu*cp.norm1(x)))

problems = {"superscs_lasso" : prob}


