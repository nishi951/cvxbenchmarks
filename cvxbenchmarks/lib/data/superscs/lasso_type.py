# LASSO-type problem

# https://kul-forbes.github.io/scs/page_benchmarks.html

# Note that the methodology is roughly the same, even if the exact matrices generated are not.

import numpy as np
import cvxpy as cp
import scipy as sp

import scipy.sparse as sps


# Variable declarations
np.random.seed(1)

n = 10000
m = 2000
s = n//10
x_true = np.vstack([np.random.randn(s, 1), np.zeros((n-s, 1))])
x_true = np.random.permutation(x_true)

density = 0.1
rcA = 0.1

def sprandn(m, n, density):
    A = sps.rand(m, n, density)
    A.data = np.random.randn(A.nnz)
    return A

A = sprandn(m, n, density).todense()

b = A*x_true + 0.1*np.random.randn(m, 1)
mu = 1


# Problem construction
x = cp.Variable(n)

prob = cp.Problem(cp.Minimize(0.5*cp.sum_squares(A*x) + mu*cp.norm1(x)))

problemDict = {
    "problemID": "lasso_type",
    "problem": prob,
    "opt_val": None
}

problems = [problemDict]

# For debugging individual problems:
if __name__ == "__main__":
    def printResults(problemID = "", problem = None, opt_val = None):
        print(problemID)
        problem.solve()
        print("\tstatus: {}".format(problem.status))
        print("\toptimal value: {}".format(problem.value))
        print("\ttrue optimal value: {}".format(opt_val))
    printResults(**problems[0])
