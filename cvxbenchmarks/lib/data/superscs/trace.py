# SDP #2

# https://kul-forbes.github.io/scs/page_benchmarks.html

# Note that the methodology is roughly the same, even if the exact matrices generated are not.

import numpy as np
import cvxpy as cp
import scipy as sp

import scipy.sparse as sps
import scipy.linalg as la

np.random.seed(1)

n = 100
A = np.diag(-np.logspace(-0.5, 1, n))
U = la.orth(np.random.randn(n,n))
A = U.T.dot(A.dot(U))

P = cp.Symmetric(n, n)
f = cp.trace(P)
C = [A.T*P + P*A << np.eye(n),
     P >> np.eye(n)]

prob = cp.Problem(cp.Minimize(f), C)

problemDict = {
    "problemID": "trace",
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

