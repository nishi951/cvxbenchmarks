# Regularized Matrix Completion

# https://kul-forbes.github.io/scs/page_benchmarks.html

# Note that the methodology is roughly the same, even if the exact matrices generated are not.

import numpy as np
import cvxpy as cp
import scipy as sp

import scipy.sparse as sps

np.random.seed(1)


def sprandn(m, n, density):
    A = sps.rand(m, n, density)
    A.data = np.random.randn(A.nnz)
    return A

m=200
n=200
n_nan = int(np.ceil(0.8*m*n))
M = sprandn(m, n, 0.4).todense()
idx = np.random.permutation(m*n)
M = M.flatten()
M[:,idx[:n_nan]] = np.nan
M = M.reshape((m, n))
lam = 0.5
X = cp.Variable(m, n)
f = cp.norm(X, "nuc") + lam*cp.sum_squares(X)
C = []
for i in range(m):
    for j in range(n):
        if np.isnan(M[i, j]):
            C += [X[i, j] == M[i, j]]

prob = cp.Problem(cp.Minimize(f), C)

problemDict = {
    "problemID": "mat_comp",
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


