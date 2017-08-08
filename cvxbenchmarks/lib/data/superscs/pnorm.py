# Minimization of P-norm

# https://kul-forbes.github.io/scs/page_benchmarks.html

# Note that the methodology is roughly the same, even if the exact matrices generated are not.

import numpy as np
import cvxpy as cp
import scipy as sp

import scipy.sparse as sps


np.random.seed(1)

n = 2000
m = n//4
density = 0.1

def sprandn(m, n, density):
    A = sps.rand(m, n, density)
    A.data = np.random.randn(A.nnz)
    return A.todense()

G = sprandn(m,n,density);
f = np.random.randn(m,1) * n * density
power = 1.5
x = cp.Variable(n)
obj = cp.norm(x, power)
C = [G*x == f]
prob = cp.Problem(cp.Minimize(obj), C)

problemDict = {
    "problemID": "pnorm",
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
