# Norm-constrained Minimum-norm

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

m = 3000
n = m//2
A = sprandn(m,n,0.5);
b = 10*np.random.randn(m,1);
G = 2*sprandn(2*n, n, 0.1);

x = cp.Variable(n)
f = cp.norm(A*x - b)
C = [cp.norm(G*x) <= 1]


prob = cp.Problem(cp.Minimize(f), C)

# Single problem collection
problemDict = {
    "problemID" : "min_norm",
    "problem"   : prob,
    "opt_val"   : None
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
