# SDP #1

# https://kul-forbes.github.io/scs/page_benchmarks.html

# Note that the methodology is roughly the same, even if the exact matrices generated are not.

import numpy as np
import cvxpy as cp
import scipy as sp

import scipy.sparse as sps

np.random.seed(1)

n = 800
P = np.random.randn(n, n)
Z = cp.Semidef(n, n)
f = cp.norm(P - Z, "fro")
# Toeplitz condition: all diagonals are equal
C = []
for i in range(n):
    for j in range(i+1):
        C += [Z[i,j] == Z[(i+1) % n, ((j+1) % n)]]

prob = cp.Problem(cp.Minimize(f), C)

# Single problem collection
problemDict = {
    "problemID" : "sdp",
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
