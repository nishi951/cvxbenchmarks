# Logistic Regression

# https://kul-forbes.github.io/scs/page_benchmarks.html

# Note that the methodology is roughly the same, even if the exact matrices generated are not.

import numpy as np
import cvxpy as cp
import scipy as sp

import scipy.sparse as sps

np.random.seed(1)
density = 0.1
p = 1000   # features
q = 10*p  # total samples

def sprandn(m, n, density):
    A = sps.rand(m, n, density)
    A.data = np.random.randn(A.nnz)
    return A

w_true = sprandn(p, 1, density).todense()
X_tmp = sprandn(p, q, density).todense()

ips = -w_true.T.dot(X_tmp)
ps = (np.exp(ips)/(1 + np.exp(ips))).T
labels = 2*(np.random.rand(q,1) < ps) - 1
X_pos = X_tmp[:,np.where(labels==1)[0]]
X_neg = X_tmp[:,np.where(labels==-1)[0]]
X = np.hstack([X_pos, -X_neg]) # include labels with data
lam = 2


w = cp.Variable(p, 1)
f = cp.sum_entries(cp.log_sum_exp(cp.vstack([np.zeros((1,q)), w.T*X]), axis = 0)) + lam * cp.norm(w,1)

prob = cp.Problem(cp.Minimize(f))

problemDict = {
    "problemID": "log_reg",
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
