# lp_0
# http://foges.github.io/pogs/egs/linear-program.html
# minimize c^T x
# subject to Ax = b and x >= 0

from cvxpy import *
import numpy as np

m = 20
n = 10
np.random.seed(1)

c = np.random.rand(n)
A = np.random.rand(m, n)
x_hat = np.random.rand(n)
b = A.dot(x_hat)

x = Variable(n)

objective = Minimize(c*x)
constraints = [A*x == b, x >= 0]

prob = Problem(objective, constraints)

# prob.solve()

# print "status:", prob.status
# print "optimal value", prob.value
# print "optimal var", x.value
