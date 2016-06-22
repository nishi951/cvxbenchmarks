# lasso_0
# minimize ||Ax -b||^2 + Lambda*||x||_1

from cvxpy import *
import numpy as np

m = 20
n = 10
np.random.seed(1)

A = np.random.rand(m, n)
b = np.random.rand(m)
Lambda = 5

x = Variable(n)

objective = Minimize(0.5*sum_squares(A*x - b) + Lambda*pnorm(x, 1))

prob = Problem(objective)

# prob.solve()

# print "status:", prob.status
# print "optimal value", prob.value
# print "optimal var", x.value
