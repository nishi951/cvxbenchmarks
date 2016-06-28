# lp_0
# minimize c^T x
# subject to Ax = b and x >= 0

# Lagrangian:
# c^T x - Lambda^T x + mu^T (Ax - b)
# For feasible x, Ax - b = 0
# Then, KKT conditions imply Lambda^T x = 0 (complementary slackness)
# Finally, we must have c = Lambda - A.T.dot(mu), choosing mu randomly
#

from cvxpy import *
import numpy as np

m = 10
n = 20
np.random.seed(1)

A = np.random.rand(m, n)

y = (np.random.rand(n) - 0.5)
x_hat = -np.select([y < 0], [y])
Lambda = np.select([y >= 0], [y]) # x_hat^T Lambda = 0

mu = np.random.rand(m)
c = Lambda - A.T.dot(mu)
b = A.dot(x_hat)
x = Variable(n)

objective = Minimize(c*x)
constraints = [A*x == b, x >= 0]

prob = Problem(objective, constraints)

# prob.solve()

# print "status:", prob.status
# print "optimal value", prob.value
# print "optimal var", x.value
