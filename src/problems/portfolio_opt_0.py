# portfolio_opt_0
# maximize mu^T z - gamma (z^T S z)
# subject to 1^T z = 1 and z >= 0

from cvxpy import *
import numpy as np

np.random.seed(2)
n = 10
gamma = 1

mu = np.exp(np.random.normal(0, 1, n))
# Generate S as FF^T + D
F = np.random.normal(0, 0.1, [n, n])
D = np.diag(0.1*np.random.rand(n)+ 0.1)

S = F*F.T + D

x = Variable(n)

objective = Maximize(mu*x - gamma*quad_form(x,S))
constraints = [sum_entries(x) == 1, x >= 0]

prob = Problem(objective, constraints)

	# prob.solve()

	# print "status:", prob.status
	# print "optimal value", prob.value
	# print "optimal x", x.value
