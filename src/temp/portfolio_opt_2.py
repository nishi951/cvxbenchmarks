# This is automatically-generated code.

# Uses the jinja2 library for templating.

# 

from cvxpy import *
import numpy as np

# Variable declarations

np.random.seed(1)
n = 50
gamma = 1

mu = np.exp(np.random.normal(0, 1, n))
# Generate S as FF^T + D
F = np.random.normal(0, 0.1, [n, n])
D = np.diag(0.1*np.random.rand(n)+ 0.1)

S = F*F.T + D

x = Variable(n)


# Problem construction

objective = Maximize(mu*x - gamma*quad_form(x,S))
constraints = [sum_entries(x) == 1, x >= 0]

prob = Problem(objective, constraints)


# For debugging individual problems:
if __name__ == "__main__":
	prob.solve()
	print "status:", prob.status
	print "optimal value", prob.value