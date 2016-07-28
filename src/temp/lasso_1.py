# This is automatically-generated code.

# Uses the jinja2 library for templating.

# 

from cvxpy import *
import numpy as np

# Variable declarations

n = 20
m = 300

np.random.seed(1)

A = np.random.rand(m, n)
b = np.random.rand(m)
Lambda = 5

x = Variable(n)


# Problem construction

objective = Minimize(0.5*sum_squares(A*x - b) + Lambda*pnorm(x, 1))
prob = Problem(objective)


# For debugging individual problems:
if __name__ == "__main__":
	prob.solve()
	print "status:", prob.status
	print "optimal value", prob.value