# This is automatically-generated code.

# Uses the jinja2 library for templating.

# 

from cvxpy import *
import numpy as np

# Variable declarations

m = 70
n = 90
np.random.seed(1)

A = np.random.rand(m, n)

y = (np.random.rand(n) - 0.5)
x_hat = -np.select([y < 0], [y])
Lambda = np.select([y >= 0], [y]) # x_hat^T Lambda = 0

mu = np.random.rand(m)
c = Lambda - A.T.dot(mu)
b = A.dot(x_hat)
x = Variable(n)


# Problem construction

objective = Minimize(c*x)
constraints = [A*x == b, x >= 0]

prob = Problem(objective, constraints)


# For debugging individual problems:
if __name__ == "__main__":
	prob.solve()
	print "status:", prob.status
	print "optimal value", prob.value