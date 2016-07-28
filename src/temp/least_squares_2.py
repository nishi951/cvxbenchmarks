# This is automatically-generated code.

# Uses the jinja2 library for templating.

# 

from cvxpy import *
import numpy as np

# Variable declarations

n = 200
m = 300
np.random.seed(1)

x = Variable(n)
A = np.random.rand(m, n)
b = np.random.rand(m)


# Problem construction

objective = Minimize(sum_squares(A*x - b))
prob = Problem(objective)


# For debugging individual problems:
if __name__ == "__main__":
	prob.solve()
	print "status:", prob.status
	print "optimal value", prob.value