# least_squares_0

from cvxpy import *
import numpy as np

m = 20
n = 10
np.random.seed(1)

x = Variable(n)
A = np.random.rand(m, n)
b = np.random.rand(m)

objective = Minimize(sum_squares(A*x - b))
prob = Problem(objective)

# prob.solve()

# print "status:", prob.status
# print "optimal value", prob.value
# print "optimal var", x.value
