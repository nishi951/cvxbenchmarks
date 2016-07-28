# Test Segfault CVXOPT

from cvxpy import *
import numpy as np

#############
# control_0 #
#############

# Variable declarations

np.random.seed(1)
n = 40
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

control_0 = Problem(objective, constraints)

###################
# portfolio_opt_1 #
###################

# Variable declarations

np.random.seed(1)
n = 40
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

portfolio_opt_1 = Problem(objective, constraints)

################
# Test Objects #
################



