# This is automatically-generated code.

# Uses the jinja2 library for templating.

# 

from cvxpy import *
import numpy as np

# Variable declarations

# Generate data for control problem.
np.random.seed(1)
m = 4 # number of inputs
n = 20 # number of states
T = 50 # number of time steps
alpha = 0.2
beta = 5
A = np.eye(n) + alpha*np.random.randn(n,n)
B = np.random.randn(n,m)
x_0 = beta*np.random.randn(n,1)

# Form and solve control problem.
x = Variable(n, T+1)
u = Variable(m, T)


# Problem construction

states = []
for t in range(T):
    cost = pnorm(u[:,t], 1)
    constr = [x[:,t+1] == A*x[:,t] + B*u[:,t],
              norm(u[:,t], 'inf') <= 1]
    states.append( Problem(Minimize(cost), constr) )
# sums problem objectives and concatenates constraints.
prob = sum(states)
prob.constraints += [x[:,T] == 0, x[:,0] == x_0]


# For debugging individual problems:
if __name__ == "__main__":
	prob.solve()
	print "status:", prob.status
	print "optimal value", prob.value