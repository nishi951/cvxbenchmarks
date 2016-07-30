# Test Segfault CVXOPT

from cvxpy import *
import numpy as np

#############
# control_0 #
#############

# Variable declarations

# Generate data for control problem.
np.random.seed(1)
m = 2 # number of inputs
n = 8 # number of states
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
control_0 = sum(states)
control_0.constraints += [x[:,T] == 0, x[:,0] == x_0]

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

#####################
# Solve in parallel #
#####################
import multiprocessing

def worker(problem):
    print "solving"
    problem.solve(solver="CVXOPT")
    print "done."

p1 = multiprocessing.Process(target = worker, args = (control_0,))
p2 = multiprocessing.Process(target = worker, args = (portfolio_opt_1,))

p1.start()
p2.start()

p1.join()
p2.join()

print "process",p1,"exited with code",p1.exitcode
print "process",p2,"exited with code",p2.exitcode





