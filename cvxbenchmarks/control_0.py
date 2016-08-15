# This is automatically-generated code.

# Uses the jinja2 library for templating.

# 

from cvxpy import *
import numpy as np

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
prob = sum(states)
prob.constraints += [x[:,T] == 0, x[:,0] == x_0]


# For debugging individual problems:
if __name__ == "__main__":
    # is prob pickleable:
    import pickle
    try:
        pickle.dumps(prob)
    except:
        print 'not picklable.'


    # Test the solve under CVXOPT in a separate process.
    import multiprocessing
    import TestFramework

    def worker(work_queue, done_queue):
        while True:
            prob = work_queue.get()
            if prob == "STOP":
                print "exiting process."
                break
            kwargs = {}
            prob.solve(solver = "CVXOPT", verbose = False, **kwargs)
            print "done solving."
            done_queue.put((prob.status, prob.value))
        return

    work_queue = multiprocessing.Queue()
    done_queue = multiprocessing.Queue()

    work_queue.put(prob)

    workers = 1 # Number of processes.
    p = multiprocessing.Process(target = worker, args = (work_queue, done_queue))
    p.start()
    work_queue.put("STOP")

    # Wait for process to finish.
    p.join()
    print "process",p,"exited with code",p.exitcode

    done_queue.put(("STOP", "STOP"))

    for status, value in iter(done_queue.get, ("STOP", "STOP")):
        print "status:", status
        print "optimal value", value






	# prob.solve()
	# print "status:", prob.status
	# print "optimal value", prob.value