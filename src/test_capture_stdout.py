# Test capturing output with a StringIO object and multiprocessing.

# Uses the jinja2 library for templating.

# 

import os, sys, inspect
import numpy as np
cvxfolder = os.path.realpath(os.path.abspath(os.path.join(os.path.split(inspect.getfile(inspect.currentframe()))[0],"cvxpy")))
if cvxfolder not in sys.path:
    sys.path.insert(0, cvxfolder)
import cvxpy as cvx
print cvx
# Variable declarations

n = 100
m = 100
np.random.seed(1)

x = cvx.Variable(n)
A = np.random.rand(m, n)
b = np.random.rand(m)


# Problem construction

objective = cvx.Minimize(cvx.sum_squares(A*x - b))
prob = cvx.Problem(objective)

# Overwrite sys.stdout
# http://stackoverflow.com/questions/19420211/access-standard-output-of-a-sub-process-in-python
import sys
from StringIO import StringIO
from multiprocessing import Queue, Process, current_process

class CaptureVerbose(StringIO):
    def __init__(self, tag, *args, **kwargs):
        StringIO.__init__(self, *args, **kwargs)
        self.tag = tag
        self.str = ""
    def write(self, value):
        # self.queue.put(self.tag+value)
        self.str += (value)
        self.truncate(0)

def worker(problem, tag, done_queue):
    output_capture = CaptureVerbose(tag)
    sys.stdout = output_capture
    problem.solve(solver= "ECOS", verbose = True)
    done_queue.put(output_capture)

# For debugging individual problems:
if __name__ == "__main__":
    # captured_queue = Queue()
    # done_queue = Queue()
    # p1 = Process(target = worker, args = (prob, "run_1", done_queue))
    # p2 = Process(target = worker, args = (prob, "run_2", done_queue))

    # p1.start()
    # p2.start()

    # p1.join()
    # p2.join()

    # captured_queue.put("STOP")
    # done_queue.put("STOP")

    # print "process",p1,"exited with code",p1.exitcode
    # print "process",p2,"exited with code",p2.exitcode

    # for value in iter(captured_queue.get, "STOP"):
    #     # print value.replace("\n", "")
    #     pass

    # for value in iter(done_queue.get, "STOP"):
    #     print value.str


    results = prob.solve(solver = "CVXOPT", verbose = True)
    print prob.solver_stats.solve_time
    print prob.solver_stats.setup_time
    print prob.solver_stats.num_iters
    print "status:", prob.status
    print "optimal value", prob.value