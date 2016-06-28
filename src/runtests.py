import cvxpy as cvx
import numpy as np
import multiprocessing as mp
import time
import os, sys, inspect
import pandas as pd

def computeResidualStats(problem):
    """compute the average absolute residual and the maximum residual"""
    sum_residuals = 0
    max_residual = 0
    nresiduals = 0
    for constraint in problem.constraints:
        # if constraint.__class__.__name__ is "LeqConstraint":
        # compute average absolute residual:
        nresiduals += np.prod(constraint._expr.size)
        sum_residuals += np.absolute(constraint._expr.value).sum()
        # get max absolute residual:
        thismax = np.absolute(constraint._expr.value).max()
        if max_residual < thismax:
            max_residual = thismax
    if nresiduals is 0:
        return ("-", "-")
    else:
        return (sum_residuals/nresiduals, max_residual)



# Worker function
# Modelled after:
# http://eli.thegreenplace.net/2012/01/16/python-parallelizing-cpu-bound-tasks-with-multiprocessing/
def worker(problemID, problem, configs):
    outdict = {}
    for configID in configs:
        config = configs[configID]
        # output = {}
        # Default values:
        runtime = "-"
        status = "-"
        opt_val = "-"
        avg_abs_resid = "-"
        max_resid = "-"
        try:
            start = time.time() # Time the solve
            print "starting",problemID,"with config",configID,"at",start
            problem.solve(**config)
            runtime = time.time() - start
            status = problem.status
            opt_val = problem.value
            # Record residual gross stats:
            avg_abs_resid, max_resid = computeResidualStats(problem)
        except:
            # Configuration could not solve the given problem
            print "failure in solving."
        outdict[(problemID, configID)] = [status, opt_val, runtime, avg_abs_resid, max_resid];
    return outdict


# Create solver configurations
configs ={solver : {"solver": solver} for solver in ['MOSEK', 'CVXOPT', 'SCS', 'ECOS']}
problemDict = {}

# Read in problems
# http://stackoverflow.com/questions/279237/import-a-module-from-a-relative-path
cmd_folder = os.path.realpath(os.path.abspath(os.path.join(os.path.split(inspect.getfile(inspect.currentframe()))[0],"problems")))
if cmd_folder not in sys.path:
    sys.path.insert(0, cmd_folder)

for dirname, dirnames, filenames in os.walk(cmd_folder):
    for filename in filenames:
        # print filename
        if filename[-3:] == ".py" and filename != "__init__.py":
            problemID = filename[0:-3]
            problemDict[problemID] = __import__(problemID).prob



# Run every solver configuration against every problem and save the results
# Create process pool:
pool = mp.Pool(mp.cpu_count())

# Check runtime
startall = time.time()

if __name__ == "__main__":
    results = [ pool.apply_async(worker, args = (problemID, problemDict[problemID], configs)) for problemID in problemDict]

    # Prevent adding more tasks and wait for processes to finish
    pool.close()
    pool.join()

    # for result in results:
        # print result.get(timeout = 1), "\n"

    labels = [  "status",           # optimal, unbounded, etc.
                "opt_val",          # The optimal value of the solution
                "time",             # The amount of time it took to solve the problem
                "avg_abs_resid",    # The average absolute residual for LeqConstraints
                "max_resid"     # The maximum absolute residual for LeqConstraints
    ]           
    problemList = [problemID for problemID in problemDict]
    configList = [config for config in configs]

    # Display results:
    problemOutputs = pd.Panel(  items = labels, 
                            major_axis = problemList, 
                            minor_axis = configList)

    for result in results:
        for (problemID, configID), data in result.get(timeout = 1).items():
            problemOutputs.loc[:,problemID, configID] = data

    # Compute Statistics
    # percent_error - using MOSEK as a standard, the percent error in the optimal value
    percent_error = pd.DataFrame(index = problemList, columns = configList)
    for configID in configs:
        for problemID in problemDict:
            try:
                percent_error.loc[problemID, configID] = 100*np.absolute((problemOutputs.loc["opt_val", problemID,configID] - problemOutputs.loc["opt_val", problemID, "MOSEK"])/problemOutputs.loc["opt_val", problemID,"MOSEK"])
            except:
                percent_error.loc[problemID, configID] = "-"
    problemOutputs["percent_error"] = percent_error

    print problemOutputs.to_frame()


print "Runtime:",str(time.time() - startall)
