# import cvxpy as cvx
import numpy as np
import multiprocessing as mp
import time
import os, sys, inspect
import pandas as pd

# Use local repository:
cvxfolder = os.path.realpath(os.path.abspath(os.path.join(os.path.split(inspect.getfile(inspect.currentframe()))[0],"cvxpy")))
if cvxfolder not in sys.path:
    sys.path.insert(0, cvxfolder)
import cvxpy as cvx
print cvx

def computeResidualStats(problem):
    """Computes the average absolute residual and the maximum residual
    """
    sum_residuals = 0
    max_residual = 0
    n_residuals = 0
    for constraint in problem.constraints:
        res = constraint.residual.value
        thismax = 0

        # Compute average absolute residual:
        if isinstance(res, np.matrix):
            n_residuals += np.prod(res.size)
            sum_residuals += res.sum()
            thismax = np.absolute(res).max()
        elif isinstance(res, float) or isinstance(res, int):
            # res is a float
            n_residuals += 1
            thismax = np.absolute(res)
        elif isinstance(res, type(None)):
            # res is None
            continue
        else:
            print "Unknown residual type:", type(res)

        # Get max absolute residual:
        if max_residual < thismax:
            max_residual = thismax
    if n_residuals == 0:
        return ("-", "-")
    return (sum_residuals/n_residuals, max_residual)

def computeConstantStats(problem):
    """Computes the number of constant data values used and the 
    length of the longest side of a matrix of all the constants.
    """
    if len(problem.constants()) == 0:
        return ("-", "-")
    max_length = 0
    n_data = 0
    for const in problem.constants():
        thismax = 0
        # Compute number of data
        if isinstance(const, np.matrix):
            n_data += np.prod(const.size)
            thismax = max(const.shape)
        elif isinstance(const, float) or isinstance(const, int):
            # const is a float
            n_data += 1
            thismax = 1
        else:
            print "Unknown constant type:", type(const)
        # Get max absolute residual:
        if max_length < thismax:
            max_length = thismax
    return (n_data, max_length)

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
        except:
            # Configuration could not solve the given problem
            print "failure in solving."

        # Record residual gross stats:
        avg_abs_resid, max_resid = computeResidualStats(problem)

        # Record the number of variables:
        n_vars = sum(np.prod(var.size) for var in problem.variables())

        # Record the number of constraints:
        n_eq = problem.n_eq()
        n_leq = problem.n_leq()

        # Record constant stats
        n_data, n_big = computeConstantStats(problem)
        outdict[(problemID, configID)] = [  status, 
                                            opt_val, 
                                            runtime, 
                                            avg_abs_resid, 
                                            max_resid,
                                            n_vars,
                                            n_eq,
                                            n_leq,
                                            n_data,
                                            n_big
        ]
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
                "max_resid",        # The maximum absolute residual for LeqConstraints
                "n_vars",           # The number of variables in the problem
                "n_eq",             # The number of equality constraints in the problem.
                "n_leq",            # The number of inequality constraints in the problem.
                "n_data",           # The number of data values in constants in the problem.
                "n_big",            # The length of the largest dimension of any matrix
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
