
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

def worker(problemID, problem, configs):
    """A function for multithreading the solving of problems
    in parallel.

    Parameters
    ----------
    problemID : string
        The name (or unique identifier) for the problem.
    problem : cvxpy.Problem
        The problem to be solved
    configs : list of dictionaries
        The list of solver configurations to be applied to the problem.   
    """
    outdict = {}
    for configID, config in configs.items():
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


class TestFramework(object):
    """An object for managing the running of lots of configurations against lots
    of individual problem instances.

    Attributes
    ----------
    test_problems : list of TestFramework.TestProblem
        list of problems to run tests on
    solver_configs : list of TestFramework.SolverConfiguration
        list of configurations to solve the problems with

    """

    def __init__(self, test_problems = [], solver_configs = []):
        self.test_problems = test_problems
        self.solver_configs = solver_configs

    def load_problem(self, problemFile):
        """Loads a single problem and appends it to self.test_problems

        Parameters
        ----------
        problemFile : string
            The file where the problem is stored. Must be named "prob" within the file.
        """
        problem

    def preload_all_problems(self, problemDir = "problems"):
        """Loads all the problems in problemDir and adds them to self.test_problems

        Parameters
        ----------
        problemDir : string
            The name of the folder where the problems are located. Defaults to "problems".
        """

        # Might need to be parallelized:
        cmd_folder = os.path.realpath(os.path.abspath(os.path.join(os.path.split(inspect.getfile(inspect.currentframe()))[0], problemDir)))
        if cmd_folder not in sys.path:
            sys.path.insert(0, cmd_folder)

        for dirname, dirnames, filenames in os.walk(cmd_folder):
            for filename in filenames:
                # print filename
                if filename[-3:] == ".py" and filename != "__init__.py":
                    problemID = filename[0:-3]
                    self.test_problems.append(TestProblem(__import__(problemID).prob, self.solver_configs))




    def solve_all(self):
        """Worker function for solving all problems with all configurations
        """
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

    def solve_all_parallel(self):


class SolverConfiguration(object):
    """An object for managing the configuration of the cvxpy solver.

    Attributes
    ----------
    solver_name : string
        The name of the solver for which we are creating the configuration.
    verbose : boolean
        True if we want to capture the solver output, false otherwise.
    kwargs : dictionary
        Specifies the keyword arguments for the specific solver we are using.
    """

    def __init__(self, solver_name, verbose, kwargs):

        self.solver_name = solver_name
        self.verbose = verbose
        self.kwargs = kwargs

class TestProblem(object):
    """An object for managing the data collection for a particular problem instance.

    Attributes
    ----------
    problem : cvxpy.Problem
       The problem to be solved
    solver_configs : 
    
    """

    def __init__(self, problem, solver_configs):
        self.problem = problem
        self.solver_configs = solver_configs
        return

    def read_problem(self, problemFile):
        """Takes a file name of a python script containing a cvxpy problem named
        prob, reads it, and assigns the problem contained in it to self.problem.
        """



    def solve_problem(self):
        return
    


    def computeResidualStats():
        """Computes the average absolute residual and the maximum residual
        of the current problem.
        
        Returns:
        --------
        average residual : float
            The average value of the residuals of all the problem constraints.
        max residual : float
            The highest absolute residual across all constraints.

        If the problem has no constraints, the function returns "-", "-".
        """
        if len(self.problem.constraints) == 0:
            return ("-", "-")
        sum_residuals = 0
        max_residual = 0
        n_residuals = 0
        for constraint in self.problem.constraints:
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
            else:
                print "Unknown residual type:", type(res)

            # Get max absolute residual:
            if max_residual < thismax:
                max_residual = thismax

        return (sum_residuals/n_residuals, max_residual)

    def computeConstantStats(problem):
        """Computes the number of constant data values used and the 
        length of the longest side of a matrix of all the constants.

        Returns:
        --------
        number of data values : int
            The number of constants used across all matrices, vectors, in the problem.
            Some constants are not apparent when the problem is constructed: for example,
            The sum_squares expression is a wrapper for a quad_over_lin expression with a 
            constant 1 in the denominator.
        max length : int
            The number of data values of the longest dimension of any matrix.

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



