from cvxbenchmarks.base import Instance, Results

from cvxpy.problems.solvers.utilities import SOLVERS # For compatibility check
from cvxpy.error import SolverError

import numpy as np

from collections import namedtuple
import time
import re
import hashlib

# For hashing
from cvxpy.settings import VAR_PREFIX, PARAM_PREFIX
VARSUB = re.compile(VAR_PREFIX + "[0-9]+")
PARAMSUB = re.compile(PARAM_PREFIX + "[0-9]+")


class CVXInstance(Instance):
    """An object for managing the data collection for a particular problem instance and
    a particular solver configuration.

    Attributes
    ----------
    problem : CVXProblem
       The problem to be solved.
    config : CVXConfig
       The configuration to use when solving this particular problem instance.

    """

    def run(self):
        """Runs the CVXProblem with the CVXConfig configuration.

        Returns
        -------
        results : cvxbenchmarks.CVXResults
            The results of running this instance.
        """
        # For convenience:
        problemID = self.problem.problemID
        configID = self.config.configID
        cvxpy_problem = self.problem.problem
        try:
            print("starting {} with config {}".format(problemID, configID))
            start = time.time() # Time the solve
            cvxpy_problem.solve(**self.config.configure())
            solve_time = time.time() - start
            print("finished solve for {} with config {}".format(problemID, configID))
            if cvxpy_problem.solver_stats.solve_time is not None:
                solve_time = cvxpy_problem.solver_stats.solve_time
            else: # pragma: no cover
                warn(configID + " did not report a solve time for " + problemID)
            setup_time = cvxpy_problem.solver_stats.setup_time
            num_iters = cvxpy_problem.solver_stats.num_iters
            status = cvxpy_problem.status
            opt_val = cvxpy_problem.value
        except Exception as e:
            print(e)
            # Configuration could not solve the given problem
            print(("failure solving {} " + 
                   "with config {} " +
                   "in {} sec.").format(problemID, 
                                        configID,
                                        round(time.time()-start, 1)))
            return CVXResults(problemID=problemID, configID=configID, 
                                 instancehash=hash(self))

        # Record residual gross stats:
        avg_abs_resid, max_resid = CVXResults.compute_residual_stats(cvxpy_problem)
        print("computed stats for {} with config {}".format(problemID, configID))

        print("finished {} with config {} in {} sec.".format(problemID, configID, round(time.time()-start, 1)))
        return CVXResults(problemID=problemID, configID=configID,
                             instancehash=hash(self), solve_time=solve_time,
                             setup_time=setup_time, num_iters=num_iters,
                             status=status, opt_val=opt_val,
                             avg_abs_resid=avg_abs_resid, max_resid=max_resid,
                         )

    def check_compatibility(self):
        """Ensure that the solver specified by the configuration
        is capable of solving the problem type.

        Leverages existing CVXPY functionality.
        """
        solver = SOLVERS[self.config.solver_opts["solver"]]
        try:
            _, constraints = self.problem.problem.canonicalize()
            solver.validate_solver(constraints)
        except SolverError as e:
            return False
        return True


    def __eq__(self, other):
        return str(self) == str(other)

    def __hash__(self):
        """
        Computes a string representation of the problem (including
        truncated, reduced-precision versions of all data matrices), replaces
        various variable names and parameter names, then hashes the resulting
        string.
        """
        old_options = np.get_printoptions()
        np.set_printoptions(threshold=10, precision=3) # Shorten the string representation.
        # Replace var<id> with just var and param<id> with just param
        problemString = str(self)
        problemString = re.sub(VARSUB, "var", problemString)
        problemString = re.sub(PARAMSUB, "param", problemString)
        digest = int(hashlib.sha256(problemString.encode("utf-16")).hexdigest(), 16)
        # print(problemString)
        np.set_printoptions(**old_options) # Restore original options
        return digest



cvxresultstp = namedtuple("CVXResultsd", 
    ["problemID", "configID", "instancehash",
     "solve_time", "setup_time", "num_iters",
     "status", "opt_val", "avg_abs_resid", "max_resid"])
class CVXResults(Results, cvxresultstp):
    """Holds the results of running a test instance.

    Attributes
    ----------
    Solve-related statistics:

    problemID : str
        The ID of the TestProblem.
    configID : str
        The ID of the SolverConfiguration.
    instancehash : int
        The hash digest of the TestInstance (used for caching).
    solve_time : float
        The time (in seconds) it took for the solver to solve the problem.
    setup_time : float
        The time (in seconds) it took for the solver to setup the problem.
    num_iters : int
        The number of iterations the solver had to go through to find a solution.
    status : string
        The status of the problem after solving, reported by the problem itself. (e.g. optimal, optimal_inaccurate, unbounded, etc.)
    opt_val : float
        The optimal value of the problem, as determined by the given solver configuration.
    avg_abs_resid : float
        The average absolute residual across all scalar problem constraints.
    max_resid : float
        The maximum absolute residual across all scalar problem constraints.
    """

    @staticmethod
    def compute_residual_stats(problem):
        """Computes the average absolute residual and the maximum residual
        of the current problem.

        Parameters
        ----------
        problem : cvxpy.Problem
            The problem whose residual stats we are computing.

        Returns
        -------
        avg_abs_resid : float or None
            The average absolute residual over all the scalar constraints of the problem.
        max_resid : float or None
            The maximum value of any residual over all scalar constraints of the problem.


        If the problem has no constraints, the function returns None, None.
        """
        if len(problem.constraints) == 0:
            return (None, None)
        sum_residuals = 0
        max_residual = 0
        n_residuals = 0

        for constraint in problem.constraints:
            # print(constraint.residual.is_constant())
            res = constraint.residual.value
            # print("1")
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
            elif isinstance(res, type(None)): # pragma: no cover
                continue
            else: # pragma: no cover
                print("Unknown residual type: {}".format(type(res)))
                continue

            # Get max absolute residual:
            if max_residual < thismax:
                max_residual = thismax
        if n_residuals == 0: # pragma: no cover
            return (None, None)
        return (sum_residuals/n_residuals, max_residual)

# 11 entries:
CVXResults.__new__.__defaults__ = (None, None, None, None, None, None, None, None,
                                    None, None)