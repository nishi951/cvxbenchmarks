import numpy as np
import multiprocessing, signal
import time
import os, sys, inspect, glob, re
import pandas as pd
import math

import hashlib, pickle as pkl

from collections import namedtuple

import cvxbenchmarks.settings as s

# Constraint types
from cvxpy.constraints.semidefinite import SDP
from cvxpy.constraints.exponential import ExpCone
from cvxpy.constraints.second_order import SOC
from cvxpy.constraints.bool_constr import BoolConstr

# SizeMetrics
from cvxpy.problems.problem import SizeMetrics


from warnings import warn

STOP = "STOP" # Poison pill for parallel solve subroutine.

# Use local repository:

# cvxfolder = os.path.realpath(os.path.abspath(os.path.join(os.path.split(inspect.getfile(inspect.currentframe()))[0],"cvxpy")))
# if cvxfolder not in sys.path:
    # sys.path.insert(0, cvxfolder) 
# sys.path.insert(0, "/Users/mark/Documents/Stanford/reu2016/cvxpy")
import cvxpy as cvx
print(cvx)

# Variable Regular expression for hashing:
from cvxpy.settings import VAR_PREFIX, PARAM_PREFIX
VARSUB = re.compile(VAR_PREFIX + "[0-9]+")
PARAMSUB = re.compile(PARAM_PREFIX + "[0-9]+")




def worker(problemDir, configDir, work_queue, done_queue): 
    """Worker function for multithreading the solving of test instances.
    Parameters
    ----------
    problemDir : string
        The directory where the test problem is located
    configDir : string
        The directory where the test configuration is located
    work_queue : multiprocessing.Queue
        A queue with the pairs (problemID, configID) that represents the
        next test instance to solve.
    done_queue : multiprocessing.Queue
        A queue for returning the TestResult objects to the main process.

    A note on the design: because future problems might become quite large, 
    it might become infeasible to pass them directly to the workers via 
    queues. Instead we pass just the file paths and let the worker
    read the problem directly into its own memory space. 
    """
    while True:
        problemID, configID = work_queue.get()
        print("received")
        if problemID == STOP:
            # Poison pill
            print("Exiting worker process.")
            done_queue.put(STOP)
            break
        testproblemList = TestProblem.get_all_from_file(problemID, problemDir)
        solverconfig = SolverConfiguration.from_file(configID, configDir)
        for testproblem in testproblemList:
            test_instance = TestInstance(testproblem, solverconfig)
            result = test_instance.run()
            done_queue.put(result)
    return 


class TestFramework(object):
    """An object for managing the running of lots of configurations 
    against lots of individual problem instances.

    Attributes
    ----------
    problemDir : string
        Directory containing desired problem files.
    configDir : string
        Directory containing desired config files.
    problems : list of TestFramework.TestProblem
        list of problems to solve.
    configs : list of TestFramework.SolverConfiguration
        list of configurations under which to solve the problems.
    cache : string
        File containing a shelf object for storing TestInstance hashes mapped
        to TestResults objects.

        Options:
        --------
        parallel : boolean
            Whether or not to run the TestInstances in parallel with each other.
        tags : list 
            List of tags specifying which problems types (e.g. SDP, SOCP)
            we should solve.

    instances : list of TestFramework.TestInstance
        list of test instances to run.
    results : list of TestFramework.TestResults
        list of results from running each test instance.


    Workflow:
    Read in problems (TestProblem)
        Optional: Use index to filter for problems.
    Read in solver configurations (SolverConfiguration)
    Generate TestInstances
    Run all TestInstances (possibly in parallel?) and record TestResults

    """

    def __init__(self, problemDir, configDir, 
                 testproblems=None, solverconfigs=None, cacheFile="cache.pkl", 
                 parallel=False, tags=None, instances=None, results=None):
        self.problemDir = problemDir
        self.configDir = configDir
        if testproblems is None:
            self.testproblems = []
        else:
            self.testproblems = testproblems
        if solverconfigs is None:
            self.solverconfigs = []
        else:
            self.solverconfigs = solverconfigs
        self.cacheFile = cacheFile

        # Runtime options
        self.parallel = parallel
        if tags is None:
            self.tags = []
        else:
            self.tags = tags

        # Properties
        if instances is None:
            self.instances = []
        else:
            self.instances = instances
        if results is None:
            self.results = []
        else:
            self.results = results

    def load_problem_file(self, fileID):
        """Loads a single problem file and appends all problems
        in it to self.testproblems.

        Parameters
        ----------
        fileID : string
            A unique identifier for the file to be loaded. File 
            containing the problem should be in the format <fileID>.py.
            <fileID>.py can also contain a list of problems.
        """
        self.testproblems.extend(TestProblem.get_all_from_file(fileID, self.problemDir))

    def preload_all_problems(self):
        """Loads all the problems in self.problemDir and adds them to 
        self.testproblems.
        """
        for _, _, filenames in os.walk(self.problemDir):
            for filename in filenames:
                if filename[-3:] == ".py" and filename != "__init__.py":
                    self.load_problem_file(filename[0:-3])

    def load_config(self, configID):
        """Loads a single solver configuration, checking if 
           cvxpy supports it:

        Parameters
        ----------
        configID : string
            A unique identifier for the solver configuration to be 
            loaded. File containing the configuration should be in the form 
            <configID>.py.
        """
        solverconfig = SolverConfiguration.from_file(configID, self.configDir)
        if solverconfig is not None:
            self.solverconfigs.append(solverconfig)
        else:
            warn(UserWarning("{} configuration specified but not installed.".format(configID)))

    def preload_all_configs(self):
        """Loads all the configs in self.configDir and adds them to self.solverconfigs.
        """
        for _, _, filenames in os.walk(self.configDir):
            for filename in filenames:
                if filename[-3:] == ".py" and filename != "__init__.py":
                    configID = filename[0:-3]
                    self.load_config(configID)

    def generate_test_instances(self):
        """Generates a test problem for every pair of (problem, config).
        """
        for testproblem in self.testproblems:
            for solverconfig in self.solverconfigs:
                self.instances.append(TestInstance(testproblem, solverconfig))

    def clear_cache(self): # pragma: no cover
        """Clear the cache used to store TestResults
        """
        # Overwite with an empty dictionary
        with open(self.cacheFile, "wb") as f:
            pkl.dump({}, f)
        return

    def solve(self, use_cache=True):
        """Solve all the TestInstances we have queued up.

        Parameters
        ----------
        use_cache : boolean
            Whether or not we should use the cache specified in self.cacheFile
        """
        if self.parallel:
            self.solve_all_parallel(use_cache)
        else:
            self.solve_all(use_cache)

    def solve_all(self, use_cache=True):
        """Solves all test instances and reports the results.
        """
        self.generate_test_instances()
        self.results = []
        if use_cache: 
            # Load the cache dictionary from the cache file:
            cachedResults = {}
            try:
                with open(self.cacheFile, "rb") as f:
                    cachedResults = pkl.load(f)
            except Exception as e: # pragma: no cover
                print(e)
                print("Creating new cache file: {}".format(self.cacheFile))
                self.clear_cache()

            for instance in self.instances:
                instancehash = hash(instance)
                if instancehash in cachedResults:
                    # Retrieve TestResult from the results dictionary:
                    self.results.append(cachedResults[instancehash])
                    print(("Retrieved instance result ({}, {}) " +
                           "from cache.").format(instance.testproblem.problemID,
                                                 instance.solverconfig.configID))
                else:
                    # Add this result to the cache
                    result = instance.run()
                    self.results.append(result)
                    cachedResults[instancehash] = result
                    # Write the modified dictionary back to the cache file.

                with open(self.cacheFile, "wb") as f: # Overwrite previous cache
                    pkl.dump(cachedResults, f)

        else:
            for instance in self.instances:
                self.results.append(instance.run())
        return


    def solve_all_parallel(self, use_cache=True):
        """Solves all test instances in parallel and reports the results.
        DO NOT USE WHEN USING THE CVXOPT SOLVER!
        """
        self.generate_test_instances()

        # workers = multiprocessing.cpu_count()/2
        workers = 8

        # create two queues: one for files, one for results
        work_queue = multiprocessing.Queue()
        done_queue = multiprocessing.Queue()
        processes = []

        # add filepaths to work queue
        # format is (problemID, configID)
        # start processes
        if use_cache:
            cachedResults = {}
            try:
                with open(self.cacheFile, "rb") as f:
                    cachedResults = pkl.load(f)
            except: # pragma: no cover
                print("Creating new cache file: {}".format(self.cacheFile))
            with open(self.cacheFile, "wb") as f:
                for instance in self.instances:
                    instancehash = hash(instance)
                    if instancehash in cachedResults:
                        # Retrieve TestResult from the results dictionary:
                        self.results.append(cachedResults[instancehash])
                    else:
                        # Add this result to the cache
                        work_queue.put((instance.testproblem.problemID, instance.solverconfig.configID))

        else:
            for instance in self.instances:
                work_queue.put((instance.testproblem.problemID, instance.solverconfig.configID))

        for w in range(workers):
            p = multiprocessing.Process(target=worker,
                                        args=(self.problemDir,
                                              self.configDir,
                                              work_queue,
                                              done_queue))
            p.start()
            processes.append(p)
            work_queue.put((STOP,STOP))

        # Poll done_queue and empty it right away.
        # keep track of the number of poison pills we get-
        # once it's equal to the number of workers, stop.
        processes_left = workers
        while processes_left:

            if not done_queue.empty():
                result = done_queue.get()
                if result == STOP:
                    processes_left -= 1
                    print("Processes left: {}".format(str(processes_left)))
                else:
                    self.results.append(result)
                    if use_cache: # Add new cached result to the cache.
                        with open(self.cacheFile, "wb") as f:
                            cachedResults[result.instancehash] = result
                            pkl.dump(cachedResults, f)
            time.sleep(0.5) # Wait for processes to run.

        for p in processes:
            print("process {} exited with code {}".format(p,p.exitcode))
        return

    def export_results(self):
        """Convert results into a pandas (multiindex) dataframe object for easier data visualization

        Returns
        -------
        output : pandas.DataFrame
            A (multiindex) dataframe containing the results of the testing.
            The first index is the multiindex combining the problem IDs 
            and the config IDs.
            The second index is a normal 1D index consisting of the keys 
            (e.g. solve_time, num_iters, etc.)

            The access format is:
            output.loc[(problemID, configID), key]

            Example: To access the solve time of scs on the problem basis_pursuit_0,
            we would do:

            output.loc[("basis_pursuit_0", "scs_config"), "solve_time"]

        """
        problemIDs = list(set([result.problemID for result in self.results]))
        configIDs = list(set([result.configID for result in self.results]))

        labels = []
        labels.extend(TestResults._fields)
        labels.extend(SizeMetrics._fields)    
        # Remove unused columns
        labels.remove("size_metrics")
        labels.remove("problemID")
        labels.remove("configID")

        # output = pd.Panel(items=labels, major_axis=problemIDs, minor_axis=configIDs)
        multiindex = pd.MultiIndex.from_product([problemIDs, configIDs], names=["problems", "configs"])

        output = pd.DataFrame(index=multiindex, columns=labels)
        output.columns.names = ["stats"]

        for result in self.results:
            problemID = result.problemID
            configID = result.configID
            for label in [label for label in TestResults._fields if label in labels]:
                output.loc[(problemID, configID), label] = getattr(result, label)
            for label in [label for label in SizeMetrics._fields if label in labels]:
                output.loc[(problemID, configID), label] = getattr(result.size_metrics, label)

        # Compute Statistics
        output.fillna(value=np.nan, inplace=True)
        output.sort_index(inplace=True)
        try:
            TestFramework.compute_mosek_error(output, "opt_val", "mosek_config")
        except (KeyError): # pragma: no cover
            print("TestFramework.compute_mosek_error: 'mosek_config' or 'opt_val' field not found.")
        try:
            TestFramework.compute_performance(output, "solve_time")
        except (KeyError): # pragma: no cover
            print("TestFramework.compute_performance: 'solve_time' field not found.")
        return output

    @staticmethod
    def compute_mosek_error(results, opt_val, mosek_config, abstol=10e-4):
        """Takes a dataframe of results including a field of optimal values and computes the relative error

            error - using MOSEK as a standard, the error in the optimal value
                defined as |value - MOSEK|/(abstol + |MOSEK|)

        Does not alter results if mosek wasn't used to solve the problem.

        Parameters
        ----------
        results : pandas.Panel
            A pandas panel where index = <metric> (e.g. "time", "opt_val", "status", etc.),
            major_axis = <problemID> (e.g. "least_squares_0"), and minor_axis = <configID> (e.g. "mosek_config").
            Contains the results of solving each problem with each solver configuration.
        opt_val : string
            The name of the index in results where the optimal value of the problem under a specific configuration is found.
        mosek_config : string
            The configID for the configuration that used mosek to solve the problems.
        abstol : float
            The absolute tolerance used for computing the error. Added to the denominator to avoid division by zero.
        """
        problemsIndex = results.axes[0].levels[0]
        configsIndex = results.axes[0].levels[1]
        error = pd.Series(index=results.axes[0]) # (problem, config) multiindex
        for configID in configsIndex:
            for problemID in problemsIndex:
                absdiff = np.absolute(
                    (results.loc[(problemID, configID), opt_val] - 
                        results.loc[(problemID, mosek_config), opt_val]))
                absmosek = np.absolute(
                               results.loc[(problemID, mosek_config), opt_val])
                error.loc[(problemID, configID)] = absdiff/(abstol + absmosek)
        results["error"] = error

    @staticmethod
    def compute_performance(results, time, rel_max=10e10):
        """Takes a panel of results including a field of time data and computes the relative performance
        as defined in Dolan, More 2001. "Benchmarking optimization software with performance profiles"

            performance - for each config, for each solver, the time it took for the config to solve the problem,
                divided by the fastest time for any config to solve the problem.

            rel_max is a dummy value that should be larger than any relative performance value. It represents the case
                when the solver does not solve the problem.
        Does not alter results if no time field is recorded in the results already.

        Parameters
        ----------
        results : pandas.Panel
            A pandas panel where index = <metric> (e.g. "time", "opt_val", "status", etc.),
            major_axis = <problemID> (e.g. "least_squares_0"), and minor_axis = <configID> (e.g. "mosek_config").
            Contains the results of solving each problem with each solver configuration.
        time : string
            The name of the index in results where the solve time is stored.
        rel_max : float 
            A dummy value that should be larger than any relative performance value. It represents the case
            when the solver does not solve the problem. Defaults to 10e10.
        """
        problemsIndex = results.axes[0].levels[0]
        configsIndex = results.axes[0].levels[1]
        performance = pd.Series(index=results.axes[0]) # (problem, config) multiindex
        num_problems = 0
        for problem in problemsIndex:
            num_problems += 1
            best = rel_max
            for config in configsIndex:
                # Get best performance for each problem.
                this = results.loc[(problem, config), time]
                if this < best: # also works if this is NaN
                    best = this

            if best == rel_max:
                # No solver could solve this problem.
                print("all solvers failed on {}".format(problem))
                for config in configsIndex:
                    performance.loc[(problem, config)] = rel_max;
                continue


            else: # Compute t/t_best for each problem for each config
                for config in configsIndex:
                    if math.isnan(results.loc[(problem, config), time]):
                        performance.loc[(problem, config)] = rel_max
                    else:
                        performance.loc[(problem, config)] = \
                            results.loc[(problem, config), time]/best

        results["performance"] = performance


testproblemtp = namedtuple("TestProblem", ["problemID", "problem", "tags"])
class TestProblem(testproblemtp):
    """Expands the Problem class to contain extra details relevant to the testing architecture.

    Attributes
    ----------
    id : string
        A unique identifier for this problem.
    problem : cvxpy.Problem
        The cvxpy problem to be solved.
    tags : list
        A list of tags in {LP, SOCP, SDP, EXP, MIP} that describe
        the required solver capabilities to solve this problem.
    """
    def __new__(cls, problemID, problem):
        try:
            tags = cls.get_cone_types(problem)
        except:
            tags = None
        return super(TestProblem, cls).__new__(cls, problemID, problem, tags)


    @classmethod
    def get_all_from_file(cls, fileID, problemDir):
        """Loads a file with name <fileID>.py and returns a list of
        testproblem objects, one for each problem found in the file.

        Parameters
        ----------
        fileID : string
        problemDir : string
            The directory where the problem files are located.
            Each problem file should have a list named "problems" 
            of dicionaries that describe the problem objects.

        Returns
        -------
        a list of cvxbenchmarks.framework.TestProblem objects.
        """

        # Load the module
        if problemDir not in sys.path:
            sys.path.insert(0, problemDir)
        try:
            problemModule = __import__(fileID)
        except Exception as e: # pragma: no cover
            warn("Could not import file " + fileID)
            print(e)
            return []

        foundProblems = [] # Holds the TestProblems we find in this file

        # Look for a dictionary
        PROBLEM_LIST = ["problems"]
        for problemList in PROBLEM_LIST:
            if problemList in [name for name in dir(problemModule)]:
                problems = getattr(problemModule, "problems")
                for problemDict in problems:
                    foundProblems.append(cls.process_problem_dict(**problemDict))
        if len(foundProblems) == 0: # pragma: no cover
            warn(fileID + " contains no problem objects.")
        return foundProblems


    @classmethod
    def process_problem_dict(cls, **problemDict):
        """Unpacks a problem dictionary object of the form:
        {
            "problemID": problemID,
            "problem": prob,
            "opt_val": opt_val
        }

        and returns a single TestPoblem.

        Parameters
        ----------
        **problemDict : **kwargs object containing the above fields.

        Returns
        -------
        TestFramework.TestProblem object.
        """
        return cls(problemDict["problemID"],
                   problemDict["problem"])


    @staticmethod
    def get_cone_types(problem):
        """
        Parameters
        ----------
        problem : cvxpy.Problem
            The problem whose cones we are investigating.

        Returns
        -------
        coneTypes : list of str
            A list of strings (defined in cvxbenchmarks.settings). Contains "LP" by default,
            going through additional cones might add more.
        """

        coneTypes = set([s.LP]) # Better be able to handle these...
        for constr in problem.canonicalize()[1]:
            if isinstance(constr, SDP): # Semidefinite program
                coneTypes.add(s.SDP)
            elif isinstance(constr, ExpCone): # Exponential cone program
                coneTypes.add(s.EXP)
            elif isinstance(constr, SOC): # Second-order cone program
                coneTypes.add(s.SOCP)
            elif isinstance(constr, BoolConstr): # Mixed-integer program
                coneTypes.add(s.MIP)
        return coneTypes

    @staticmethod
    def get_cone_sizes(problem): # pragma: no cover
        """
        Parameters
        ----------
        problem : cvxpy.Problem
            The problem whose cones we are investigating.

        Returns
        -------
        coneSizes : dict
            A dictionary mapping the cone type to the total number of variables constrained to be in that cone.
        """
        return NotImplemented

    def __repr__(self):
        return str(self.problemID) + " " + str(self.problem)

TestProblem.__new__.__defaults__ = (None, None, None)



solverconfigurationtp = namedtuple("SolverConfiguration", ["configID", "config"])
class SolverConfiguration(solverconfigurationtp):
    """An object for managing the configuration of the cvxpy solver.

    Attributes
    ----------
    id : string
        A unique identifier for this configuration.
    config : dict
        A dictionary containing the configuration data. CVXPY's Problem.solve()
        uses the following arguments:

            solver,
            ignore_dcp,
            warm_start,
            verbose,
            parallel

        And any extra arguments are collected into a **kwargs argument.

        For example:
        {
            "solver": "SCS",
            "verbose": True,
            "eps": 1e-4
        }
        would set the solver to be SCS, verbose to be True, and the epsilon
        tolerance of the solver to be 1e-4.
    """

    @classmethod
    def from_file(cls, configID, configDir):
        """Alternative constructor for loading a configuration from a text file.
        Loads a python file named <configID>.py with variables "solver", "verbose",
        and "kwargs".

        Parameters
        ----------
        configID : string
            A unique identifier for this config. Config file should be of the form <configID>.py.
        configDir : string
            The directory where the config files are located.

        Returns
        -------
        SolverConfiguration - the newly created SolverConfiguration object.
        """
        if configDir not in sys.path:
            sys.path.insert(0, configDir)
        configObj = __import__(configID)
        try:
            if configObj.config["solver"] in cvx.installed_solvers():
                return cls(configID, configObj.config)
            else:
                return None
        except: # pragma: no cover
            warn("Could not import configuration: " + configID)
            return None

SolverConfiguration.__new__.__defaults__ = (None, None)


testinstancetp = namedtuple("TestInstance", ["testproblem", "solverconfig"])
class TestInstance(testinstancetp):
    """An object for managing the data collection for a particular problem instance and
    a particular solver configuration.

    Attributes
    ----------

    testproblem : TestFramework.TestProblem
       The problem to be solved.
    solverconfig : TestFramework.SolverConfiguration
       The configuration to use when solving this particular problem instance.

    """

    def run(self):
        """Runs the problem instance against the solver configuration.

        Returns
        -------
        results : cvxbenchmarks.TestResults
            A TestResults instance with the results of running this instance.
        """
        problem = self.testproblem.problem
        problemID = self.testproblem.problemID
        configID = self.solverconfig.configID
        instancehash = hash(self)
        # Record problem size metrics first:
        size_metrics = problem.size_metrics

        try:
            start = time.time() # Time the solve
            print("starting {} with config {}".format(self.testproblem.problemID, self.solverconfig.configID))
            problem.solve(**self.solverconfig.config)
            print("finished solve for {} with config {}".format(self.testproblem.problemID, self.solverconfig.configID))
            if problem.solver_stats.solve_time is not None:
                solve_time = problem.solver_stats.solve_time
            else:
                warn(self.solverconfig.configID + " did not report a solve time for " + self.testproblem.problemID)
                solve_time = time.time() - start
            setup_time = problem.solver_stats.setup_time
            num_iters = problem.solver_stats.num_iters
            status = problem.status
            opt_val = problem.value
        except Exception as e:
            print(e)
            # Configuration could not solve the given problem
            print(("failure solving {} " + 
                   "with config {} " +
                   "in {} sec.").format(problemID, 
                                        configID,
                                        round(time.time()-start, 1)))
            return TestResults(problemID=problemID, configID=configID, 
                                 instancehash=instancehash, size_metrics=size_metrics)

        # Record residual gross stats:
        avg_abs_resid, max_resid = TestResults.compute_residual_stats(problem)
        print("computed stats for {} with config {}".format(problemID, configID))

        print("finished {} with config {} in {} sec.".format(problemID, configID, round(time.time()-start, 1)))
        return TestResults(problemID=problemID, configID=configID,
                             instancehash=instancehash, solve_time=solve_time,
                             setup_time=setup_time, num_iters=num_iters,
                             status=status, opt_val=opt_val,
                             avg_abs_resid=avg_abs_resid, max_resid=max_resid,
                             size_metrics=size_metrics)

    def __eq__(self, other):
        return str(self) == str(other)

    def __hash__(self):
        old_options = np.get_printoptions()
        np.set_printoptions(threshold=10, precision=3) # Shorten the string representation.
        # Replace var<id> with just var and param<id> with just param
        problemString = str(self)
        problemString = re.sub(VARSUB, "var", problemString)
        problemString = re.sub(PARAMSUB, "param", problemString)
        digest = int(hashlib.sha256(problemString.encode("utf-16")).hexdigest(), 16)
        # print(problemString)
        np.set_printoptions(**old_options) # Restore 
        return digest

TestInstance.__new__.__defaults__ = (None, None)


testresultstp = namedtuple("TestResults", 
    ["problemID", "configID", "instancehash",
     "solve_time", "setup_time", "num_iters",
     "status", "opt_val", "avg_abs_resid", "max_resid",
     "size_metrics"])
class TestResults(testresultstp):
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
    size_metrics : cvxpy.SizeMetrics
        A SizeMetrics object holding stats about the size of the problem.

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
            elif isinstance(res, type(None)):
                pass
            else:
                print("Unknown residual type: {}".format(type(res)))

            # Get max absolute residual:
            if max_residual < thismax:
                max_residual = thismax
        if n_residuals == 0:
            return (None, None)
        return (sum_residuals/n_residuals, max_residual)

# 11 entries:
TestResults.__new__.__defaults__ = (None, None, None, None, None, None, None, None,
                                    None, None, None)




