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







