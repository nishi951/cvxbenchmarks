import numpy as np
import multiprocessing, signal
import time
import os, sys, inspect, glob, re
import pandas as pd
import math

import hashlib, pickle as pkl

from collections import namedtuple

import cvxbenchmarks.settings as s
from cvxbenchmarks.cvx.cvxproblem import CVXProblem
from cvxbenchmarks.cvx.cvxconfig import CVXConfig
from cvxbenchmarks.cvx.cvxcore import CVXInstance, CVXResults

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


Tickettp = namedtuple("Ticket", ["problemID", "problemDir", "configID", "configDir"])
class Ticket(Tickettp):
    """Class for storing the relevant info for retrieving one or more problem instances from a file,
    without actually needing to load the instance into memory.
    """
    def serve(cls, Problem, Config, Instance):
        """
        Parameters
        ----------
        Problem : class
            The class (e.g. CVXProblem) of the problem described by |problemDir| and |problemID|.
        Config : class
            The class (e.g. CVXConfig) of the config described by |configDir| and |configID|.
        Instance : class
            The class (e.g. CVXInstance) of the instance corresponding to Problem and Config

        Returns
        -------
        A list of |Instance|s
        """
        problems = Problem.read(self.problemID, self.problemDir)
        config = Config.read(os.path.join(self.configDir, self.configID))
        return [Instance(problem, config) for problem in problems]


Ticket.__new__.__defaults__ = (None, None, None, None)



class CVXFramework(object):
    """An object for managing the running of lots of configurations 
    against lots of individual problem instances.

    Attributes
    ----------
    problems : list of (problemID, problemDir)
        list of tuples providing info about how to load the problems to solve.
    configs : list of (configID, configDir)
        list of tuples providing info about how to load the 
        configurations under which to solve the problems.
    cache : Cache object
        Cache for storing Results
    tickets : list of |Ticket| objects
        List of |Ticket| objects, each describing an Instance to solve
    results : list of Results
        list of results from running each Instance.

    Examples
    --------
    framework = CVXFramework()
    framework.load_all_problems(problemDir) # Load list of (problemDir, problemID)
    framework.load_all_configs(configDir)   # Load list of (configDir, configID)
    framework.generate_tickets()            # Create |Ticket| objects
    framework.solve()                       # Load and solve each ticket

    framework = CVXFramework()
    framework.load_all_problems(problemDir) # Load list of (problemDir, problemID)
    framework.load_all_configs(configDir)   # Load list of (configDir, configID)
    framework.generate_tickets()            # Create |Ticket| objects
    framework.solve(filter)                 # Load and solve each ticket, solving only
                                            # those that match a given filter



    """

    def __init__(self, problems=None, configs=None,
                 instances=None, results=None, cache=None):
        if problems is None:
            self.problems = []
        else:
            self.problems = problems
        if configs is None:
            self.configs = []
        else:
            self.configs = configs

        # Check stuff
        assert all(len(prob) == 2 for prob in self.problems) # should be 2-tuples
        assert all(len(conf) == 2 for conf in self.configs) # should be 2-tuples

        # Properties
        if instances is None:
            self.instances = []
        else:
            self.instances = instances
        if results is None:
            self.results = []
        else:
            self.results = results

        self.cache = cache



    def load_problem(self, problemID, problemDir):
        """Loads a single problem file and appends all problems
        in it to self.problems.

        Parameters
        ----------
        problemID : string
            A unique identifier for the file to be loaded. File 
            containing the problem should be in the format <problemID>.py.
            <problemID>.py can also contain a list of problems.
        problemDir : string
            The directory in which the problem resides.

        Appends a tuple of (problemID, problemDir) to |self.problems|.

        """
        # self.problems.extend(CVXProblem.read(fileID, problemDir))
        self.problems.append((problemID, problemDir))

    def load_all_problems(self, problemDir):
        """Loads all the problems in |problemDir| and adds them to 
        self.problems.
        """
        nfound = 0
        for _, _, filenames in os.walk(problemDir):
            for filename in filenames:
                if filename[-3:] == ".py" and filename != "__init__.py":
                    nfound += 1
                    self.load_problem(os.path.splitext(filename)[0], problemDir)
        print("Found {} problems in {}.".format(nfound, problemDir))

    def load_config(self, configID, configDir):
        """Loads a single solver configuration, checking if 
           cvxpy supports it:

        Parameters
        ----------
        configID : string
            A unique identifier for the solver configuration to be 
            loaded.
        configDir : string
            The directory in which the config file resides.

        Appends a tuple of (configID, configDir) to |self.configs|
        """
        # config = CVXConfig.read(configID, self.configDir)
        # if config is not None:
        #     self.configs.append(solverconfig)
        # else:
        #     warn(UserWarning("{} configuration specified but not installed.".format(configID)))
        self.configs.append((configID, configDir))

    def load_all_configs(self, configDir):
        """Loads all the configs in configDir
        """
        for _, _, filenames in os.walk(configDir):
            for filename in filenames:
                if filename[-4:] == ".yml" or filename[-5:] == ".yaml":
                    # Remove extension
                    configID = os.path.splitext(filename)[0]
                    self.load_config(configID, configDir)

    def generate_tickets(self):
        """Generates a |Ticket| for every pair of (problem, config).
        """
        for problem in self.problems:
            for config in self.configs:
                self.instances.append(Ticket(*(problem + config)))

    def clear_cache(self): # pragma: no cover
        """Clear the cache used to store TestResults
        """
        # Overwite with an empty dictionary
        with open(self.cacheFile, "wb") as f:
            pkl.dump({}, f)
        return

    def solve(self, use_cache=True, parallel=True):
        """Solve all the TestInstances we have queued up.

        Parameters
        ----------
        use_cache : boolean
            Whether or not we should use the cache specified in self.cacheFile
        """
        if parallel:
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
