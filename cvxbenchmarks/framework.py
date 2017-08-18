import numpy as np
import multiprocessing
import time
import os, sys, inspect, glob
import pandas as pd
import math

import hashlib, pickle as pkl

import cvxbenchmarks.settings as s

# Constraint types
from cvxpy.constraints.semidefinite import SDP
from cvxpy.constraints.exponential import ExpCone
from cvxpy.constraints.second_order import SOC
from cvxpy.constraints.bool_constr import BoolConstr

from warnings import warn

STOP = "STOP" # Poison pill for parallel solve subroutine.

# Use local repository:

# cvxfolder = os.path.realpath(os.path.abspath(os.path.join(os.path.split(inspect.getfile(inspect.currentframe()))[0],"cvxpy")))
# if cvxfolder not in sys.path:
    # sys.path.insert(0, cvxfolder) 
# sys.path.insert(0, "/Users/mark/Documents/Stanford/reu2016/cvxpy")
import cvxpy as cvx
print(cvx)


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

    To return the result, we need to 
    """
    while True:
        problemID, configID = work_queue.get()
        if problemID == STOP:
            # Poison pill
            print("Exiting worker process.")
            done_queue.put(STOP)
            break
        testproblemList = TestProblem.get_all_from_file(problemID, problemDir)
        config = SolverConfiguration.from_file(configID, configDir)
        for testproblem in testproblemList:
            test_instance = TestInstance(testproblem, config)
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
                 problems=None, configs=None, cacheFile="cache.pkl", 
                 parallel=False, tags=None, instances=None, results=None):
        self.problemDir = problemDir
        self.configDir = configDir
        if problems is None:
            self.problems = []
        else:
            self.problems = problems
        if configs is None:
            self.configs = []
        else:
            self.configs = configs
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
        in it to self.problems.

        Parameters
        ----------
        fileID : string
            A unique identifier for the file to be loaded. File 
            containing the problem should be in the format <fileID>.py.
            <fileID>.py can also contain a list of problems.
        """
        self.problems.extend(TestProblem.get_all_from_file(fileID, self.problemDir))

    def preload_all_problems(self):
        """Loads all the problems in self.problemDir and adds them to 
        self.test_problems.
        """
        for dirname, dirnames, filenames in os.walk(self.problemDir):
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
        config = SolverConfiguration.from_file(configID, self.configDir)
        if config is not None:
            self.configs.append(config)
        else:
            warn(UserWarning("{} configuration specified but not installed.".format(configID)))

    def preload_all_configs(self):
        """Loads all the configs in self.configDir and adds them to self.configs.
        """
        for dirname, dirnames, filenames in os.walk(self.configDir):
            for filename in filenames:
                if filename[-3:] == ".py" and filename != "__init__.py":
                    configID = filename[0:-3]
                    self.load_config(configID)

    def generate_test_instances(self):
        """Generates a test problem for every pair of (problem, config).
        """
        for problem in self.problems:
            for config in self.configs:
                self.instances.append(TestInstance(problem, config))

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
            except: # pragma: no cover
                print("Creating new cache file: {}".format(self.cacheFile))
            with open(self.cacheFile, "wb") as f: # Overwrite previous cache
                for instance in self.instances:
                    instancehash = hash(instance)
                    if instancehash in cachedResults:
                        # Retrieve TestResult from the results dictionary:
                        self.results.append(cachedResults[instancehash])
                        print(("Retrieved instance result ({}, {}) " +
                               "from cache.").format(instance.testproblem.id,
                                                     instance.config.id))
                    else:
                        # Add this result to the cache
                        result = instance.run()
                        self.results.append(result)
                        cachedResults[instancehash] = result
                # Write the modified dictionary back to the cache file.
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
            except:
                print("Creating new cache file: {}".format(self.cacheFile))
            with open(self.cacheFile, "wb") as f:
                for instance in self.instances:
                    instancehash = hash(instance)
                    if instancehash in cachedResults:
                        # Retrieve TestResult from the results dictionary:
                        self.results.append(cachedResults[instancehash])
                    else:
                        # Add this result to the cache
                        work_queue.put((instance.testproblem.id, instance.config.id))

        else:
            for instance in self.instances:
                print((instance.testproblem.id, instance.config.id))
                work_queue.put((instance.testproblem.id, instance.config.id))

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
                # print "received!"
            time.sleep(0.5) # Wait for processes to run.
            print("waiting...")

        if use_cache: # Add all results to the cache.
            with open(self.cacheFile, "wb") as f:
                for result in self.results:        
                    cachedResults[result.instancehash] = result
                pkl.dump(cachedResults, f)

        for p in processes:
            print("process {} exited with code {}".format(p,p.exitcode))
        return

    def export_results(self):
        """Convert results into a pandas (multiindex) dataframe object for easier data visualization

        Returns
        -------
        output : pandas.DataFrame
            A panel containing the results of the testing.
        """
        problemIDs = [problem.id for problem in self.problems]
        configIDs = [config.id for config in self.configs]

        # Make a dummy TestResults instance to generate labels:
        dummy = TestResults(None)
        attributes = inspect.getmembers(dummy, lambda a: not(inspect.isroutine(a)))
        labels = [label[0] for label in attributes if not(label[0].startswith('__') and label[0].endswith('__'))]
        # Unpack size_metrics label with another dummy
        dummy = cvx.Problem(cvx.Minimize(cvx.Variable())).size_metrics
        attributes = inspect.getmembers(dummy, lambda a: not(inspect.isroutine(a)))
        size_metrics_labels = [label[0] for label in attributes if not(label[0].startswith('__') and \
                                                                       label[0].endswith('__'))]

        labels += size_metrics_labels

        # Remove unused (i.e. redundant) columns
        labels.remove("size_metrics")
        labels.remove("test_problem")
        labels.remove("config")

        # output = pd.Panel(items=labels, major_axis=problemIDs, minor_axis=configIDs)
        multiindex = pd.MultiIndex.from_product([problemIDs, configIDs])
        s = pd.DataFrame(index = multiindex, columns = labels)

        for result in self.results:
            result_dict = result.__dict__

            # Unpack the size_metrics object inside it:
            sizemetrics_dict = result_dict["size_metrics"].__dict__
            del(result_dict["size_metrics"])

            result_dict.update(sizemetrics_dict)

            problemID = result_dict["test_problem"]
            del(result_dict["test_problem"])
            configID = result_dict["config"]
            del(result_dict["config"])

            for key, value in list(result_dict.items()):
                output.loc[(problemID, configID), key] = value

        # Compute Statistics
        try:
            TestFramework.compute_mosek_error(output, "opt_val", "mosek_config")
        except (KeyError):
            print("TestFramework.compute_mosek_error: 'mosek_config' or 'opt_val' field not found.")
        try:
            TestFramework.compute_performance(output, "solve_time")
        except (KeyError):
            print("TestFramework.compute_performance: 'solve_time' field not found.")
        return output

    @classmethod
    def compute_mosek_error(self, results, opt_val, mosek_config, abstol=10e-4):
        """Takes a panel of results including a field of optimal values and computes the relative error

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
        error = pd.DataFrame(index=problemsIndex, # Problems 
                             columns=configsIndex) # Configs
        for configID in configsIndex:
            for problemID in problemsIndex:
                absdiff = np.absolute(
                    (results.loc[(problemID, configID), opt_val] - 
                        results.loc[(problemID, mosek_config), opt_val]))
                absmosek = np.absolute(
                               results.loc[(problemID, mosek_config), opt_val])
                error.loc[problemID, configID] = absdiff/(abstol + absmosek)
        results["error"] = error

    @classmethod
    def compute_performance(self, results, time, rel_max=10e10):
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
        performance = pd.DataFrame(index=problemsIndex, # Problems 
                                   columns=configsIndex) # Configs
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
                for config in results.axes[2]:
                    performance.loc[problem, config] = rel_max;
                continue


            else: # Compute t/t_best for each problem for each config
                for config in results.axes[2]:
                    if math.isnan(results.loc[(problem, config), time]):
                        performance.loc[problem, config] = rel_max
                    else:
                        performance.loc[problem, config] = \
                            results.loc[(problem, config), time]/best

        results["performance"] = performance



class TestProblem(object):
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
    def __init__(self, problemID, problem):
        self.id = problemID
        self.problem = problem
        self.tags = TestProblem.check_cone_types(problem)

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
        except Exception as e:
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
                    foundProblems.append(cls.processProblemDict(**problemDict))

        if len(foundProblems) == 0:
            warn(fileID + " contains no problem objects.")
        return foundProblems


    @classmethod
    def processProblemDict(cls, **problemDict):
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
        return cls(problemDict["problemID"], problemDict["problem"])


    @staticmethod
    def check_cone_types(problem):
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

    def __repr__(self):
        return str(self.id) + ": " + str(self.problem)

    def __eq__(self, other):
        return self.id == other.id and self.problem == other.problem


class SolverConfiguration(object):
    """An object for managing the configuration of the cvxpy solver.

    Attributes
    ----------
    id : string
        A unique identifier for this configuration.
    solver : string
        The name of the solver for which we are creating the configuration.
    verbose : boolean
        True if we want to capture the solver output, false otherwise.
    kwargs : dictionary
        Specifies the keyword arguments for the specific solver we are using.
    """

    def __init__(self, configID, solver, verbose, kwargs):
        self.id = configID
        self.solver = solver
        self.verbose = verbose
        self.kwargs = kwargs

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
        if configObj.solver in cvx.installed_solvers():
            return cls(configID, configObj.solver, configObj.verbose, configObj.kwargs)
        else:
            return None

    def __repr__(self):
        return str((str(self.id), str(self.solver), str(self.verbose), str(self.kwargs)))

    def __eq__(self, other):
        return (self.id == other.id) and \
               (self.solver == other.solver) and \
               (self.verbose == other.verbose) and \
               (self.kwargs == other.kwargs)


class TestInstance(object):
    """An object for managing the data collection for a particular problem instance and
    a particular solver configuration.

    Attributes
    ----------

    problem : TestFramework.TestProblem
       The problem to be solved.
    config : TestFramework.SolverConfiguration
       The configuration to use when solving this particular problem instance.

    Results: dictionary
        Contains the following keys:
        solve_time :

        status : string
            The status of the problem after solving, reported by the problem itself. 
            (e.g. optimal, optimal_inaccurate, unbounded, etc.)
        opt_val : float
            The optimal value of the problem, as determined by the given solver configuration.
        avg_abs_resid : float
            The average absolute residual across all scalar problem constraints.
        max_resid : float
            The maximum absolute residual across all scalar problem constraints.
        size_metrics : cvxpy.SizeMetrics
            An object containing various metrics regarding the scale of the problem.

    """

    def __init__(self, testproblem, config):
        self.testproblem = testproblem
        self.config = config

    def run(self):
        """Runs the problem instance against the solver configuration.

        Returns
        -------
        TestResults - A TestResults instance with the results of running this instance.
        """
        problem = self.testproblem.problem
        results = TestResults(self)
        # results = Test
        try:
            start = time.time() # Time the solve
            print("starting {} with config {}".format(self.testproblem.id, self.config.id))
            problem.solve(solver=self.config.solver, verbose=self.config.verbose, **self.config.kwargs)
            print("finished solve for {} with config {}".format(self.testproblem.id, self.config.id))
            if problem.solver_stats.solve_time is not None:
                results.solve_time = problem.solver_stats.solve_time
            else:
                warn(self.config.id + " did not report a solve time for " + self.testproblem.id)
                results.solve_time = time.time() - start
            if problem.solver_stats.setup_time is not None:
                results.setup_time = problem.solver_stats.setup_time
            if problem.solver_stats.num_iters is not None:
                results.num_iters = problem.solver_stats.num_iters

            results.status = problem.status
            results.opt_val = problem.value
        except Exception as e:
            print(e)
            # Configuration could not solve the given problem
            results = TestResults(self)
            results.size_metrics = problem.size_metrics
            print(("failure solving {} " + 
                   "with config {} " +
                   "in {} sec.").format(self.testproblem.id, 
                                        self.config.id,
                                        round(time.time()-start, 1)))
            return results

        # Record residual gross stats:
        results.avg_abs_resid, results.max_resid = TestInstance.compute_residual_stats(problem)
        print("computed stats for {} with config {}".format(self.testproblem.id, self.config.id))

        # Record problem metrics:
        results.size_metrics = problem.size_metrics

        print("finished {} with config {} in {} sec.".format(self.testproblem.id, self.config.id, round(time.time()-start, 1)))
        return results


    def __repr__(self):
        return str((str(self.testproblem), str(self.config)))

    def __eq__(self, other):
        return str(self) == str(other)

    def __hash__(self):
        np.set_printoptions(threshold=10, precision=3) # Shorten the string representation.
        digest = int(hashlib.sha256(str(self).encode("utf-16")).hexdigest(), 16)
        np.set_printoptions(threshold=1000, precision = 8) # Restore defaults
        return digest

class TestResults(object):
    """Holds the results of running a test instance.

    Attributes
    ----------
    test_instance : TestFramework.TestInstance
        The hash of the test instance that generated this
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
        An object containing various metrics regarding the scale of the problem.

    """

    def __init__(self, test_instance):
        if test_instance is not None:
            self.test_problem = test_instance.testproblem.id
            self.config = test_instance.config.id
            self.instancehash = hash(test_instance)
        else:
            self.test_problem = None
            self.config = None
            self.instancehash = None
        self.solve_time = None
        self.setup_time = None
        self.num_iters = None
        self.status = None
        self.opt_val = None
        self.avg_abs_resid = None
        self.max_resid = None
        self.size_metrics = None


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





