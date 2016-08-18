
import numpy as np
import multiprocessing
import time
import os, sys, inspect, glob
import pandas as pd
import math

STOP = "STOP" # Poison pill for parallel solve subroutine.

# Use local repository:

# cvxfolder = os.path.realpath(os.path.abspath(os.path.join(os.path.split(inspect.getfile(inspect.currentframe()))[0],"cvxpy")))
# if cvxfolder not in sys.path:
    # sys.path.insert(0, cvxfolder) 
sys.path.insert(0, "/Users/mark/Documents/Stanford/reu2016/cvxpy")
import cvxpy as cvx
print cvx


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

    A note on the design: because future problems might become quite large, it might become infeasible
    to pass them directly to the workers via queues. Instead we pass just the file paths and let the worker
    read the problem directly into its own memory space. Because TestResult objects are inherently 
    size-constrained, it's feasible to pass them back as objects directly (also note that they are serializable).
    """
    while True:
        problemID, configID = work_queue.get()
        if problemID == STOP:
            # Poison pill
            print "Exiting worker process."
            break
        testproblem = TestProblem.from_file(problemID, problemDir)
        config = SolverConfiguration.from_file(configID, configDir)
        test_instance = TestInstance(testproblem, config)

        result = test_instance.run()
        done_queue.put(result)
    return 


class TestFramework(object):
    """An object for managing the running of lots of configurations against lots
    of individual problem instances.

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

    def __init__(self, problemDir, configDir, problems = [], configs = []):
        self.problemDir = problemDir
        self.configDir = configDir
        self.problems = problems
        self.configs = configs
        self._instances = []
        self._results = []

    @property
    def instances(self):
        """The problem instances generated by this framework.
        """
        return self._instances

    @property
    def results(self):
        """The results of the problem instances generated by this framework.
        """
        return self._results

    def load_problem(self, problemID):
        """Loads a single problem and appends it to self.problems.

        Parameters
        ----------
        problemID : string
            A unique identifier for the problem to be loaded. File containing the problem
            should be in the format <problemID>.py.
        """
        self.problems.append(TestProblem.from_file(problemID, self.problemDir))

    def preload_all_problems(self):
        """Loads all the problems in self.problemDir and adds them to self.test_problems.
        """
        for dirname, dirnames, filenames in os.walk(self.problemDir):
            for filename in filenames:
                if filename[-3:] == ".py" and filename != "__init__.py":
                    problemID = filename[0:-3]
                    self.load_problem(problemID)

    def load_config(self, configID):
        """Loads a single solver configuration and appends it to self.configs

        Parameters
        ----------
        configID : string
            A unique identifier for the solver configuration to be loaded. File containing
            the configuration should be in the form <configID>.py.
        """
        self.configs.append(SolverConfiguration.from_file(configID, self.configDir))

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
                self._instances.append(TestInstance(problem, config))

    def solve_all(self):
        """Solves all test instances and reports the results.
        """
        self.generate_test_instances()
        self._results = []
        for instance in self._instances:
            self._results.append(instance.run())

    def solve_all_parallel(self):
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
        for instance in self._instances:
            work_queue.put((instance.testproblem.id, instance.config.id))

        # start processes
        for w in xrange(workers):
            p = multiprocessing.Process(target=worker,
                                        args=(self.problemDir,
                                              self.configDir,
                                              work_queue,
                                              done_queue))
            p.start()
            processes.append(p)
            work_queue.put((STOP,STOP))

        # wait until all processes finished
        for p in processes:
            p.join()

        done_queue.put(STOP)

        # beautify results and return them
        for result in iter(done_queue.get, STOP):
            if result is not None:
                self._results.append(result)


        for p in processes:
            print "process",p,"exited with code",p.exitcode

    def export_results_as_panel(self):
        """Convert results into a pandas panel object for easier data visualization

        Returns
        -------
        output : pandas.Panel
            A panel containing the results of the testing.
        """
        problemIDs = [problem.id for problem in self.problems]
        configIDs = [config.id for config in self.configs]

        # Make a dummy TestResults instance to generate labels:
        dummy = TestResults(TestProblem(None,None), SolverConfiguration(None, None, None, None))
        attributes = inspect.getmembers(dummy, lambda a: not(inspect.isroutine(a)))
        labels = [label[0] for label in attributes if not(label[0].startswith('__') and label[0].endswith('__'))]
        # Unpack size_metrics label with another dummy
        dummy = cvx.Problem(cvx.Minimize(cvx.Variable())).size_metrics
        attributes = inspect.getmembers(dummy, lambda a: not(inspect.isroutine(a)))
        size_metrics_labels = [label[0] for label in attributes if not(label[0].startswith('__') and \
                                                                       label[0].endswith('__'))]

        labels += size_metrics_labels

        # Remove unused columns
        labels.remove("size_metrics")
        labels.remove("test_problem")
        labels.remove("config")

        output = pd.Panel(items = labels, major_axis = problemIDs, minor_axis = configIDs)
        for result in self._results:
            result_dict = result.__dict__

            # Unpack the size_metrics object inside it:
            sizemetrics_dict = result_dict["size_metrics"].__dict__
            del(result_dict["size_metrics"])

            result_dict.update(sizemetrics_dict)

            problemID = result_dict["test_problem"]
            del(result_dict["test_problem"])
            configID = result_dict["config"]
            del(result_dict["config"])

            for key, value in result_dict.items():
                output.loc[key, problemID, configID] = value

        # Compute Statistics
        try:
            TestFramework.compute_mosek_error(output, "opt_val", "mosek_config")
        except (KeyError):
            print "TestFramework.compute_mosek_error: 'mosek_config' or 'opt_val' field not found."
        try:
            TestFramework.compute_performance(output, "solve_time")
        except (KeyError):
            print "TestFramework.compute_performance: 'solve_time' field not found."
        return output

    @classmethod
    def compute_mosek_error(self, results, opt_val, mosek_config):
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
        """
        abstol = 10e-4
        error = pd.DataFrame(index = results.axes[1], columns = results.axes[2])
        for configID in results.axes[2]:
            for problemID in results.axes[1]:
                absdiff = np.absolute((results.loc[opt_val, problemID, configID] - results.loc[opt_val, problemID, mosek_config]))
                absmosek = np.absolute(results.loc[opt_val, problemID, mosek_config])
                error.loc[problemID, configID] = absdiff/(abstol + absmosek)
                results["error"] = error

    @classmethod
    def compute_performance(self, results, time, rel_max = 10e10):
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
        performance = pd.DataFrame(index = results.axes[1], columns = results.axes[2])
        num_problems = 0
        for problem in results.axes[1]:
            num_problems += 1
            best = rel_max
            for config in results.axes[2]:
                # Get best performance for each problem.
                this = results.loc[time, problem, config]
                if this < best: # also works if this is NaN
                    best = this

            if best == rel_max:
                # No solver could solve this problem.
                print "all solvers failed on",problem
                for config in results.axes[2]:
                    performance.loc[problem, config] = rel_max;
                continue


            else: # Compute t/t_best for each problem for each config
                for config in results.axes[2]:
                    if math.isnan(results.loc[time, problem, config]):
                        performance.loc[problem, config] = rel_max
                    else:
                        performance.loc[problem, config] = results.loc[time, problem, config]/best

        results["performance"] = performance


class TestProblem(object):
    """Expands the Problem class to contain extra details relevant to the testing architecture.

    Attributes
    ----------
    id : string
        A unique identifier for this problem.
    problem : cvxpy.Problem
        The cvxpy problem to be solved.
    """
    def __init__(self, problemID, problem):
        self.id = problemID
        self.problem = problem

    @classmethod
    def from_file(self, problemID, problemDir):
        """Alternative constructor for loading a problem from a directory containing a
        <problemID>.py file defining a cvxpy.Problem instance named prob.

        Parameters
        ----------
        problemID : string
            A unique identifier for this problem. Problem file should be of the form <problemID>.py.
        problemDir : string
            The directory where the problem files are located.

        Returns
        -------
        TestProblem - the newly created TestProblem object.
        """
        if problemDir not in sys.path:
            sys.path.insert(0, problemDir)
        return TestProblem(problemID, __import__(problemID).prob)

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
    def from_file(self, configID, configDir):
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
        return SolverConfiguration(configID, configObj.solver, configObj.verbose, configObj.kwargs)

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
        results = TestResults(self.testproblem, self.config)
        # results = Test
        try:
            start = time.time() # Time the solve
            print "starting",self.testproblem.id,"with config",self.config.id,"at",start
            problem.solve(solver = self.config.solver, verbose = self.config.verbose, **self.config.kwargs)
            print "finished solve for", self.testproblem.id, "with config", self.config.id
            if problem.solver_stats.solve_time is not None:
                results.solve_time = problem.solver_stats.solve_time
            else:
                results.solve_time = time.time() - start
            if problem.solver_stats.setup_time is not None:
                results.setup_time = problem.solver_stats.setup_time
            if problem.solver_stats.num_iters is not None:
                results.num_iters = problem.solver_stats.num_iters
            results.status = problem.status
            results.opt_val = problem.value
        except:
            # Configuration could not solve the given problem
            print "failure in solving."
        # Record residual gross stats:
        results.avg_abs_resid, results.max_resid = TestInstance.compute_residual_stats(problem)
        print "computed stats for", self.testproblem.id, "with config", self.config.id
        # Record problem metrics:
        results.size_metrics = problem.size_metrics

        print "finished",self.testproblem.id,"with config",self.config.id,"at",time.time()-start
        return results

    @classmethod
    def compute_residual_stats(self, problem):
        """Computes the average absolute residual and the maximum residual
        of the current problem.

        Returns:
        --------
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
                pass
            else:
                print "Unknown residual type:", type(res)

            # Get max absolute residual:
            if max_residual < thismax:
                max_residual = thismax
        if n_residuals == 0:
            return (None, None)
        return (sum_residuals/n_residuals, max_residual)


class TestResults(object):
    """Holds the results of running a test instance.

    Attributes
    ----------
    test_problem : TestFramework.TestProblem
        The problem used to generate these results.
    config : TestFramework.SolverConfiguration
        The configuration used to solve the problem.
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

    def __init__(self, test_problem, config):
        self.test_problem = test_problem.id
        self.config = config.id
        self.solve_time = None
        self.setup_time = None
        self.num_iters = None
        self.status = None
        self.opt_val = None
        self.avg_abs_resid = None
        self.max_resid = None
        self.size_metrics = None







