
import numpy as np
import multiprocessing
import time
import os, sys, inspect, glob
import pandas as pd

STOP = "STOP"

# Use local repository:
cvxfolder = os.path.realpath(os.path.abspath(os.path.join(os.path.split(inspect.getfile(inspect.currentframe()))[0],"cvxpy")))
if cvxfolder not in sys.path:
    sys.path.insert(0, cvxfolder)
import cvxpy as cvx
print cvx


def worker(problemDir, configDir, work_queue, done_queue):
    """Worker function for multithreading the solving of test instances.
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
        Currently, this directory must be a subdirectory of the current frame.
    configDir : string
        Directory containing desired config files.
        Currently, this directory must be a subdirectory of the current frame.

    problems : list of TestFramework.TestProblem
        list of problems to solve.
    configs : list of TestFramework.SolverConfiguration
        list of configurations to solve the problems under
    instances : list of TestFramework.TestInstance
        list of test instances to run.
    results : list of TestFramework.TestResults
        list of results from running each test instance.

    
    Workflow:
    Read in problems (TestProblem)
        Optional: Use index to filter for problems.
    Read in solver configurations (SolverConfiguration)
    Generate TestInstances
    Run all TestInstances (possibly in parallel?) and record Results

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
        """Loads a single problem and appends it to self.problems
        """
        self.problems.append(TestProblem.from_file(problemID, self.problemDir))


    def preload_all_problems(self):
        """Loads all the problems in self.problemDir and adds them to self.test_problems
        """
        for dirname, dirnames, filenames in os.walk(self.problemDir):
            for filename in filenames:
                # print filename
                if filename[-3:] == ".py" and filename != "__init__.py":
                    problemID = filename[0:-3]
                    self.load_problem(problemID)

    def load_config(self, configID):
        """Loads a single solver configuration and appends it to self.configs
        """
        self.configs.append(SolverConfiguration.from_file(configID, self.configDir))

    def preload_all_configs(self):
        """Loads all the configs in self.configDir and adds them to self.configs
        """
        for dirname, dirnames, filenames in os.walk(self.configDir):
            for filename in filenames:
                # print filename
                if filename[-3:] == ".py" and filename != "__init__.py":
                    configID = filename[0:-3]
                    self.load_config(configID)

    def generate_test_instances(self):
        """Generates a test problem for every pair of (problem, config)
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
        CONTAINS BUGS! DO NOT USE! ESPECIALLY WHEN USING THE CVXOPT SOLVER!
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
            p = multiprocessing.Process(target=worker, args=(self.problemDir, self.configDir, work_queue, done_queue))
            p.start()
            processes.append(p)
            work_queue.put((STOP,STOP))

        # wait until all processes finished
        for p in processes:
            p.join(timeout = 5)

        done_queue.put(STOP)

        # beautify results and return them
        for result in iter(done_queue.get, STOP):
            if result is not None:
                self._results.append(result)


        for p in processes:
            print "process",p,"exited with code",p.exitcode


    def export_results_as_panel(self):
        """Convert results into a pandas panel object for easier data visualization"""
        problemIDs = [problem.id for problem in self.problems]
        print problemIDs
        configIDs = [config.id for config in self.configs]
        print configIDs

        # Make a dummy TestResults instance to generate labels:
        dummy = TestResults(TestProblem(None,None), SolverConfiguration(None, None, None, None))
        attributes = inspect.getmembers(dummy, lambda a: not(inspect.isroutine(a)))
        labels = [label[0] for label in attributes if not(label[0].startswith('__') and label[0].endswith('__'))]
        # Unpack size_metrics label with another dummy
        dummy = cvx.Problem(cvx.Minimize(cvx.Variable())).size_metrics
        attributes = inspect.getmembers(dummy, lambda a: not(inspect.isroutine(a)))
        size_metrics_labels = [label[0] for label in attributes if not(label[0].startswith('__') and label[0].endswith('__'))]
       
        labels += size_metrics_labels

        # Remove unused columns
        labels.remove("size_metrics")
        labels.remove("test_problem")
        labels.remove("config")
        print labels

        output = pd.Panel(items = labels, major_axis = problemIDs, minor_axis = configIDs)
        print output.to_frame()
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
        # error - using MOSEK as a standard, the error in the optimal value
        #     defined as |value - MOSEK|/(abstol + |MOSEK|)
        abstol = 10e-4
        error = pd.DataFrame(index = problemIDs, columns = configIDs)
        for configID in configIDs:
            for problemID in problemIDs:
                absdiff = np.absolute((output.loc["opt_val", problemID,configID] - output.loc["opt_val", problemID, "mosek_config"]))
                absmosek = np.absolute(output.loc["opt_val", problemID,"mosek_config"])
                error.loc[problemID, configID] = absdiff/(abstol + absmosek)
        output["error"] = error
        
        print output.axes

        return output


class TestProblem(object):
    """Expands the Problem class to contain extra details relevant to the testing architecture.
    
    Attributes
    ----------
    id : string
        A unique identifier for this problem.
    """
    def __init__(self, problemID, problem):
        self.id = problemID
        self.problem = problem

    @classmethod
    def from_file(self, problemID, problemDir):
        """Alternative constructor for loading a problem from a directory containing a
        <problemID>.py file defining a cvxpy.Problem instance named prob.
        """
        cmd_folder = os.path.realpath(os.path.abspath(os.path.join(os.path.split(inspect.getfile(inspect.currentframe()))[0], problemDir)))
        if cmd_folder not in sys.path:
            sys.path.insert(0, cmd_folder)
        return TestProblem(problemID, __import__(problemID).prob)
        


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
        """
        cmd_folder = os.path.realpath(os.path.abspath(os.path.join(os.path.split(inspect.getfile(inspect.currentframe()))[0], configDir)))
        if cmd_folder not in sys.path:
            sys.path.insert(0, cmd_folder)
        configObj = __import__(configID)
        return SolverConfiguration(configID, configObj.solver, configObj.verbose, configObj.kwargs)




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
        time : float
            The time (in sec) it took to solve the problem.
            TODO : Split into setup_time and solve_time?
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
        """
        problem = self.testproblem.problem
        results = TestResults(self.testproblem, self.config)
        # results = Test
        try:
            start = time.time() # Time the solve
            print "starting",self.testproblem.id,"with config",self.config.id,"at",start
            problem.solve(solver = self.config.solver, verbose = self.config.verbose, **self.config.kwargs)
            print "finished solve for", self.testproblem.id, "with config", self.config.id
            results.time = time.time() - start
            results.status = problem.status
            results.opt_val = problem.value
        except:
            # Configuration could not solve the given problem
            print "failure in solving."
        # Record residual gross stats:
        results.avg_abs_resid, results.max_resid = TestInstance.computeResidualStats(problem)
        print "computed stats for", self.testproblem.id, "with config", self.config.id
        # Record problem metrics:
        results.size_metrics = problem.size_metrics
        
        print "finished",self.testproblem.id,"with config",self.config.id,"at",time.time()-start
        return results
    
    @classmethod
    def computeResidualStats(self, problem):
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
    time : float
        The time (in sec) it took to solve the problem.
        TODO : Split into setup_time and solve_time?
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
        self.time = None
        self.status = None
        self.opt_val = None
        self.avg_abs_resid = None
        self.max_resid = None
        self.size_metrics = None










