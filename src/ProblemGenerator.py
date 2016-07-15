
from jinja2 import Environment, FileSystemLoader
import re, os, sys, inspect
import numpy as np

cvxfolder = os.path.realpath(os.path.abspath(os.path.join(os.path.split(inspect.getfile(inspect.currentframe()))[0],"cvxpy")))
if cvxfolder not in sys.path:
    sys.path.insert(0, cvxfolder)
import cvxpy as cvx
print cvx

# Format of the Parameter file:
# One tuple at the beginning containing the names of the parameters.
# After that, one tuple for each problem instance that the person wants to generate.

# Example:
# (m, n, seed) = (20, 30, 1), (3, 300, 1), (4, 4000, 1), etc.

PARAM_REGEX = "(\((?:\w+)(?:,\s*\w+)*?\))"
PARAM_PATTERN = re.compile(PARAM_REGEX)


class ProblemTemplate(object):
    """A templated version of a cvxpy problem that can be read and written.

    Attributes
    ----------
    problemID : string
        A unique identifier for this problem. The name of the template file is by default 
        <problemID>.j2
    paramFile : string
        The file path to the file containing the parameters for the instances of this template.
        
        The format of the parameter files is:
        - One tuple at the beginning containing the names of the parameters.
        - After that, one tuple for each problem instance that the person wants to generate.

        Example:
        (m, n, seed) = (20, 30, 1), (3, 300, 1), (4, 4000, 1)

        will generate three instances of a template with fields "m", "n" and "seed" with the
        corresponding values (20, 30, 1), (3, 300, 1), and (4, 4000, 1)

    template : jinja2.template
        The problem template that can be rendered to produce the problem instances, after filling
        the parameters specified in params.
    params : list of dictionaries
        A list of dictionaries, each one representing an instance of the problem template.
        The data in the dictionaries corresponds to the fields in the template that 
        need to be filled in.
    """

    def __init__(self, problemID, paramFile, templateDir = "templates"):
        self.problemID = problemID
        # self.templateFile = templateFile
        self.paramFile = paramFile
        self.template, self.params = self.read(templateDir)

    def read_template(self, templateDir):
        """Read in the template.
        """
        temp = None
        try:
            env = Environment(loader = FileSystemLoader(templateDir))
            temp = env.get_template(self.problemID + ".j2")
            return temp
        except Exception as e:
            print "Problem locating template ",self.problemID+".j2 in",templateDir+". Check template file path."
            print e
            return temp

    def read_params(self):
        """Read in the parameters.
        """
        paramDicts = []
        paramList = []
        f = None
        try:
            with open(self.paramFile, "r") as f:
                paramList = re.findall(PARAM_PATTERN, f.readline())
                # Remove parentheses, spaces, and split on commas
                paramNames = paramList[0].strip("()").replace(" ", "").split(",")
        except Exception as e:
            print "Problem loading parameters in",self.paramFile,". Check parameter file path."
            print e
            return None
        for instance in paramList[1:]:
            paramVals = instance.strip("()").replace(" ", "").split(",")
            try:
                paramDicts.append(dict(zip(paramNames, paramVals)))
            except Exception as e:
                print "Problem loading parameters in ",self.paramFile,". Check all parameter instances."
                print e
                return None
        return paramDicts

    def read(self, templateDir):
        """Read in the template and the parameters.
        """
        return self.read_template(templateDir), self.read_params()

    def write(self, outputdir):
        """Write all the template instances to files.

        Parameters
        ----------
        outputdir : string
            The directory where the template instances should be written.
        """
        for idx, args in enumerate(self.params):
            with open(os.path.join(outputdir, self.problemID+"_"+str(idx)+".py"), "wb") as f:
                f.write(self.template.render(args))


class Index(object):
    """An index that catalogues the problems currently ready to be processed.
    Can write itself to a file to give the user useful information about the relative size and
    complexity of the problems and help them to pick the ones to test their solvers on.

    Attributes
    ----------
    problems : dictionary mapping strings to dictionaries
        Maps problemIDs to dictionaries of tags that denote various properties of the problem.
        Fields:
            n_vars : the number of variables in the problem
            n_constants : the number of data values in the problem
            n_constraints : the number of constraints in the problem
            size : TODO - a rough measure of size that will be either "small", "medium", or "large
            complexity : TODO - a measure of solve time (big*small^2)
    problemsDir : string
        The directory from which problems are being read.
    """
    def __init__(self, problemsDir = "problems"):
        """problemsDir defaults to 'problems'"""
        problemsDir = os.path.realpath(os.path.abspath(os.path.join(os.path.split(inspect.getfile(inspect.currentframe()))[0], problemsDir)))
        if problemsDir not in sys.path:
            sys.path.insert(0, problemsDir)

        self.problemsDir = problemsDir
        self.problems = self.read_problems(problemsDir)

    def read_problems(self, problemsDir):
        """Reads the python files in problemsDir, extracts the problems, and appends them to 
        self.problems. Each problem should be named "prob" in the corresponding python file.

        Parameters
        ----------
        problemsDir : string
            The name of the directory that contains the problems we want to read. 
        """
        problems = {}

        # Might need to be parallelized:
        for dirname, dirnames, filenames in os.walk(problemsDir):
            for filename in filenames:
                # print filename
                if filename[-3:] == ".py" and filename != "__init__.py":
                    problemID = filename[0:-3]
                    problem = (__import__(problemID).prob)
                    problems[problemID] = Index.compute_problem_stats(problem)
        return problems

    @classmethod
    def compute_problem_stats(self, problem):
        """Computes the values of the tags required to catagorize this problem.

        Parameters
        ----------
        problem : cvxpy.Problem
            The problem we want to analyze.
        """
        stats = {}

        # n_vars
        stats["n_vars"] = problem.n_variables()

        # n_constants
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
        stats["n_constants"] = n_data

        # n_constraints
        stats["n_constraints"] = problem.n_eq() + problem.n_leq()

        return stats


    def write(self, filename = "index.txt"):
        """Writes descriptions of the problems in self.problems to a text file.

        Parameters
        ----------
        filename : string
            The name of the file to write the index to. Defaults to "index.txt".

        Format of the index file:
        For each problem:
        <problem name>
            n_vars: <number of variables>
            n_constants: <number of constants>
            n_constraints: <number of constraints>
        """
        with open(filename, "w") as f:
            for problemID in self.problems:
                out = problemID + "\n" + \
                "\tn_vars: " + str(self.problems["n_vars"]) + "\n" + \
                "\tn_constants: " + str(self.problems["n_constants"]) + "\n" + \
                "\tn_constraints: " + str(self.problems["n_constraints"]) + "\n"
                f.write(out)
