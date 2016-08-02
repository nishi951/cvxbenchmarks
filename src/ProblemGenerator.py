
from jinja2 import Environment, FileSystemLoader
import re, os, sys, inspect
import numpy as np
import pandas as pd

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
        Ex. If the template is stored in the file "least_squares.j2" in the folder
        "templates", then self.problemID should be "least_squares" and templateDir should be "templates".

        Parameters
        ----------
        templateDir - The 

        Returns
        -------
        temp : jinja2.Template
            The template stored in the file.
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

        The format of the parameter files is:
        - One tuple at the beginning containing the names of the parameters.
        - After that, one tuple for each problem instance that the person wants to generate.

        Example:
        (m, n, seed) = (20, 30, 1), (3, 300, 1), (4, 4000, 1)

        will generate three instances of a template with fields "m", "n" and "seed" with the
        corresponding values (20, 30, 1), (3, 300, 1), and (4, 4000, 1)

        Returns
        -------
        paramDicts : dictionary
            The parameters stored in the file.
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
    problems : pandas.DataFrame
        Index = 

    problemsDir : string
        The directory from which problems are being read.
    """
    def __init__(self, problemsDir = "problems"):
        """Initialize the index by reading in all the problems in problemsDir.
        problemsDir defaults to 'problems'
        """
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

        Returns
        -------
        problems : pandas.DataFrame
            The problems in problemsDir, organized by problemID.
        """
        problem_list = []
        # Might need to be parallelized:
        for dirname, dirnames, filenames in os.walk(problemsDir):
            for filename in filenames:
                # print filename
                if filename[-3:] == ".py" and filename != "__init__.py":
                    problemID = filename[0:-3]
                    print problemID
                    problem = (__import__(problemID).prob)
                    next = pd.Series(problem.size_metrics.__dict__, name = problemID)
                    problem_list.append(next)
        problems = pd.DataFrame(problem_list)
        return problems

    def write(self, filename = "index.txt"):
        """Writes descriptions of the problems in self.problems to a text file.

        Parameters
        ----------
        filename : string
            The name of the file to write the index to. Defaults to "index.txt".
        """
        from tabulate import tabulate
        with open(filename, "w") as f:
            f.write(tabulate(self.problems, headers = "keys", tablefmt = "psql"))



