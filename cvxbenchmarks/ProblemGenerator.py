
from jinja2 import Environment, FileSystemLoader
import re, os, sys, inspect
import numpy as np
import pandas as pd

# cvxfolder = os.path.realpath(os.path.abspath(os.path.join(os.path.split(inspect.getfile(inspect.currentframe()))[0],"cvxpy")))
# if cvxfolder not in sys.path:
    # sys.path.insert(0, cvxfolder) 
sys.path.insert(0, "/Users/mark/Documents/Stanford/reu2016/cvxpy")
import cvxpy as cvx
print cvx

# Format of the Parameter file:
# Csv file readable by pandas
# ex.
# problemID, m, n, seed
# "ls_0", 20, 30, 1
# "ls_1", 40, 60, 1
# "ls_big" 500, 600, 1


class ProblemTemplate(object):
    """A templated version of a cvxpy problem that can be read and written.

    Attributes
    ----------
    IO Stuff:
        templateFile : string
            The actual file of the template. e.g. "lib/cvxbenchmarks/least_squares.j2"
        paramFile : string
            The file path to the file containing the parameters for the instances of this template.
            e.g. "lib/cvxbenchmarks/least_squares_params.txt"
        templateDir : string
            The file containing the templateFile (necessary for jinja2 to find the template)

    Generated:
        template : jinja2.template
            The problem template that can be rendered to produce the problem instances, after filling
            the parameters specified in params.
        params : pandas.DataFrame
            Each row of the dataframe represents an instance of the problem template.
            The data in the rows corresponds to the fields in the template that 
            need to be filled in.
    """

    def __init__(self, templateFile, paramFile, templateDir = os.path.join("lib", "cvxbenchmarks")):
        self.templateFile = templateFile
        self.paramFile = paramFile
        self.templateDir = templateDir
        self.template, self.params = self.read()

    def read_template(self):
        """Read in the template.
        Ex. If the template is stored in the file "least_squares.j2" in the folder
        "templates", then self.problemID should be "least_squares" and templateDir should be "templates".

        Returns
        -------
        temp : jinja2.Template
            The template stored in the file.
        """
        temp = None
        try:
            env = Environment(loader = FileSystemLoader(self.templateDir))
            temp = env.get_template(self.templateFile)
            return temp
        except Exception as e:
            print "Problem locating template",self.templateFile,"in",self.templateDir+". Check template file path."
            print e
            return temp

    def read_params(self):
        """Read in the parameter file csv.
        """
        self.params = None
        try:
            self.params = pd.read_csv(self.paramFile, skipinitialspace=True)
        except Exception as e:
            print "Problem loading parameters in",self.paramFile,". Check parameter file path."
            print e
        return self.params

    def read(self):
        """Read in the template and the parameters.
        """
        return self.read_template(), self.read_params()

    def write_to_dir(self, outputdir):
        """Write all the template instances to files.

        Parameters
        ----------
        outputdir : string
            The directory where the template instances should be written.
        """
        print self.params
        for idx, row in self.params.iterrows():
            instanceID = row["problemID"]
            with open(os.path.join(outputdir, instanceID + ".py"), "wb") as f:
                f.write(self.template.render(row.to_dict()))


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



