
import re, os, sys, inspect
import numpy as np
import pandas as pd
from cvxbenchmarks.framework import TestProblem

class Index(object):
    """An index that catalogues the problems currently ready to be processed.
    Can write itself to a file to give the user useful information about 
    the relative size and complexity of the problems and help them to 
    pick the ones to test their solvers on.

    Attributes
    ----------
    problems : pandas.DataFrame

    problemDir : string
        The directory from which problems are being read.
    """
    def __init__(self, problemDir = "problems"):
        """Initialize the index by reading in all the problems in problemDir.
        problemDir defaults to 'problems'
        """
        # problemDir = \
        # os.path.realpath(
        #     os.path.abspath(
        #         os.path.join(
        #             os.path.split(
        #                 inspect.getfile(inspect.currentframe()))[0], problemDir)))
        problemDir = os.path.realpath(os.path.abspath(problemDir))
        if problemDir not in sys.path:
            sys.path.insert(0, problemDir)
        print("Index: reading from {}".format(problemDir))
        print("Found problems:")
        self.problemDir = problemDir
        self.problems = self.read_problems(problemDir)

    def read_problems(self, problemDir):
        """Reads the python files in problemDir, extracts the problems, 
        and appends them to self.problems. Each problem should be named 
        "prob" in the corresponding python file.

        Parameters
        ----------
        problemDir : string
            The name of the directory that contains the problems we want to read.

        Returns
        -------
        problems : pandas.DataFrame
            The problems in problemDir, organized by problemID.
        """
        problem_list = []
        # Might need to be parallelized:
        for dirname, dirnames, filenames in os.walk(problemDir):
            for filename in filenames:
                # print filename
                if filename[-3:] == ".py" and filename != "__init__.py":
                    problemID = filename[0:-3]
                    print("\t{}".format(problemID))
                    problems = TestProblem.from_file(problemID, problemDir)
                    for testproblem in problems:
                        next = pd.Series(testproblem.problem.size_metrics.__dict__, name = problemID)
                        # Add cone types
                        next.loc["tags"] = TestProblem.check_cone_types(testproblem.problem)
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

    def write_latex(self,
                    keys = ("num_scalar_variables",
                            "num_scalar_eq_constr",
                            "num_scalar_leq_constr",
                            "tags"),
                    filename = "index.tex"):
        """Writes a latex tabular snippet suitable for posting in a latex document.

        Parameters
        ----------
        keys : list
            A list of the keys to put in the table. Defaults to:
            ["num_scalar_variables", "num_scalar_eq_constr", "num_scalar_neq_constr"]
        filename : string
            The name of the file to write the latex to. Defaults to "index.tex".
        """
        with open(filename, "w") as f:
            f.write("\\begin{figure}[h]\n")
            f.write("\\centering\n")
            f.write("\\begin{tabular}{|" + (" c |" * (len(keys) + 1)) + "}\n")
            # Header
            f.write("\\hline \n")
            header = ["ProblemID"] + keys
            f.write(' & '.join(header) + "\\\\ \n")
            f.write("\\hline \n")
            for problem in self.problems.index:
                values = self.problems.loc[problem, keys].tolist()
                values = [str(problem)] + [str(value) for value in values]
                f.write(' & '.join(values) + " \\\\ \n")
                f.write("\\hline \n")
            f.write("\\end{tabular}\n")
            f.write("\\end{figure}\n")
