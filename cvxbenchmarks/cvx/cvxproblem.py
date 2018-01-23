from cvxbenchmarks.base import Problem
import cvxbenchmarks.settings as s
from warnings import warn

# Constraint types
from cvxpy.constraints.semidefinite import SDP
from cvxpy.constraints.exponential import ExpCone
from cvxpy.constraints.second_order import SOC
from cvxpy.constraints.bool_constr import BoolConstr

import sys

class CVXProblem(Problem):
    """
    Problem class capable of reading a cvxbenchmarks problem 
    in a .py file. This class expects the .py file to contain
    a list of problem dictionaries. For example, 

    problemDict = {
      "problemID" : problemID,
      "problem"   : prob,
      "opt_val"   : opt_val
    }
    problems = [problemDict]

    Attributes
    ----------
    problemID : string
        A unique identifier for this problem.
    problem : cvxpy.Problem
        The cvxpy problem to be solved.
    opt_val : numeric type
        the optimal value of the problem
    metadata : dict
        A dict mapping metadata types to their values. Possible
        types include:
        - cone_types: A set of solver capabilities required
                      to solve this problem.
                      Possible Values: {LP, SOCP, SDP, EXP, MIP} 
        - Also include all the things from cvxpy.SizeMetrics, for example:
            - num_variables: The number of variables in the problem.
            - num_leq_constraints: The number of scalar inequality constraints.
            - num_eq_constraints: The number of scalar equality constraints.
            - etc.

    """

    def __init__(self, problemID, problem, opt_val=None, metadata=None):
        self.problemID = problemID
        self.problem = problem
        self.opt_val = opt_val
        if metadata == None:
            self.metadata = {"cone_types": self.get_cone_types(self.problem)}
            # print(self.problem.size_metrics.__dict__)
            self.metadata.update(self.problem.size_metrics.__dict__)
        else: # pragma: no cover
            self.metadata = metadata

    @classmethod
    def read(cls, problemID, problemDir):
        """Loads a file with name <problemID>.py and returns a list of
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
        a list of CVXProblem objects.
        """

        # Load the module
        if problemDir not in sys.path:
            sys.path.insert(0, problemDir)
        try:
            problemModule = __import__(problemID)
        except Exception as e: # pragma: no cover
            warn("Could not import file " + problemID)
            print(e)
            return []

        foundProblems = [] # Holds the problems we find in this file

        # Look for a dictionary
        PROBLEM_LIST = ["problems"]
        for problemList in PROBLEM_LIST:
            if problemList in [name for name in dir(problemModule)]:
                problems = getattr(problemModule, "problems")
                for problemDict in problems:
                    foundProblems.append(CVXProblem(**problemDict))
        if len(foundProblems) == 0: # pragma: no cover
            warn(problemID + " contains no problem objects.")
        return foundProblems

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

    def __repr__(self): # pragma: no cover
        return repr(self.problemID) + " " + repr(self.problem)

