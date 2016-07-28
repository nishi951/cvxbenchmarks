
from src.ProblemGenerator import ProblemTemplate, Index
from src.tests.base_test import BaseTest
import sys, os, inspect


class TestProblemTemplate(BaseTest):
    """Unit tests for the ProblemTemplate class"""
    def setUp(self):
        self.templ = ProblemTemplate("least_squares","tests/templates/least_squares_params.txt", "tests/templates")


    def test_read_template(self):
        """Tests the read_template method.
        """
        template = self.templ.read_template("tests/templates")
        a = template.render(m = 100, n = 100, seed = 1)
        # Read the already-generated version from the problems directory
        b = None
        with open("problems/least_squares_0.py", "r") as f:
            b = f.read()
        self.assertEqual(a, b)

    def test_read_params(self):
        """Tests the read_params method.
        """
        params = self.templ.read_params()
        ref = [{"m": "100", "n": "100", "seed": "1"},
               {"m": "30", "n": "20", "seed": "1"},
               {"m": "300", "n": "200", "seed": "1"} ]
        self.assertEqual(params, ref)


class TestIndex(BaseTest):
    """Unit tests for the Index class"""
    def setUp(self):
        self.problemsDir = "tests/problems"
        self.index = Index(problemsDir = self.problemsDir)

    def test_read_problems(self):
        """Tests the read_problems method.
        """
        # Note that Index.__init__() calls read_problems already:
        ref = {"least_squares_0": {"n_vars": 100,
                                   "n_constants": 10101,
                                   "n_constraints": 0} 
        }
        print self.index
        self.assertEqual(self.index.problems, ref)

    def test_compute_problem_stats(self):
        """Tests the compute_problem_stats class method.
        """
        ref = {}
        cmd_folder = os.path.realpath(os.path.abspath(os.path.join(os.path.split(inspect.getfile(inspect.currentframe()))[0], self.problemsDir)))
        if cmd_folder not in sys.path:
            sys.path.insert(0, cmd_folder)
        problemID = "least_squares_0"
        problem = (__import__(problemID).prob)
        # print "problem:",problem
        ans = Index.compute_problem_stats(problem)
        ref = {'n_vars': 100, 'n_constraints': 0, 'n_constants': 10101}
        self.assertEqual(ans, ref)

    def test_write(self):
        """Tests the write method.
        """
        self.index.write()



        

