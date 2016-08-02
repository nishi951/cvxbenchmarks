
from src.ProblemGenerator import ProblemTemplate, Index
from src.tests.base_test import BaseTest
import sys, os, inspect
import pandas as pd


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
        ref_max_big_small_squared = 20000
        ref_max_data_dimension = 50
        ref_num_scalar_data = 1071
        ref_num_scalar_eq_constr = 20
        ref_num_scalar_leq_constr = 50
        ref_num_scalar_variables = 50
        
        self.assertEqual(self.index.problems.loc["lp_0","max_big_small_squared"], ref_max_big_small_squared)
        self.assertEqual(self.index.problems.loc["lp_0","max_data_dimension"], ref_max_data_dimension) 
        self.assertEqual(self.index.problems.loc["lp_0","num_scalar_data"], ref_num_scalar_data)
        self.assertEqual(self.index.problems.loc["lp_0","num_scalar_eq_constr"], ref_num_scalar_eq_constr)
        self.assertEqual(self.index.problems.loc["lp_0","num_scalar_leq_constr"], ref_num_scalar_leq_constr)
        self.assertEqual(self.index.problems.loc["lp_0","num_scalar_variables"], ref_num_scalar_variables)




        

